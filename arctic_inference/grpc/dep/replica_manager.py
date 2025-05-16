#!/usr/bin/env python3

import os
import sys
import asyncio
import logging
import random
import time
import argparse
import signal
import subprocess
from dataclasses import dataclass
from enum import Enum
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import grpc
from grpc import aio
import json
import uuid
import threading
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import arctic_inference.grpc.proto.python.inference_pb2 as inference_pb2
import arctic_inference.grpc.proto.python.inference_pb2_grpc as inference_pb2_grpc

logger = logging.getLogger("arctic_inference.grpc.replica_manager")


class LoadBalancingPolicy(Enum):
    ROUND_ROBIN = "round_robin"
    RANDOM = "random"
    LEAST_LOADED = "least_loaded"


@dataclass
class ReplicaConfig:
    host: str
    port: int
    model: str
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.9
    max_model_len: Optional[int] = None
    gpu_ids: Optional[List[int]] = None


class ReplicaState:
    """Tracks the state of a replica."""

    def __init__(self, config: ReplicaConfig, replica_id: int):
        self.config = config
        self.replica_id = replica_id
        self.process: Optional[subprocess.Popen] = None
        self.healthy = False
        self.ready = False
        self.last_checked = 0
        self.active_requests = 0
        self.total_requests = 0
        self.stub = None
        self.channel = None
        self.address = f"{config.host}:{config.port}"
        self.error_count = 0
        self.info = {}

    def __str__(self):
        return f"Replica {self.replica_id} ({self.address}), healthy: {self.healthy}, ready: {self.ready}, active: {self.active_requests}"


class ReplicaManager:
    """Manages multiple vLLM replicas for inference."""

    def __init__(
        self,
        replica_configs: List[ReplicaConfig],
        load_balancing_policy: LoadBalancingPolicy = LoadBalancingPolicy.ROUND_ROBIN,
        health_check_interval: float = 5.0,
        replica_startup_timeout: float = 120.0,
        max_retries: int = 3,
    ):
        self.replica_configs = replica_configs
        self.replicas: List[ReplicaState] = []
        self.policy = load_balancing_policy
        self.health_check_interval = health_check_interval
        self.replica_startup_timeout = replica_startup_timeout
        self.max_retries = max_retries
        self.next_replica_idx = 0
        self.lock = asyncio.Lock()
        self.is_running = False
        self.health_check_task = None

    async def start(self):
        """Start all replicas and initialize connections."""
        logger.info(
            f"Starting replica manager with {len(self.replica_configs)} replicas"
        )
        self.is_running = True

        # Initialize replica states
        for i, config in enumerate(self.replica_configs):
            replica = ReplicaState(config, i)
            self.replicas.append(replica)

        # Start replicas in parallel
        startup_tasks = [self._start_replica(replica) for replica in self.replicas]
        await asyncio.gather(*startup_tasks)

        # Start health checking
        self.health_check_task = asyncio.create_task(self._health_check_loop())

        logger.info(
            f"Replica manager started with {sum(1 for r in self.replicas if r.ready)} ready replicas"
        )

    async def stop(self):
        """Stop all replicas and cleanup connections."""
        logger.info("Stopping replica manager")
        self.is_running = False

        if self.health_check_task:
            self.health_check_task.cancel()
            try:
                await self.health_check_task
            except asyncio.CancelledError:
                pass

        # Stop all replicas
        for replica in self.replicas:
            await self._stop_replica(replica)

        logger.info("Replica manager stopped")

    async def _start_replica(self, replica: ReplicaState):
        """Start a single replica process."""
        config = replica.config
        logger.info(f"Starting replica {replica.replica_id} at {replica.address}")

        # Build command
        cmd = [
            "python",
            "-m",
            "arctic_inference.grpc.replica",
            "--model",
            config.model,
            "--host",
            config.host,
            "--port",
            str(config.port),
            "--tensor-parallel-size",
            str(config.tensor_parallel_size),
            "--gpu-memory-utilization",
            str(config.gpu_memory_utilization),
        ]

        if config.max_model_len:
            cmd.extend(["--max-model-len", str(config.max_model_len)])

        if config.gpu_ids:
            gpu_list = ",".join(map(str, config.gpu_ids))
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = gpu_list
        else:
            env = os.environ.copy()

        # Start the process
        try:
            replica.process = subprocess.Popen(
                cmd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )

            # Start log monitoring threads
            threading.Thread(
                target=self._monitor_process_output,
                args=(replica, replica.process.stdout, "stdout"),
                daemon=True,
            ).start()

            threading.Thread(
                target=self._monitor_process_output,
                args=(replica, replica.process.stderr, "stderr"),
                daemon=True,
            ).start()

            # Wait for startup
            start_time = time.time()
            logger.info("waiting for replica to be ready")
            while time.time() - start_time < self.replica_startup_timeout:
                if await self._check_replica_health(replica, print_log=False):
                    print(".", end="")
                    # logger.info(f"Replica {replica.replica_id} is healthy and ready")
                    return
                await asyncio.sleep(2)

            logger.error(f"Timeout waiting for replica {replica.replica_id} to start")
            await self._stop_replica(replica)

        except Exception as e:
            logger.exception(f"Error starting replica {replica.replica_id}: {e}")
            await self._stop_replica(replica)

    def _monitor_process_output(self, replica: ReplicaState, pipe, name):
        """Monitor process output in a separate thread."""
        for line in pipe:
            line = line.strip()
            print(f"Replica {replica.replica_id} {name}: {line}")

    async def _stop_replica(self, replica: ReplicaState):
        """Stop a replica process and cleanup."""
        if replica.channel:
            await replica.channel.close()
            replica.channel = None
            replica.stub = None

        if replica.process:
            print(f"Stopping replica {replica.replica_id}")
            try:
                replica.process.terminate()
                # Wait for a short time for graceful shutdown
                await asyncio.sleep(2)
                if replica.process.poll() is None:
                    # Force kill if still running
                    replica.process.kill()
            except Exception as e:
                logger.error(f"Error stopping replica {replica.replica_id}: {e}")

            replica.process = None

        replica.healthy = False
        replica.ready = False

    async def wait_for_replica_ready(self, replica: ReplicaState):
        while not replica.ready:
            await asyncio.sleep(1)

    async def _check_replica_health(
        self, replica: ReplicaState, print_log: bool = True
    ) -> bool:
        """Check if a replica is healthy and ready for serving."""
        if not replica.channel:
            # Create channel and stub if not exists
            try:
                replica.channel = aio.insecure_channel(
                    replica.address,
                    options=[
                        ("grpc.max_send_message_length", 100 * 1024 * 1024),
                        ("grpc.max_receive_message_length", 100 * 1024 * 1024),
                        ("grpc.keepalive_time_ms", 30000),
                    ],
                )
                replica.stub = inference_pb2_grpc.InferenceServiceStub(replica.channel)
            except Exception as e:
                logger.error(
                    f"Error creating channel for replica {replica.replica_id}: {e}"
                )
                return False

        try:
            # Check health
            health_response = await replica.stub.HealthCheck(
                inference_pb2.HealthCheckRequest(), timeout=5
            )

            # Get replica info
            info_response = await replica.stub.GetReplicaInfo(
                inference_pb2.ReplicaInfoRequest(), timeout=5
            )

            replica.healthy = health_response.healthy
            replica.ready = info_response.ready
            replica.info = {
                "model_name": info_response.model_name,
                "task": info_response.task,
                "dtype": info_response.dtype,
            }

            replica.last_checked = time.time()
            replica.error_count = 0

            return replica.healthy and replica.ready

        except Exception as e:
            if print_log:
                logger.error(
                    f"Health check failed for replica {replica.replica_id}: {e}"
                )
            replica.error_count += 1

            # Reset connection after multiple failures
            if replica.error_count >= 3:
                if replica.channel:
                    await replica.channel.close()
                    replica.channel = None
                    replica.stub = None

            return False

    async def _health_check_loop(self):
        """Periodically check the health of all replicas."""
        while self.is_running:
            for replica in self.replicas:
                try:
                    await self._check_replica_health(replica)
                except Exception as e:
                    logger.error(
                        f"Error checking health of replica {replica.replica_id}: {e}"
                    )

            # Log status
            healthy_count = sum(1 for r in self.replicas if r.healthy)
            ready_count = sum(1 for r in self.replicas if r.ready)
            logger.info(
                f"Health check: {healthy_count}/{len(self.replicas)} healthy, {ready_count}/{len(self.replicas)} ready"
            )

            await asyncio.sleep(self.health_check_interval)

    async def _select_replica(self) -> Optional[ReplicaState]:
        """Select a replica based on the load balancing policy."""
        async with self.lock:
            available_replicas = [r for r in self.replicas if r.ready]
            if not available_replicas:
                return None

            if self.policy == LoadBalancingPolicy.ROUND_ROBIN:
                # Simple round-robin
                replica = available_replicas[
                    self.next_replica_idx % len(available_replicas)
                ]
                self.next_replica_idx = (self.next_replica_idx + 1) % len(
                    available_replicas
                )
                return replica

            elif self.policy == LoadBalancingPolicy.RANDOM:
                # Random selection
                return random.choice(available_replicas)

            elif self.policy == LoadBalancingPolicy.LEAST_LOADED:
                # Select replica with fewest active requests
                return min(available_replicas, key=lambda r: r.active_requests)

    async def encode(
        self, request: inference_pb2.EncodeRequest
    ) -> inference_pb2.EncodeResponse:
        """Encode a request by routing it to a selected replica."""
        retry_count = 0
        request_id = request.request_id or str(uuid.uuid4())

        while retry_count < self.max_retries:
            replica = await self._select_replica()
            if not replica:
                if retry_count < self.max_retries - 1:
                    logger.warning(
                        f"No replicas available, retrying in 1s... ({retry_count + 1}/{self.max_retries})"
                    )
                    await asyncio.sleep(1)
                    retry_count += 1
                    continue
                else:
                    return inference_pb2.EncodeResponse(
                        request_id=request_id,
                        error="No replicas available for processing",
                    )

            try:
                # Track request stats
                replica.active_requests += 1
                replica.total_requests += 1

                # Set request_id if not provided
                if not request.request_id:
                    request.request_id = request_id

                # Forward the request to the selected replica
                response = await replica.stub.Encode(request)
                return response

            except Exception as e:
                logger.error(
                    f"Error from replica {replica.replica_id} for request {request_id}: {e}"
                )
                retry_count += 1

            finally:
                replica.active_requests -= 1

        # All retries failed
        return inference_pb2.EncodeResponse(
            request_id=request_id,
            error=f"Failed after {self.max_retries} retries",
        )

    async def generate(
        self, request: inference_pb2.GenerateRequest
    ) -> List[inference_pb2.GenerateResponse]:
        """Generate completions by routing to a selected replica."""
        retry_count = 0
        request_id = request.request_id or str(uuid.uuid4())

        while retry_count < self.max_retries:
            replica = await self._select_replica()
            if not replica:
                if retry_count < self.max_retries - 1:
                    logger.warning(
                        f"No replicas available, retrying in 1s... ({retry_count + 1}/{self.max_retries})"
                    )
                    await asyncio.sleep(1)
                    retry_count += 1
                    continue
                else:
                    return [
                        inference_pb2.GenerateResponse(
                            request_id=request_id,
                            text="",
                            finished=True,
                            token_ids=[],
                            error="No replicas available for processing",
                        )
                    ]

            try:
                # Track request stats
                replica.active_requests += 1
                replica.total_requests += 1

                # Set request_id if not provided
                if not request.request_id:
                    request.request_id = request_id

                # Forward the request to the selected replica
                responses = []
                async for response in replica.stub.Generate(request):
                    responses.append(response)

                return responses

            except Exception as e:
                logger.error(
                    f"Error from replica {replica.replica_id} for request {request_id}: {e}"
                )
                retry_count += 1

            finally:
                replica.active_requests -= 1

        # All retries failed
        return [
            inference_pb2.GenerateResponse(
                request_id=request_id,
                text="",
                finished=True,
                token_ids=[],
                error=f"Failed after {self.max_retries} retries",
            )
        ]

    async def abort(self, request_id: str) -> bool:
        """Abort a request on all replicas."""
        abort_request = inference_pb2.AbortRequest(request_id=request_id)

        # Try to abort on all replicas since we don't know which one is processing it
        aborted = False
        for replica in self.replicas:
            if not replica.ready or not replica.stub:
                continue

            try:
                response = await replica.stub.Abort(abort_request)
                if response.success:
                    aborted = True
            except Exception as e:
                logger.error(
                    f"Error aborting request {request_id} on replica {replica.replica_id}: {e}"
                )

        return aborted

    def get_stats(self) -> Dict:
        """Get statistics about all replicas."""
        stats = {
            "total_replicas": len(self.replicas),
            "healthy_replicas": sum(1 for r in self.replicas if r.healthy),
            "ready_replicas": sum(1 for r in self.replicas if r.ready),
            "total_requests": sum(r.total_requests for r in self.replicas),
            "active_requests": sum(r.active_requests for r in self.replicas),
            "replicas": [
                {
                    "id": r.replica_id,
                    "address": r.address,
                    "healthy": r.healthy,
                    "ready": r.ready,
                    "active_requests": r.active_requests,
                    "total_requests": r.total_requests,
                    "info": r.info,
                }
                for r in self.replicas
            ],
        }
        return stats


class ReplicaManagerServer(inference_pb2_grpc.InferenceServiceServicer):
    """gRPC server that uses ReplicaManager to handle requests."""

    def __init__(self, replica_manager: ReplicaManager):
        self.replica_manager = replica_manager

    async def Encode(
        self, request: inference_pb2.EncodeRequest, context: grpc.ServicerContext
    ) -> inference_pb2.EncodeResponse:
        return await self.replica_manager.encode(request)

    async def Generate(
        self, request: inference_pb2.GenerateRequest, context: grpc.ServicerContext
    ):
        async for response in self.replica_manager.generate(request):
            yield response

    async def Abort(
        self, request: inference_pb2.AbortRequest, context: grpc.ServicerContext
    ) -> inference_pb2.AbortResponse:
        success = await self.replica_manager.abort(request.request_id)
        return inference_pb2.AbortResponse(
            success=success,
            message="Request aborted" if success else "Failed to abort request",
        )

    async def GetReplicaInfo(
        self, request: inference_pb2.ReplicaInfoRequest, context: grpc.ServicerContext
    ) -> inference_pb2.ReplicaInfoResponse:
        stats = self.replica_manager.get_stats()

        # Get info from the first ready replica or use default values
        ready_replicas = [r for r in self.replica_manager.replicas if r.ready]
        if ready_replicas:
            replica = ready_replicas[0]
            model_name = replica.info.get("model_name", "")
            task = replica.info.get("task", "")
            dtype = replica.info.get("dtype", "")
        else:
            model_name = ""
            task = ""
            dtype = ""

        return inference_pb2.ReplicaInfoResponse(
            model_name=model_name,
            task=task,
            dtype=dtype,
            ready=bool(ready_replicas),
            parallel_config="",
            decoding_config="",
            scheduler_config="",
            lora_config="",
        )

    async def HealthCheck(
        self, request: inference_pb2.HealthCheckRequest, context: grpc.ServicerContext
    ) -> inference_pb2.HealthCheckResponse:
        stats = self.replica_manager.get_stats()
        healthy = stats["ready_replicas"] > 0

        return inference_pb2.HealthCheckResponse(
            healthy=healthy,
            message=f"{stats['ready_replicas']}/{stats['total_replicas']} replicas ready",
        )


async def serve_manager(args):
    """Start the replica manager server."""
    # Create replica configs
    replica_configs = []
    for i, port in enumerate(
        range(args.start_port, args.start_port + args.num_replicas)
    ):
        gpu_ids = None
        if args.gpu_assignment == "dedicated":
            gpu_ids = [i % args.num_gpus]
        elif args.gpu_assignment == "shared":
            gpu_ids = list(range(args.num_gpus))

        config = ReplicaConfig(
            host=args.replica_host,
            port=port,
            model=args.model,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            gpu_ids=gpu_ids,
        )
        replica_configs.append(config)

    # Create policy
    if args.load_balancing == "round_robin":
        policy = LoadBalancingPolicy.ROUND_ROBIN
    elif args.load_balancing == "random":
        policy = LoadBalancingPolicy.RANDOM
    elif args.load_balancing == "least_loaded":
        policy = LoadBalancingPolicy.LEAST_LOADED
    else:
        policy = LoadBalancingPolicy.ROUND_ROBIN

    # Create and start replica manager
    replica_manager = ReplicaManager(
        replica_configs=replica_configs,
        load_balancing_policy=policy,
        health_check_interval=args.health_check_interval,
        replica_startup_timeout=args.replica_startup_timeout,
    )

    # Start the manager
    await replica_manager.start()

    # Create and start gRPC server
    server = aio.server(
        futures.ThreadPoolExecutor(max_workers=args.max_workers),
        options=[
            ("grpc.max_send_message_length", 100 * 1024 * 1024),
            ("grpc.max_receive_message_length", 100 * 1024 * 1024),
            ("grpc.keepalive_time_ms", 30000),
        ],
    )

    servicer = ReplicaManagerServer(replica_manager)
    inference_pb2_grpc.add_InferenceServiceServicer_to_server(servicer, server)

    server.add_insecure_port(f"{args.host}:{args.port}")
    logger.info(f"Starting replica manager server on {args.host}:{args.port}")

    await server.start()

    # Handle signals for graceful shutdown
    loop = asyncio.get_event_loop()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(
            sig, lambda: asyncio.create_task(shutdown(server, replica_manager))
        )

    try:
        await server.wait_for_termination()
    finally:
        await replica_manager.stop()


async def shutdown(server, replica_manager):
    """Shutdown the server and replica manager gracefully."""
    logger.info("Shutting down...")
    await replica_manager.stop()
    await server.stop(0)


if __name__ == "__main__":
    from concurrent import futures
    from vllm import AsyncEngineArgs

    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(description="vLLM Replica Manager")
    parser = AsyncEngineArgs.add_cli_args(parser)

    parser.add_argument(
        "--host", type=str, default="0.0.0.0", help="Host to bind the manager to"
    )
    parser.add_argument(
        "--port", type=int, default=50050, help="Port to bind the manager to"
    )
    parser.add_argument(
        "--replica-host",
        type=str,
        default="127.0.0.1",
        help="Host for replicas to bind to",
    )
    parser.add_argument(
        "--start-port",
        type=int,
        default=50100,
        help="Starting port number for replicas",
    )
    parser.add_argument(
        "--num-replicas", type=int, default=2, help="Number of replicas to launch"
    )
    parser.add_argument(
        "--num-gpus", type=int, default=1, help="Number of GPUs available"
    )
    parser.add_argument(
        "--gpu-assignment",
        type=str,
        choices=["dedicated", "shared"],
        default="dedicated",
        help="GPU assignment strategy: dedicated (one GPU per replica) or shared (all replicas can use all GPUs)",
    )

    parser.add_argument(
        "--max-workers", type=int, default=10, help="Maximum number of gRPC workers"
    )
    parser.add_argument(
        "--load-balancing",
        type=str,
        choices=["round_robin", "random", "least_loaded"],
        default="least_loaded",
        help="Load balancing policy",
    )
    parser.add_argument(
        "--health-check-interval",
        type=float,
        default=5.0,
        help="Health check interval in seconds",
    )
    parser.add_argument(
        "--replica-startup-timeout",
        type=float,
        default=120.0,
        help="Timeout for replica startup in seconds",
    )

    parser.set_defaults(
        host="0.0.0.0",
        port=50050,
        replica_host="127.0.0.1",
        start_port=50100,
        num_replicas=4,
        health_check_interval=20,
        replica_startup_timeout=120,
        model="BAAI/bge-base-en-v1.5",
    )

    args = parser.parse_args()

    asyncio.run(serve_manager(args))
