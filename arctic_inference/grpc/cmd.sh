#! /bin/bash


export PYTHONPATH=/home/yak/vllm/:$PYTHONPATH

python arctic_inference/grpc/generate_proto.py

python arctic_inference/grpc/replica_v0.py --model "BAAI/bge-base-en-v1.5"

python arctic_inference/grpc/client.py --prompt "hello world!"

python arctic_inference/grpc/replica_manager.py --model "BAAI/bge-base-en-v1.5" --num-replicas 4

uv pip install grpcio grpcio-tools protobuf grpcio-reflection



python arctic_inference/grpc/benchmark.py --server localhost:50050 --batch-sizes 1,4,16,64 --requests 1280 --concurrency 8 --prompt-length 508 --server localhost:50050

vllm serve BAAI/bge-base-en-v1.5
python arctic_inference/grpc/benchmark_http.py --endpoint v1/embeddings --no-health-check --requests 1280 --prompt-length 508 --concurrency 8
