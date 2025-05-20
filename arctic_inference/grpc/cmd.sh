#! /bin/bash
export PYTHONPATH=/home/yak/vllm/:$PYTHONPATH

model="BAAI/bge-base-en-v1.5"

python arctic_inference/grpc/generate_proto.py

python arctic_inference/grpc/replica.py --model "BAAI/bge-base-en-v1.5"

python arctic_inference/grpc/client.py --prompt "hello world!"

python arctic_inference/grpc/replica_manager.py --model "BAAI/bge-base-en-v1.5" --num-replicas 4

uv pip install grpcio grpcio-tools protobuf grpcio-reflection

# long sequence
bash benchmark/run_benchmark.sh BAAI/bge-base-en-v1.5 1024 512 16 fixed 1,16 4
# short sequence H200
bash benchmark/run_benchmark.sh BAAI/bge-base-en-v1.5 4096 50 1024 fixed 1,64 32
# short sequence L40
bash benchmark/run_benchmark.sh BAAI/bge-base-en-v1.5 4096 50 256 fixed 1,16,64 8
# short sequence A10
# bash benchmark/run_benchmark.sh BAAI/bge-base-en-v1.5 4096 50 256 fixed 1,64 2


python arctic_inference/grpc/replica_manager.py --model "BAAI/bge-base-en-v1.5" --num-replicas 4

python benchmark/benchmark.py --server localhost:50050 --batch-sizes 1,4,16,64 --requests 1280 --concurrency 16 --prompt-length 512
python benchmark/benchmark.py --server localhost:50050 --batch-sizes 1,4,16,64 --requests 12800 --concurrency 1024 --prompt-length 50

vllm serve BAAI/bge-base-en-v1.5
python benchmark/benchmark_http.py --endpoint v1/embeddings --no-health-check --requests 1280 --prompt-length 512 --concurrency 16
/usr/local/bin/text-embeddings-router --model-id "BAAI/bge-base-en-v1.5" --port 8080
python benchmark/benchmark_http.py --server http://localhost:8080 --backend TEI --requests 1280 --prompt-length 512 --batch-size 1,16 --concurrency 8
python benchmark/benchmark_http.py --server http://localhost:8080 --backend TEI --requests 4096 --prompt-length 50 --batch-size 1,16,32 --concurrency 256
