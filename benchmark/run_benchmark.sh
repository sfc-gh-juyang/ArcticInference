#!/bin/bash

MODEL_LIST=(
    "BAAI/bge-base-en-v1.5" # 0.1B
    # "Snowflake/snowflake-arctic-embed-m-v1.5" 0.1B
    # "nomic-ai/nomic-embed-text-v2-moe" 0.3B
    # "Snowflake/snowflake-arctic-embed-m-v2.0" 0.3B
    # "Snowflake/snowflake-arctic-embed-l-v2.0" 0.56B
    
    # "Alibaba-NLP/gte-Qwen2-1.5B-instruct" # 1.5B
    # "Alibaba-NLP/gte-Qwen2-7B-instruct" # 7B
)

CURR_DIR=$(pwd)
FILE_DIR=$(dirname $0)
DISTRIBUTION="uniform"

function run_benchmark_vllm() {
    echo "Running vllm benchmark for $model"
    vllm serve $model --port 8000 > vllm.log &
    pid=$!
    sleep 60
    python ${FILE_DIR}/benchmark_http.py --model $model \
    --server http://localhost:8000 \
    --batch-sizes 1,4,16,64 \
    --requests 1280 \
    --concurrency 64 \
    --prompt-length 508 \
    --distribution $DISTRIBUTION
    kill $pid
    pkill -f vllm
}

function run_benchmark_arctic() {
    echo "Running arctic_inference benchmark for $model"
    python arctic_inference/grpc/replica_manager.py --model $model --num-replicas 4 --port 50050 > arctic.log &
    pid=$!
    sleep 60
    python ${FILE_DIR}/benchmark.py --model $model \
    --server localhost:50050 \
    --batch-sizes 1,4,16,64 \
    --requests 1280 \
    --concurrency 64 \
    --prompt-length 508 \
    --distribution $DISTRIBUTION
    kill $pid
    pkill -f python
}

# Generate gRPC code
pushd ${FILE_DIR}/../ && \
python arctic_inference/grpc/generate_proto.py >> benchmark.log 2>&1 && \
popd;

for model in "${MODEL_LIST[@]}"; do
    run_benchmark_vllm $model
    run_benchmark_arctic $model
done
