#! /bin/bash


apt update &&  apt install -yqq nano vim git tmux htop protobuf-compiler zsh net-tools 
git clone https://github.com/sfc-gh-juyang/ArcticInference.git
cd ArcticInference
git checkout juncheng/grpc

curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH=$HOME/.local/bin:$PATH
echo "PATH=$HOME/.local/bin:$PATH" >> ~/.bashrc
uv venv --python 3.12 --seed
source .venv/bin/activate
uv pip install -U grpcio grpcio-tools protobuf grpcio-reflection hf_transfer aiohttp
uv pip install git+https://github.com/huggingface/transformers.git


python arctic_inference/grpc/generate_proto.py

/usr/local/bin/text-embeddings-router --model-id "BAAI/bge-base-en-v1.5" --port 8080 > engine.log 2>&1 &
echo "waiting for engine to start..."
sleep 20

python benchmark/benchmark_http.py --server http://localhost:8080 --backend TEI --requests 1280 --prompt-length 512 --batch-size 1,8,16,32 --concurrency 16
python benchmark/benchmark_http.py --server http://localhost:8080 --backend TEI --requests 12800 --prompt-length 50 --batch-size 1,16,32 --concurrency 256


