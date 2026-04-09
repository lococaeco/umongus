#!/usr/bin/env bash
# Start a local OpenAI-compatible vLLM server for AmongUs game agents.
#
# Environment variables:
#   MODEL           HuggingFace model id or local path (default: Qwen/Qwen2.5-32B-Instruct)
#   PORT            HTTP port to bind (default: 8000)
#   TP              tensor-parallel size (default: 1)
#   DP              data-parallel size / replicas (default: 1)
#   GPU             legacy: CUDA_VISIBLE_DEVICES if that env is not already set
#   MAX_LEN         max model context length (default: 16384)
#   GMEM            gpu-memory-utilization (default: 0.8)
#
# Examples:
#   # Single 32B on GPU 0
#   CUDA_VISIBLE_DEVICES=0 MODEL=Qwen/Qwen2.5-32B-Instruct ./run_vllm_server.sh
#
#   # 8 replicas of 32B, one per GPU (8x throughput)
#   CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 DP=8 ./run_vllm_server.sh
#
#   # Llama-3.3-70B with TP=2, 4 replicas (uses 8 GPUs)
#   CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 MODEL=meta-llama/Llama-3.3-70B-Instruct TP=2 DP=4 ./run_vllm_server.sh
set -e
cd "$(dirname "$0")"
source .venv/bin/activate

MODEL=${MODEL:-Qwen/Qwen2.5-32B-Instruct}
PORT=${PORT:-8000}
TP=${TP:-1}
DP=${DP:-1}
GPU=${GPU:-0}
MAX_LEN=${MAX_LEN:-16384}
GMEM=${GMEM:-0.8}

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-$GPU}

echo "[run_vllm_server] MODEL=$MODEL PORT=$PORT TP=$TP DP=$DP CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

exec vllm serve "$MODEL" \
    --port "$PORT" \
    --tensor-parallel-size "$TP" \
    --data-parallel-size "$DP" \
    --max-model-len "$MAX_LEN" \
    --gpu-memory-utilization "$GMEM" \
    --dtype bfloat16
