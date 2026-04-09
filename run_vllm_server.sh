#!/usr/bin/env bash
# Start a local OpenAI-compatible vLLM server for AmongUs game agents.
#
# Defaults are tuned for an 8x 96GB GPU node running Qwen/Qwen2.5-32B-Instruct
# as 8 data-parallel replicas (one replica per GPU, maximum throughput).
#
# Environment variables:
#   MODEL           HuggingFace model id or local path   (default: Qwen/Qwen2.5-32B-Instruct)
#   PORT            HTTP port to bind                    (default: 8000)
#   TP              tensor-parallel size (shards a single replica across GPUs)  (default: 1)
#   DP              data-parallel size / replicas                               (default: 8)
#   CUDA_VISIBLE_DEVICES  GPUs to use   (default: 0,1,2,3,4,5,6,7)
#   MAX_LEN         max model context length             (default: 16384)
#   GMEM            gpu-memory-utilization               (default: 0.85)
#
# Examples:
#   # Default: 8 replicas of Qwen2.5-32B across all 8 GPUs
#   ./run_vllm_server.sh
#
#   # Smaller model, single GPU (debug / smoke test)
#   CUDA_VISIBLE_DEVICES=0 DP=1 MODEL=Qwen/Qwen2.5-7B-Instruct ./run_vllm_server.sh
#
#   # Llama-3.3-70B with TP=2, 4 replicas (still uses all 8 GPUs)
#   MODEL=meta-llama/Llama-3.3-70B-Instruct TP=2 DP=4 ./run_vllm_server.sh
set -e
cd "$(dirname "$0")"
source .venv/bin/activate

MODEL=${MODEL:-Qwen/Qwen2.5-32B-Instruct}
PORT=${PORT:-8000}
TP=${TP:-1}
DP=${DP:-8}
MAX_LEN=${MAX_LEN:-16384}
GMEM=${GMEM:-0.85}

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}

echo "[run_vllm_server] MODEL=$MODEL PORT=$PORT TP=$TP DP=$DP CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

exec vllm serve "$MODEL" \
    --port "$PORT" \
    --tensor-parallel-size "$TP" \
    --data-parallel-size "$DP" \
    --max-model-len "$MAX_LEN" \
    --gpu-memory-utilization "$GMEM" \
    --dtype bfloat16
