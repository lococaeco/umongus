# Running AmongUs benchmark locally with vLLM (uv environment)

This document describes how to stand up the AmongUs deception benchmark entirely
on a single node using local GPUs and a self-hosted [vLLM](https://github.com/vllm-project/vllm)
server instead of OpenRouter. Everything is set up inside a [uv](https://github.com/astral-sh/uv)
virtual environment.

## TL;DR measured numbers

On 1x RTX PRO 6000 Blackwell 96GB, `Qwen/Qwen2.5-32B-Instruct` (bf16, TP=1),
running `main.py --num_games 5` (5 games concurrently via asyncio):

| metric | value |
| --- | --- |
| wall time (5 games, concurrent) | ~1271s (~21 min) |
| avg wall time per game | ~847s (~14 min, measured from first to last LLM call in that game) |
| avg game length | 11.2 game steps |
| avg LLM calls per game | 85.4 |
| total LLM calls | 427 |
| crewmates / impostors win rate | 3/5 vs 2/5 |

With `DP=8` on 8 GPUs you should see roughly an 8x throughput bump: ~2-3 min per
game wall time when running multiple games in parallel.


The upstream code ([github.com/7vik/AmongUs](https://github.com/7vik/AmongUs)) expects
an OpenRouter API key and posts to `https://openrouter.ai/api/v1/chat/completions`.
We add a minimal patch (`among-agents/amongagents/agent/agent.py`) so the same
OpenAI-compatible client talks to a local vLLM server instead.

## 0. Prerequisites

- NVIDIA GPU(s) with recent drivers (tested on 8x RTX PRO 6000 Blackwell 96GB).
- `uv` installed and available on `$PATH` (`curl -LsSf https://astral.sh/uv/install.sh | sh`).
- Enough disk to cache HuggingFace model weights.

## 1. Clone and install

```bash
git clone https://github.com/lococaeco/umongus.git
cd umongus

# Create a Python 3.10 venv with uv
uv venv --python 3.10
source .venv/bin/activate

# Core project deps
uv pip install -r requirements.txt

# Editable install of the "amongagents" subpackage
uv pip install -e ./among-agents

# vLLM (matching transformers version used by Qwen2.5 tokenizer)
uv pip install vllm==0.11.0
uv pip install "transformers==4.56.2"
```

Smoke test:

```bash
.venv/bin/python -c "import vllm, torch; print('vllm', vllm.__version__, 'torch', torch.__version__, 'cuda', torch.cuda.is_available())"
```

## 2. Configure the environment file

Create `.env` in the repo root:

```bash
cat > .env <<'EOF'
LLM_API_URL=http://localhost:8000/v1/chat/completions
LLM_API_KEY=dummy
OPENROUTER_API_KEY=dummy
EOF
```

- `LLM_API_URL` can be a **comma-separated list** of endpoints. The client
  round-robins across them across retries, so pointing it at multiple vLLM
  servers gives you client-side load balancing for free.
- OpenRouter variables are kept around so the upstream code paths that read them
  don't crash. Set a real `OPENROUTER_API_KEY` if you ever want to flip back to
  the hosted path.

## 3. Start a local vLLM server

A helper script is provided: [`run_vllm_server.sh`](./run_vllm_server.sh).
It reads configuration via environment variables:

| var | default | meaning |
| --- | --- | --- |
| `MODEL` | `Qwen/Qwen2.5-32B-Instruct` | any HF id or local path |
| `PORT` | `8000` | HTTP port |
| `TP`   | `1` | tensor-parallel size (shards model across GPUs) |
| `DP`   | `1` | data-parallel size (replicates model, adds throughput) |
| `CUDA_VISIBLE_DEVICES` | `0` | which GPUs to expose |
| `MAX_LEN` | `16384` | max context |
| `GMEM` | `0.8` | gpu-memory-utilization |

### Single-GPU run (smoke test)

```bash
CUDA_VISIBLE_DEVICES=0 MODEL=Qwen/Qwen2.5-32B-Instruct ./run_vllm_server.sh
```

Wait until `curl -s http://localhost:8000/v1/models` returns 200.

### Using all 8 GPUs: two recipes

The game has 7 concurrent agents per game, and `main.py --num_games N` runs
games concurrently via `asyncio`, so throughput matters a lot.

**Recipe A: 8 replicas of Qwen2.5-32B via vLLM data-parallel**
(best if your model already fits comfortably on one GPU)

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 DP=8 TP=1 ./run_vllm_server.sh
```

vLLM exposes a single logical port (8000) and internally load-balances across
the 8 replicas. 1 client change needed? None — just keep the single
`LLM_API_URL` in `.env`.

**Recipe B: Llama-3.3-70B with TP=2, DP=4** (matches the paper's impostor model)

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 MODEL=meta-llama/Llama-3.3-70B-Instruct TP=2 DP=4 ./run_vllm_server.sh
```

70B in bf16 needs ~140GB of weights → 2 GPUs per replica, leaving room for
4 replicas across 8 GPUs.

**Recipe C: multiple independent servers on different ports**
(useful if you want different models serving crewmate vs impostor)

```bash
# Terminal 1 – crewmate model
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=8000 MODEL=meta-llama/Llama-3.3-70B-Instruct TP=2 DP=2 ./run_vllm_server.sh

# Terminal 2 – impostor model
CUDA_VISIBLE_DEVICES=4,5,6,7 PORT=8001 MODEL=Qwen/Qwen2.5-32B-Instruct DP=4 ./run_vllm_server.sh
```

Then in `.env`:

```
LLM_API_URL=http://localhost:8000/v1/chat/completions,http://localhost:8001/v1/chat/completions
```

Round-robin happens per retry. For per-role routing (crewmate → port 8000,
impostor → port 8001) you would need a small additional patch to `agent.py`.

## 4. Run games

```bash
.venv/bin/python main.py \
    --num_games 10 \
    --crewmate_llm Qwen/Qwen2.5-32B-Instruct \
    --impostor_llm Qwen/Qwen2.5-32B-Instruct \
    --name my_experiment
```

Notes:

- `--crewmate_llm` / `--impostor_llm` **must match the `--served-model-name`**
  of the vLLM server (by default this is the HF id you passed as `MODEL`).
- Logs land in `expt-logs/<date>_<name>/`:
  - `agent-logs.json`        — pretty-printed full interactions
  - `agent-logs-compact.json` — concatenated JSON objects, one per LLM call
  - `summary.json`            — per-game winner + player assignments
  - `experiment-details.txt`  — game config + commit hash snapshot

- Upstream's "compact" file is **not** newline-delimited JSON; it is a stream
  of concatenated objects. Use the helper loader in `scripts/game_stats.py` to
  parse it (or `json.JSONDecoder().raw_decode` in a loop).

### Run 1 quick smoke-test game

```bash
.venv/bin/python main.py --num_games 1 \
    --crewmate_llm Qwen/Qwen2.5-32B-Instruct \
    --impostor_llm Qwen/Qwen2.5-32B-Instruct \
    --name smoke
```

## 5. Summarise results

```bash
.venv/bin/python scripts/game_stats.py expt-logs/<date>_<name>
```

This prints per-game rows (winner, steps, wall-time, LLM-call count, impostors)
and an aggregate (win rates, average steps, average wall time, total calls).

## 6. What we changed vs upstream

- [`among-agents/amongagents/agent/agent.py`](./among-agents/amongagents/agent/agent.py)
  - Read `LLM_API_URL` / `LLM_API_KEY` from env (with comma-separated multi-URL support).
  - Added `Content-Type: application/json` header (vLLM rejects requests without it).
  - Log non-200 response bodies so API errors stop being invisible.
- [`run_vllm_server.sh`](./run_vllm_server.sh) — new helper.
- [`scripts/game_stats.py`](./scripts/game_stats.py) — new stats reporter.
- [`REPRODUCE.md`](./REPRODUCE.md) — this file.

No game logic or prompts have been modified. Running against vLLM and running
against OpenRouter should produce identical game behaviour for the same model.
