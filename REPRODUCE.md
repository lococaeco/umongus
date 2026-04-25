# Running AmongUs locally on 8 GPUs with vLLM + uv

---

## 1. Prerequisites

- `uv` installed and on `$PATH`
  (`curl -LsSf https://astral.sh/uv/install.sh | sh`)

## 2. Clone and install

```bash
git clone git@github.com:lococaeco/umongus.git
cd umongus

uv venv --python 3.10
source .venv/bin/activate

uv pip install -r requirements.txt
uv pip install -e ./among-agents
uv pip install vllm==0.11.0 "transformers==4.56.2"
```
test:

```bash
.venv/bin/python -c "import vllm, torch; print('vllm', vllm.__version__, 'torch', torch.__version__, 'cuda', torch.cuda.is_available())"
```

## 3. Configure `.env`

```bash
cat > .env <<'EOF'
LLM_API_URL=http://localhost:8000/v1/chat/completions
LLM_API_KEY=dummy
OPENROUTER_API_KEY=dummy
EOF
```

- `LLM_API_URL` can be a **comma-separated list** of endpoints. The client
  round-robins across them per retry, so pointing it at multiple independent
  vLLM servers gives you client-side load balancing for free.
- The `OPENROUTER_API_KEY` fallback is kept so upstream code paths that read
  it don't crash. Set a real key if you ever want to flip back to hosted.

## 4. Start the vLLM server
**Recipe A — same model per role**
Default invocation — **uses all 8 GPUs**:

```bash
./run_vllm_server.sh
```

That's equivalent to:

```
MODEL=Qwen/Qwen2.5-32B-Instruct
PORT=8000
TP=1
DP=8
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
MAX_LEN=16384
GMEM=0.85
```

**Recipe B — different models per role**

```bash
# Terminal 1 — crewmate model on GPUs 0-3
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=8000 \
  MODEL=meta-llama/Llama-3.3-70B-Instruct TP=2 DP=2 ./run_vllm_server.sh

# Terminal 2 — impostor model on GPUs 4-7
CUDA_VISIBLE_DEVICES=4,5,6,7 PORT=8001 \
  MODEL=Qwen/Qwen2.5-32B-Instruct DP=4 ./run_vllm_server.sh
```

## 5. Run games

```bash
.venv/bin/python main.py \
    --num_games 10 \
    --crewmate_llm Qwen/Qwen2.5-32B-Instruct \
    --impostor_llm Qwen/Qwen2.5-32B-Instruct \
    --name my_experiment
```

Notes:

- `--crewmate_llm` / `--impostor_llm` **must match the `served-model-name` of
  the vLLM server**, which defaults to the HF id you passed as `MODEL`.
- Games run concurrently via `asyncio.gather`, so bigger `--num_games` plus
  `DP=8` gives you near-linear throughput scaling up to the number of
  replicas.
- Logs land under `expt-logs/<date>_<name>/`:
  - `agent-logs.json`         — pretty-printed full interactions
  - `agent-logs-compact.json` — concatenated JSON objects (one per LLM call)
  - `summary.json`             — per-game winner + player assignments
  - `experiment-details.txt`   — game config + commit hash snapshot
- The "compact" file is **not** newline-delimited JSON; it is a stream of
  concatenated objects. Use `scripts/game_stats.py` (or
  `json.JSONDecoder().raw_decode` in a loop) to parse it.

### Quick smoke test

```bash
.venv/bin/python main.py --num_games 1 \
    --crewmate_llm Qwen/Qwen2.5-32B-Instruct \
    --impostor_llm Qwen/Qwen2.5-32B-Instruct \
    --name smoke
```

## 6. Summarise results

```bash
.venv/bin/python scripts/game_stats.py expt-logs/<date>_<name>
```

Prints a per-game table (winner, steps, wall time, LLM-call count, impostors)
plus an aggregate (win rates, averages, totals). Example output from a
5-game batch with **one** GPU running Qwen2.5-32B (DP=1, TP=1):

```
GAME       WINNER     STEPS  WALL(s)   LLM CALLS  IMPOSTORS
-----------------------------------------------------------
Game 1     Impostors  6      535.2     54         Player 1: red, Player 6: white
Game 2     Impostors  11     1264.0    117        Player 3: yellow, Player 4: white
Game 3     Crewmates  13     809.1     85         Player 2: black, Player 5: red
Game 4     Crewmates  14     900.2     93         Player 1: orange, Player 5: blue
Game 5     Crewmates  12     727.9     78         Player 5: black, Player 7: yellow

Aggregate over 5 games
  Win rate: Crewmates 3/5, Impostors 2/5
  Avg steps per game: 11.2
  Avg wall time per game: 847.3s
  Avg LLM calls per game: 85.4
  Total LLM calls: 427
```

Total wall time for those 5 concurrent games was **~1271 s (~21 min)** on one
GPU. With `DP=8` on all 8 GPUs you should see roughly an 8× throughput
improvement (≈2–3 min wall time per game when running many games
concurrently), subject to how much the agents serialise on each other.

## 7. What changed vs upstream

| File | Change |
| --- | --- |
| [`among-agents/amongagents/agent/agent.py`](./among-agents/amongagents/agent/agent.py) | Read `LLM_API_URL` / `LLM_API_KEY` from env, support comma-separated multi-URL round-robin, add `Content-Type: application/json` header (vLLM rejects requests without it), log non-200 response bodies |
| [`run_vllm_server.sh`](./run_vllm_server.sh) | New helper, defaults to 8-way DP across all 8 GPUs |
| [`scripts/game_stats.py`](./scripts/game_stats.py) | New per-game + aggregate stats reporter |
| [`REPRODUCE.md`](./REPRODUCE.md) | This file |

No game logic or prompts have been modified. Running against vLLM and against
OpenRouter should produce identical game behaviour for the same model.
