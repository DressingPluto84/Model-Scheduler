#!/usr/bin/env bash
# Starts the three vLLM servers on the one GPU, one at a time (each one must
# be up before the next starts, so vLLM's memory check sees the others).
# Logs go to $LOG_DIR. Stop them with deploy/runpod/stop-vllm.sh.
set -euo pipefail
source "$(dirname "$0")/env.sh"
mkdir -p "$LOG_DIR"

wait_ready() {
  local name=$1 port=$2 pid=$3
  echo -n "waiting for $name on :$port"
  for _ in $(seq 1 600); do
    if curl -sf "http://127.0.0.1:$port/health" >/dev/null; then
      echo " ready"
      return 0
    fi
    if ! kill -0 "$pid" 2>/dev/null; then
      echo
      echo "$name exited during startup. Last log lines:"
      tail -n 40 "$LOG_DIR/$name.log"
      exit 1
    fi
    echo -n "."
    sleep 2
  done
  echo
  echo "$name did not become ready in 20 minutes; see $LOG_DIR/$name.log"
  exit 1
}

start() {
  local name=$1 port=$2
  shift 2
  if curl -sf "http://127.0.0.1:$port/health" >/dev/null; then
    echo "$name already running on :$port"
    return
  fi
  nohup "$VENV/bin/vllm" serve "$@" --port "$port" >"$LOG_DIR/$name.log" 2>&1 &
  echo $! >"$LOG_DIR/$name.pid"
  wait_ready "$name" "$port" $!
}

# LLM: max_model_len covers the largest request (4096 input + 4096 output).
start llm "$LLM_PORT" "$LLM_MODEL" \
  --gpu-memory-utilization "$LLM_GPU_MEM" \
  --max-model-len 8192 \
  --max-num-seqs "$LLM_MAX_NUM_SEQS"

start embed "$EMBED_PORT" "$EMBED_MODEL" \
  --runner pooling \
  --gpu-memory-utilization "$EMBED_GPU_MEM"

start rerank "$RERANK_PORT" "$RERANK_MODEL" \
  --runner pooling \
  --gpu-memory-utilization "$RERANK_GPU_MEM"

nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv
echo "All three vLLM servers are up. Next: deploy/runpod/smoke-test.sh"
