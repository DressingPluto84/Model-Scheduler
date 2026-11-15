#!/usr/bin/env bash
# Checks every vLLM endpoint the scheduler uses, then the scheduler itself.
# Run after start-vllm.sh. Takes under a minute.
set -euo pipefail
source "$(dirname "$0")/env.sh"
mkdir -p "$LOG_DIR"

LLM="http://127.0.0.1:$LLM_PORT"
EMBED="http://127.0.0.1:$EMBED_PORT"
RERANK="http://127.0.0.1:$RERANK_PORT"

check() {
  local what=$1 expr=$2
  shift 2
  local out
  if ! out=$(curl -sf "$@"); then
    echo "FAIL $what: request failed"
    curl -s "$@" | head -c 500
    echo
    exit 1
  fi
  if ! echo "$out" | jq -e "$expr" >/dev/null 2>&1; then
    echo "FAIL $what: unexpected response"
    echo "$out" | head -c 500
    echo
    exit 1
  fi
  echo "ok   $what"
}

check "llm /tokenize" '.count > 0' "$LLM/tokenize" -H 'Content-Type: application/json' \
  -d "{\"model\": \"$LLM_MODEL\", \"prompt\": \"the scheduler works\"}"
check "llm /v1/completions" '.choices[0].text | length > 0' "$LLM/v1/completions" -H 'Content-Type: application/json' \
  -d "{\"model\": \"$LLM_MODEL\", \"prompt\": \"Hello\", \"max_tokens\": 8, \"min_tokens\": 8, \"ignore_eos\": true}"
check "embed /tokenize" '.count > 0' "$EMBED/tokenize" -H 'Content-Type: application/json' \
  -d "{\"model\": \"$EMBED_MODEL\", \"prompt\": \"hello world\"}"
check "embed /v1/embeddings" '.data[0].embedding | length > 0' "$EMBED/v1/embeddings" -H 'Content-Type: application/json' \
  -d "{\"model\": \"$EMBED_MODEL\", \"input\": \"hello world\"}"
check "rerank /rerank" '.results[0].relevance_score != null' "$RERANK/rerank" -H 'Content-Type: application/json' \
  -d "{\"model\": \"$RERANK_MODEL\", \"query\": \"what is a gpu\", \"documents\": [\"a gpu is a graphics processor\"]}"

kv=$(curl -sf "$LLM/metrics" | grep '^vllm:cache_config_info' || true)
if [ -z "$kv" ]; then
  echo "FAIL llm /metrics has no vllm:cache_config_info (pass -kv-tokens to the server/bench to work around)"
  exit 1
fi
echo "ok   llm /metrics: $(echo "$kv" | grep -o 'num_gpu_blocks="[0-9]*"') $(echo "$kv" | grep -o 'block_size="[0-9]*"')"

echo "== scheduler"
cd "$REPO_DIR"
./bin/server -backend vllm "${VLLM_FLAGS[@]}" -addr 127.0.0.1:${SCHED_PORT:-8080} >"$LOG_DIR/server-smoke.log" 2>&1 &
server=$!
trap 'kill $server 2>/dev/null' EXIT
for _ in $(seq 1 60); do curl -sf 127.0.0.1:${SCHED_PORT:-8080}/healthz >/dev/null && break; sleep 1; done
check "scheduler /v1/embed" '.embedding | length > 0' 127.0.0.1:${SCHED_PORT:-8080}/v1/embed -H 'X-Tenant-ID: acme' \
  -d '{"input": "hello world"}'
check "scheduler /v1/rerank" '.results | length == 2' 127.0.0.1:${SCHED_PORT:-8080}/v1/rerank -H 'X-Tenant-ID: acme' \
  -d '{"query": "what is a gpu", "documents": ["a cat", "a gpu is a graphics processor"]}'
grep "KV budget" "$LOG_DIR/server-smoke.log" || true
echo "All checks passed. Next: deploy/runpod/run-bench.sh -duration 2m"
