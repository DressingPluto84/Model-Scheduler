#!/usr/bin/env bash
# Runs the FCFS vs EEVDF benchmark against the vLLM servers and saves the
# report to $RESULTS_DIR. Extra arguments go to the bench binary, e.g.
#   deploy/runpod/run-bench.sh -duration 2m -kv-tokens 16384
# Each policy runs for -duration of real time (default 10m, so ~21 min total).
set -euo pipefail
source "$(dirname "$0")/env.sh"
mkdir -p "$RESULTS_DIR"

out="$RESULTS_DIR/bench-$(date +%Y%m%d-%H%M%S).txt"
cd "$REPO_DIR"
{
  echo "args: $*"
  nvidia-smi --query-gpu=name --format=csv,noheader
  echo
  ./bin/bench -backend vllm "${VLLM_FLAGS[@]}" "$@"
} 2>&1 | tee "$out"
echo
echo "Saved to $out"
