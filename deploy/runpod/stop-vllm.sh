#!/usr/bin/env bash
# Stops the vLLM servers started by start-vllm.sh.
set -uo pipefail
source "$(dirname "$0")/env.sh"

for name in rerank embed llm; do
  pidfile="$LOG_DIR/$name.pid"
  if [ -f "$pidfile" ]; then
    kill "$(cat "$pidfile")" 2>/dev/null && echo "stopped $name"
    rm -f "$pidfile"
  fi
done
