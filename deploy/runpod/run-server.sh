#!/usr/bin/env bash
# Runs the scheduler's HTTP server on :8080 against the vLLM servers.
# Extra arguments go to the server binary, e.g. -policy fcfs.
set -euo pipefail
source "$(dirname "$0")/env.sh"
cd "$REPO_DIR"
exec ./bin/server -backend vllm "${VLLM_FLAGS[@]}" -tenants tenants.json "$@"
