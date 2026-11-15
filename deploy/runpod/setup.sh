#!/usr/bin/env bash
# One-time setup on a fresh RunPod pod: installs Go and vLLM, downloads the
# three models, and builds the Go binaries. Safe to re-run.
set -euo pipefail
source "$(dirname "$0")/env.sh"

mkdir -p "$LOG_DIR" "$RESULTS_DIR" "$HF_HOME"

echo "== System packages"
apt-get update -qq
apt-get install -y -qq tmux curl jq >/dev/null

echo "== Go $GO_VERSION"
if ! go version 2>/dev/null | grep -q "go$GO_VERSION"; then
  curl -fsSL "https://go.dev/dl/go${GO_VERSION}.linux-amd64.tar.gz" -o /tmp/go.tgz
  rm -rf /usr/local/go
  tar -C /usr/local -xzf /tmp/go.tgz
  rm /tmp/go.tgz
fi
go version

echo "== vLLM (in $VENV)"
if [ ! -x "$VENV/bin/vllm" ]; then
  python3 -m venv "$VENV"
  "$VENV/bin/pip" install -q --upgrade pip
  "$VENV/bin/pip" install -q vllm
fi
"$VENV/bin/python" -c 'import vllm; print("vllm", vllm.__version__)'

echo "== Models (into $HF_HOME)"
"$VENV/bin/python" - <<EOF
from huggingface_hub import snapshot_download
for m in ["$LLM_MODEL", "$EMBED_MODEL", "$RERANK_MODEL"]:
    print("downloading", m, flush=True)
    snapshot_download(m)
EOF

echo "== Go binaries"
cd "$REPO_DIR"
mkdir -p bin
go build -o bin/server ./cmd/server
go build -o bin/bench ./cmd/bench

echo "Setup done. Next: deploy/runpod/start-vllm.sh"
