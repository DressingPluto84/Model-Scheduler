# Shared settings for the RunPod scripts. Sourced, not run.
# Everything lives on the /workspace volume so it survives pod restarts.

export WORKSPACE="${WORKSPACE:-/workspace}"
export REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
export VENV="$WORKSPACE/venv"
export HF_HOME="$WORKSPACE/hf"
export LOG_DIR="$WORKSPACE/logs"
export RESULTS_DIR="$WORKSPACE/results"
export GO_VERSION="${GO_VERSION:-1.26.5}"
export PATH="/usr/local/go/bin:$VENV/bin:$PATH"

export LLM_MODEL="${LLM_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
export EMBED_MODEL="${EMBED_MODEL:-BAAI/bge-base-en-v1.5}"
export RERANK_MODEL="${RERANK_MODEL:-BAAI/bge-reranker-v2-m3}"

# Not 8001: the Runpod PyTorch image runs nginx there (a proxy to 8000).
export LLM_PORT="${LLM_PORT:-8000}"
export EMBED_PORT="${EMBED_PORT:-8002}"
export RERANK_PORT="${RERANK_PORT:-8003}"

# Flags that point the Go binaries at the servers above.
VLLM_FLAGS=(
  -llm-url "http://127.0.0.1:$LLM_PORT" -llm-model "$LLM_MODEL"
  -embed-url "http://127.0.0.1:$EMBED_PORT" -embed-model "$EMBED_MODEL"
  -rerank-url "http://127.0.0.1:$RERANK_PORT" -rerank-model "$RERANK_MODEL"
)

# GPU memory split on the one L40S (fractions of total memory).
export LLM_GPU_MEM="${LLM_GPU_MEM:-0.70}"
export EMBED_GPU_MEM="${EMBED_GPU_MEM:-0.10}"
export RERANK_GPU_MEM="${RERANK_GPU_MEM:-0.10}"

# Must be >= the scheduler's -llm-max-batch (256) so vLLM never queues.
export LLM_MAX_NUM_SEQS="${LLM_MAX_NUM_SEQS:-256}"
