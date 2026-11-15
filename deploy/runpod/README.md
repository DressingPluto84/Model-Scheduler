# Running on a RunPod L40S

These steps run the scheduler on one L40S with three vLLM servers sharing its memory (70% LLM, 10% embed, 10% rerank). The Go server and the benchmark run on the same machine.

## 1. Try it on your Mac first (free)

Start a fake vLLM that answers for all three models:

```bash
go run ./cmd/fakevllm -addr 127.0.0.1:8001
```

In a second terminal, run a short real-time benchmark against it:

```bash
go run ./cmd/bench -backend vllm -duration 30s -drain 2s -embed-url http://127.0.0.1:8001 -rerank-url http://127.0.0.1:8001
```

## 2. Create the pod

1. Add about $15 of credit to your RunPod account.
2. Add your SSH public key under Settings → SSH Public Keys.
3. Deploy a pod with:
   - **GPU:** 1 × L40S, on-demand. Not spot: a spot pod can be stopped in the middle of a run.
   - **Template:** RunPod PyTorch, any CUDA 12 version.
   - **Disk:** 30 GB container disk, plus an 80 GB volume mounted at `/workspace`. The volume holds vLLM, the models and your results.
4. Use the pod's "Connect" button to get its SSH command.

## 3. Get the code onto the pod

If you've pushed the repo to GitHub, run this on the pod:

```bash
cd /workspace && git clone https://github.com/DressingPluto84/Model-Scheduler.git
```

Or copy it from your Mac. Replace the port and IP with the ones in your pod's SSH command:

```bash
rsync -az --exclude .git -e "ssh -p <port>" ./ root@<pod-ip>:/workspace/Model-Scheduler/
```

## 4. Set up and start the servers (on the pod)

Work inside tmux so a dropped SSH connection doesn't kill anything:

```bash
tmux new -s sched
```

```bash
cd /workspace/Model-Scheduler
```

Install Go and vLLM, download the models and build the binaries. This takes about 15–20 minutes, once per volume:

```bash
bash deploy/runpod/setup.sh
```

Start the three vLLM servers. This takes about 3–5 minutes:

```bash
bash deploy/runpod/start-vllm.sh
```

Check every endpoint and the scheduler:

```bash
bash deploy/runpod/smoke-test.sh
```

## 5. Benchmark

Start with a short run. It takes about 5 minutes:

```bash
bash deploy/runpod/run-bench.sh -duration 2m -kv-tokens 16384 -embed-batch-tokens 4096 -rerank-batch-tokens 4096
```

An L40S has much more KV cache than this workload needs, so by default nothing ever queues and FCFS and EEVDF come out the same. The `-kv-tokens` and `-batch-tokens` flags shrink the budgets the scheduler admits against, so queues build up. Check that LLM time to first token is in seconds, not milliseconds. If it isn't, lower `-kv-tokens` further or add tenants, for example `-free 60 -premium 20`.

Once the short run shows queueing, do the full run. It takes about 21 minutes:

```bash
bash deploy/runpod/run-bench.sh -kv-tokens 16384 -embed-batch-tokens 4096 -rerank-batch-tokens 4096
```

Reports are saved to `/workspace/results`. Copy them to your Mac:

```bash
rsync -az -e "ssh -p <port>" root@<pod-ip>:/workspace/results/ ./results/
```

To serve real traffic instead of benchmarking, run the HTTP server on :8080:

```bash
bash deploy/runpod/run-server.sh
```

## 6. Stop paying

When you're done, stop the pod in the RunPod console. Stopping keeps `/workspace` (the volume has a small storage charge), so next time you only need `start-vllm.sh`. Terminating deletes everything.

## If something fails

- **Logs:** each vLLM server writes to `/workspace/logs/{llm,embed,rerank}.log`.
- **`--runner pooling` not recognised:** older vLLM versions used `--task embed` and `--task score`. Change those flags in `start-vllm.sh`.
- **A server runs out of memory at startup:** adjust the memory split with `LLM_GPU_MEM`, `EMBED_GPU_MEM` and `RERANK_GPU_MEM`, for example `LLM_GPU_MEM=0.65 bash deploy/runpod/start-vllm.sh`.
- **The smoke test can't read the KV cache size from `/metrics`:** pass `-kv-tokens` to the bench and the server, and they'll use that instead.
