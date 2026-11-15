# Model-Scheduler

A multi-tenant GPU request scheduler for embed, rerank and LLM inference, using EEVDF with VTC token accounting over simulated GPUs. See [Design.md](Design.md) for the design.

## Commands

Run the server:

```bash
go run ./cmd/server -tenants tenants.json -policy eevdf
```

Run the FCFS vs EEVDF benchmark:

```bash
go run ./cmd/bench
```

List benchmark options:

```bash
go run ./cmd/bench -h
```

Run against real models on a RunPod GPU with vLLM (full steps in [deploy/runpod/README.md](deploy/runpod/README.md)):

```bash
bash deploy/runpod/run-bench.sh -duration 2m
```

Run the tests:

```bash
go test ./...
```

Send an embed request:

```bash
curl -s localhost:8080/v1/embed -H 'X-Tenant-ID: acme' -d '{"input": "hello world"}'
```

Send a rerank request:

```bash
curl -s localhost:8080/v1/rerank -H 'X-Tenant-ID: acme' -d '{"query": "gpu", "documents": ["a gpu", "a cat"]}'
```

Check a tenant's usage:

```bash
curl -s localhost:8080/v1/usage -H 'X-Tenant-ID: acme'
```

Stream from the LLM over a websocket (needs [websocat](https://github.com/vi/websocat)):

```bash
echo '{"prompt": "hello there", "max_tokens": 32}' | websocat 'ws://localhost:8080/v1/llm?tenant=acme'
```
