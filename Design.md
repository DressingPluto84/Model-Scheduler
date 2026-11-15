# Multi-Tenant GPU Request Scheduler: MVP Design

## Architecture

Client → API gateway → services (embed, rerank, LLM) → schedulers (embed, rerank, LLM) → GPUs for each model.

- GPUs return output tokens to the services, which send them back to the clients.
- GPUs also return token counts to the services for VTC accounting.
- Services send those counts to the schedulers.
- Source of truth for token counts: the services are the source of truth for token counts.
- There is currently only 1 GPU per model, and for the MVP's sake we assume the GPUs will always be saturated.

## Implementation (MVP)

- Written in Go.
- For the MVP, everything runs as a single process with different modules (gateway, services, schedulers, GPUs).
- All state (rate-limit windows, vruntimes, billing usage) is kept in memory for the MVP; we will distribute it later.

## Simulated GPUs

There are no real GPUs in the MVP. Each model has one simulated GPU, running as its own goroutine, that models memory capacity and timing:

- **LLM GPU (continuous batching):** has a KV-cache budget in tokens. A request is admitted only if `input + max_tokens` fits, and that reservation is held until the request completes or is cancelled. Each step prefills newly admitted requests (time proportional to their input tokens) and decodes one token for every running request (time = `base + β × batch size`). Each request's true output length is sampled from a distribution and capped at `max_tokens`. Tokens are emitted back to the LLM service as they are generated.
- **Embed / rerank GPUs:** have a token budget for in-flight work. A batch takes `base + α × total tokens`. For rerank, each doc-query pair is one job.
- **Clock:** everything runs on a `Clock` interface. The live server uses the real clock; benchmarks use a simulated, event-driven clock so runs are fast and deterministic.
- All capacities and timing constants are configurable.

## API Gateway

We will have standard rate limiting procedures in the API gateway. The granularity can be simple (tenant, endpoint) keys with a sliding window approach. This is the best place to prevent regular abuse, so that unusually high traffic doesn't hit the rest of the scheduler and we preserve time for jobs that actually have a chance to run.

## Services

These services all enforce token limits, which depend on customer status. The rerank service splits a request into doc-query pairs to reduce the size of a single job; each pair is its own request, and the service combines the results at the end.

Only the LLM service streams; it streams tokens via websocket to the clients that sent the requests. Embed and rerank return a single response.

## Schedulers

Each scheduler runs EEVDF. Input tokens are weighted as 1, and output tokens (if any) are weighted as 2. Weights are based on premium vs. free requests, with w = 3 for premium and w = 1 for free. We use VTC accounting methods for tokens.

```
vruntime             = input + 2 × output, for all tokens processed all time
virtual time for job = num input tokens (num in pair for rerank) / weight
used                 = vruntime / weight
eligible            -> vruntime <= sum of system vruntime × (w_i / sum of all w)
                       (both sums over enqueued clients)
deadline             = used + num input / weight
choose smallest deadline of eligible
```

- vruntime is global per tenant: it is shared across the embed, rerank, and LLM schedulers.
- The scheduler queues requests, and each request carries the values from its client that we need to make decisions (vruntime, weight).
- The deadline uses only input tokens for now.
- Input tokens are charged when a job is admitted to the GPU. This prevents someone from being charged for a request that never actually ran.

For a given GPU decoding, we batch our writes for VTC accounting, incrementing vruntime in batches of k tokens generated (k could be around 32) until no more tokens are generated, flushing when done (on completion or cancellation).

We will store the requests in an RB tree, as is done in Linux. Clients that return after some time come back at the max of their current vruntime and their minimum eligible value (sum of system vruntime × w_i / sum of all w). A tenant leaves either by not making requests for a while (an idle timeout, defaulting to 30s with nothing queued or running) or by disconnecting and coming back.

### Dispatch queue

Each scheduler has a small, bounded dispatch queue (configurable depth) between it and its GPU. The scheduler picks requests by EEVDF into this queue ahead of time, so the GPU always has the next job ready, and the GPU pulls from it in order as space frees up. If the request at the head of the queue doesn't fit in the GPU's free memory, the GPU waits until it fits rather than skipping ahead.

## Admission

Admission depends on what the GPU numbers look like, so a request is only admitted if there is enough space on the GPU to fit it. For LLM endpoints, we also need to check that, beyond having space for the input, the KV cache has enough space for future and continued generation to avoid eviction costs. To do this, we reserve `input + max_tokens` upfront.

## Per-Tenant Caps (in-flight requests, max_tokens)

For free level users, we will use the following rate limits:
- /rerank -> 10 req/min
- /embed -> 15 req/min
- /llm -> 5 req/min

Premium level users will get the following:
- /rerank -> 20 req/min
- /embed -> 30 req/min
- /llm -> 10 req/min

The maximum number of input tokens per request is 4096 for now.

LLM decode should have a max output of 4096 tokens.

## Overload / Load Shedding (429)

429s are returned on rate limit hits. There is no queue-length based load shedding yet.

## Cancellation

If a client disconnects, the rest of the job is aborted off of the GPU, and the tokens used still count towards their vruntime. For rerank, any pairs that have not been computed yet are removed.

## Billing

- Output tokens cost 2x as much as input tokens, since we don't know how many could possibly be generated and want to penalize a client who sends small requests that decode for a long time.
- Bills are not by token usage; rather, there is a limit set by whether you are free or premium. Premium customers get a higher limit: 86k tokens for premium and 43k tokens for free.
- Tokens are counted the same way as vruntime (input + 2 × output), but we never augment this metric for awakened clients like we do vruntime.
- The limit is for a rolling 24 hour period.
- Once a tenant is at or over the limit, new requests return a 429 (with a message distinct from the rate-limit 429) until the rolling window moves past the limit. A request already in flight is allowed to finish, even if it pushes usage over the limit.

## Deferred (Out of Scope for MVP)

- Multiple GPUs per model; would maybe do one queue per model
- Prefix-based routing (GPU affinity)
- Queue-length based load shedding
- Distributed state

## Testing / Validation

- Mix of free/premium
- Hitting limits for token usage and num requests
- Hitting limits for request sizes
- Client comes alive after sleep
- Benchmark against FCFS. Other schedulers may be added later.

### Benchmark metrics

All metrics are reported separately for both tiers.

- LLM: TTFT (time to first token) and ITL (inter-token latency, the gap between consecutive output tokens).
- Embed / rerank: end-to-end latency.
- Fairness: compared using the lag metric (each tenant's actual service vs. its weighted fair share).
