# Multi-Tenant GPU Request Scheduler: MVP Design

## Architecture

Client → API gateway → services (embed, rerank, LLM) → schedulers (embed, rerank, LLM) → GPUs for each model.

- GPUs return output tokens to the services, which send them back to the clients.
- GPUs also return token counts to the services for VTC accounting.
- Services will send the counts to schedulers
- Source of truth for token counts: Services will be source of truth for token counts.
- Also currently only 1 gpu per model, and assume gpus will always be saturated for mvps sake

## API Gateway

We will have standard rate limiting procedures in the API gateway. Accurate granularity could be simple (tenant, endpoint) keys and a sliding window approach. This is best to prevent regular abuse, so that unusually high traffic doesn't hit the rest of the scheduler and we can preserve time used for jobs that would actually have a chance to run.

## Services

These services will all ensure token limits, which depend on customer status. The rerank service will create doc-query pairs to reduce the size of a single job.

These services will also stream tokens via websocket to the clients that have sent requests.

## Schedulers

Each scheduler will run on EEVDF. Input tokens are weighted as 1, output tokens (if any) are weighted as 2. Weights are based on premium vs free requests, at w = 3 for premium and w = 1 for free. We use VTC accounting methods for tokens.

```
vruntime             = input + 2 × output, for all tokens processed all time
virtual time for job = num input tokens (num in pair for rerank) / weight
used                 = vruntime / weight
eligible            -> vruntime <= sum of system vruntime × (w_i / sum of all w)
                       (both sums over enqueued clients)
deadline             = used + num input / weight
choose smallest deadline of eligible
```

For a given GPU decoding, we will batch our writes for VTC accounting to increment vruntime in batches of k tokens generated (k could be like 32), doing so until we have no more tokens generated, flushing when done (on completion or cancellation).

We will store the clients in an RB tree as is done in Linux. Clients that return after some time return as the max of their current vruntime and their min eligible value (sum of system vruntime × w_i / sum of all w).

## Admission

Admission is dependent on what GPU numbers look like, so admission only happens if there is enough space on the GPU to fit the new request. For LLM endpoints, we also need to check that, even if there is enough space for the input, the KV cache will have enough space for future and continued generation to avoid eviction costs.

## Per-Tenant Caps (in-flight requests, max_tokens)

For free level users, we will use the following rate limits:
- /rerank -> 10 req/min
- /embed -> 15 req/min
- /llm -> 5 req/min

Premium level users will get the following:
- /rerank -> 20 req/min
- /embed -> 30 req/min
- /llm -> 10 req/min

LLM decode should have a max output of 4096 tokens.

## Overload / Load Shedding (429)

429s are returned on rate limit hits. We could possibly also start rejecting requests if queues become too long.

## Cancellation

If a client disconnects, the rest of the job will be aborted off of the gpu, and the tokens used still count towards their vruntime.

## Billing

- Output tokens are 2x the cost of input tokens since we don't know how many could possibly be generated and want to penalize a client who sends small requests that decode for long.
- Bills are not by token usage, rather by a limit set by if you are free or premium. Premium customers get a higher limit.
- This metric is the same as vruntime, but we never augment it for awakened clients like we do vruntime.
- The limit is for a rolling 24 hour period

## Deferred (Out of Scope for MVP)

- Multiple gpus per model, would maybe do one queue per model
- Prefix based routing (gpu affinity)

## Testing / Validation

- Mix of free/premium
- Hitting limits for token usage and num requests
- Hitting limits for request sizes
- Client comes alive after sleep
- Benchmark against fcfs and find other schedulers.
