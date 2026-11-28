# Ideas

## Scheduler
- EEVDF with VTC (input ×1, output ×2), premium:free 3:1 — kept.
- Charge input on GPU admission — kept.
- Reserve input + max_tokens up front — kept.
- Bounded dispatch queue, no skipping ahead — kept.
- 24h billing caps counted like vruntime — kept.
- Per-model vruntime, billing summed across models — kept.

## Fairness
- Weights 4:2 or 2:1 instead of 3:1 — weight is now a flag.
- Reset vruntimes when the tier ratio drifts too far — favored premium, off by default.
- Premium and free queues, EEVDF inside each (`tiered`) — removed.
- Two mixed 75/25 queues — removed.
- Pick premium queue 75% of the time (`lottery`) — removed.
- Plain EEVDF, since free users don't pay their share — kept.
- Random 0–50% cut to premium deadlines (`eevdfr`) — removed.
- Random ineligible premium made eligible each pick (`eevdfre`) — removed.
- Premium-tenant competition stretches EEVDF's tail vs DRR — confirmed: about 1.2× p99 under overload, up to 2×.

## Rerank
- Chunk pairs into groups of 3 — superseded.
- Shared deadline for all pairs — superseded.
- Whole rerank as one job — kept.
- Large rerank inputs cause head-of-line blocking — confirmed.

## Workload
- Bursty arrivals (`-burst`) — kept.
- Zipf tenant skew with quotas that grow with traffic (`-skew`) — kept.
- Production prompt and output lengths (`-llm-input-median`, `-llm-output-median`) — kept.
- Matrix of burst × skew × prompt length at heavy load (`scripts/matrix.py`) — kept.
- Client timeouts and short surges instead of unbounded overload — next.

## Metrics and baselines
- DRR with strict tier priority as a baseline — kept.
- Backlog shedding — removed.
- Jain index — removed for lag.
- Goodput against SLOs — kept.
- Signed per-tier lag — kept.
- Full latency distributions (`-samples`, `scripts/dist.py`) — kept.
- Share of picks with 2+ tenants waiting — kept.
- First token within SLO, tracked apart from goodput — kept.
- Multiple seeds before trusting tail differences — kept.

## GPUs
- vLLM on one RunPod L40S, 70/10/10 split — done.
- Simulator calibrated to the L40S — done.
