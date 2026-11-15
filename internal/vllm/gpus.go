package vllm

import (
	"context"
	"flag"
	"fmt"
	"time"

	"github.com/DressingPluto84/Model-Scheduler/internal/gpu"
	"github.com/DressingPluto84/Model-Scheduler/internal/loop"
	"github.com/DressingPluto84/Model-Scheduler/internal/sched"
	"github.com/DressingPluto84/Model-Scheduler/internal/system"
	"github.com/DressingPluto84/Model-Scheduler/internal/tenant"
)

// Config points at the three vLLM servers.
type Config struct {
	LLMURL, EmbedURL, RerankURL       string
	LLMModel, EmbedModel, RerankModel string

	// KVCacheTokens caps the LLM KV-cache budget. 0 uses the size vLLM
	// reports. Setting it lower than the real cache creates contention on
	// purpose, so scheduling decisions matter at lower load.
	KVCacheTokens int
	// KVSafetyFraction of the reported KV cache is held back to cover vLLM's
	// per-request block rounding.
	KVSafetyFraction float64
	// LLMMaxBatch caps concurrent LLM sequences. Keep it at or below vLLM's
	// --max-num-seqs so vLLM never queues internally.
	LLMMaxBatch int
	// EmbedBatchTokens / RerankBatchTokens cap the input tokens in flight on
	// the pooling servers.
	EmbedBatchTokens  int
	RerankBatchTokens int
	// FlushEvery is k: output usage is reported in batches of k tokens.
	FlushEvery int
}

// DefaultConfig matches deploy/runpod/start-vllm.sh.
func DefaultConfig() Config {
	return Config{
		LLMURL:            "http://127.0.0.1:8001",
		EmbedURL:          "http://127.0.0.1:8002",
		RerankURL:         "http://127.0.0.1:8003",
		LLMModel:          "Qwen/Qwen2.5-7B-Instruct",
		EmbedModel:        "BAAI/bge-base-en-v1.5",
		RerankModel:       "BAAI/bge-reranker-v2-m3",
		KVSafetyFraction:  0.05,
		LLMMaxBatch:       256,
		EmbedBatchTokens:  16384,
		RerankBatchTokens: 16384,
		FlushEvery:        32,
	}
}

// Clients returns a client for each model.
func (c Config) Clients() (llm, embed, rerank *Client) {
	return NewClient(c.LLMURL, c.LLMModel), NewClient(c.EmbedURL, c.EmbedModel), NewClient(c.RerankURL, c.RerankModel)
}

// NewGPUs connects to the three servers (waiting for them to become ready)
// and returns GPUs for the system. lp must be a real loop.
func NewGPUs(ctx context.Context, lp *loop.Real, cfg Config) (system.GPUs, error) {
	llmC, embedC, rerankC := cfg.Clients()
	for _, c := range []*Client{llmC, embedC, rerankC} {
		if err := c.WaitReady(ctx); err != nil {
			return system.GPUs{}, err
		}
	}
	kv, err := llmC.KVCacheTokens(ctx)
	if err != nil {
		if cfg.KVCacheTokens == 0 {
			return system.GPUs{}, fmt.Errorf("read KV cache size (pass an explicit KV budget to skip): %w", err)
		}
		kv = cfg.KVCacheTokens
	}
	kv = int(float64(kv) * (1 - cfg.KVSafetyFraction))
	if cfg.KVCacheTokens > 0 && cfg.KVCacheTokens < kv {
		kv = cfg.KVCacheTokens
	}
	if kv < tenant.MaxInputTokens+tenant.MaxOutputTokens {
		return system.GPUs{}, fmt.Errorf("KV budget %d tokens is below the largest reservation %d",
			kv, tenant.MaxInputTokens+tenant.MaxOutputTokens)
	}
	for name, n := range map[string]int{"embed": cfg.EmbedBatchTokens, "rerank": cfg.RerankBatchTokens} {
		if n < tenant.MaxInputTokens {
			return system.GPUs{}, fmt.Errorf("%s batch tokens %d is below the max request size %d", name, n, tenant.MaxInputTokens)
		}
	}
	return system.GPUs{
		LLM: &LLMGPU{loop: lp, client: llmC, kvTokens: kv, maxBatch: cfg.LLMMaxBatch, flushEvery: cfg.FlushEvery,
			inflight: map[*sched.Request]context.CancelFunc{}},
		Embed: newPoolingGPU(lp, cfg.EmbedBatchTokens, func(ctx context.Context, r *sched.Request) (any, error) {
			p, _ := r.Payload.(sched.EmbedPayload)
			v, err := embedC.Embed(ctx, p.Text)
			return sched.EmbedResult(v), err
		}),
		Rerank: newPoolingGPU(lp, cfg.RerankBatchTokens, func(ctx context.Context, r *sched.Request) (any, error) {
			p, _ := r.Payload.(sched.RerankPayload)
			s, err := rerankC.Rerank(ctx, p.Query, p.Doc)
			return sched.RerankScore(s), err
		}),
	}, nil
}

// busy measures how long at least one request was in flight.
type busy struct {
	n     int
	since time.Time
	total time.Duration
}

func (b *busy) inc(now time.Time) {
	if b.n == 0 {
		b.since = now
	}
	b.n++
}

func (b *busy) dec(now time.Time) {
	if b.n--; b.n == 0 {
		b.total += now.Sub(b.since)
	}
}

func (b *busy) busyTime(now time.Time) time.Duration {
	if b.n > 0 {
		return b.total + now.Sub(b.since)
	}
	return b.total
}

// --- LLM ------------------------------------------------------------------------

// LLMGPU runs LLM requests on a vLLM server, reserving input + max_tokens of
// KV cache per request.
type LLMGPU struct {
	loop       *loop.Real
	sched      *sched.Scheduler
	client     *Client
	kvTokens   int
	maxBatch   int
	flushEvery int

	used     int
	inflight map[*sched.Request]context.CancelFunc
	busy     busy
	stats    gpu.Stats
}

func (g *LLMGPU) Attach(s *sched.Scheduler) { g.sched = s }

func (g *LLMGPU) CanAdmit(r *sched.Request) bool {
	return g.used+r.Reservation() <= g.kvTokens && len(g.inflight) < g.maxBatch
}

func (g *LLMGPU) Admit(r *sched.Request) {
	g.used += r.Reservation()
	ctx, cancel := context.WithCancel(context.Background())
	g.inflight[r] = cancel
	g.busy.inc(g.loop.Now())
	g.stats.Steps++
	r.OnUsage(r.InputTokens, 0) // input is charged when admitted to the GPU

	p, _ := r.Payload.(sched.LLMPayload)
	cr := CompletionRequest{Prompt: p.Prompt, MaxTokens: r.MaxTokens, ExactTokens: r.OutputLen}
	go func() {
		usage, err := g.client.StreamCompletion(ctx, cr, func(text string) {
			g.loop.Post(func() { g.onToken(r, text) })
		})
		g.loop.Post(func() { g.onEnd(r, usage, err) })
	}()
}

func (g *LLMGPU) onToken(r *sched.Request, text string) {
	if r.State() != sched.StateRunning {
		return // aborted; late tokens are dropped
	}
	idx := r.Generated
	r.Generated++
	g.stats.Tokens++
	if r.AddUnflushed(1) >= g.flushEvery {
		r.FlushUsage()
	}
	if r.OnToken != nil {
		r.OnToken(idx, text)
	}
}

func (g *LLMGPU) onEnd(r *sched.Request, usage *Usage, err error) {
	if r.State() != sched.StateRunning {
		return
	}
	// vLLM's usage is the final word on output tokens: some tokens arrive
	// without text of their own (e.g. a partial UTF-8 character).
	if usage != nil && usage.CompletionTokens > r.Generated {
		r.AddUnflushed(usage.CompletionTokens - r.Generated)
		r.Generated = usage.CompletionTokens
	}
	r.Err = err
	g.release(r)
	g.sched.Finish(r)
	g.sched.CapacityFreed()
}

func (g *LLMGPU) release(r *sched.Request) {
	r.FlushUsage()
	if cancel, ok := g.inflight[r]; ok {
		cancel()
		delete(g.inflight, r)
		g.used -= r.Reservation()
		g.busy.dec(g.loop.Now())
	}
}

// Abort closes the stream (vLLM aborts the request) and frees the KV
// reservation. Tokens already generated still count.
func (g *LLMGPU) Abort(r *sched.Request) {
	g.release(r)
	g.sched.Finish(r)
}

func (g *LLMGPU) Stats() gpu.Stats {
	s := g.stats
	s.Busy = g.busy.busyTime(g.loop.Now())
	s.SeqSteps = int64(len(g.inflight))
	s.UsedTokens = g.used
	s.CapacityTokens = g.kvTokens
	return s
}

// --- Embed / rerank ---------------------------------------------------------------

// PoolingGPU runs single-pass requests (embeddings or rerank pairs) on a vLLM
// pooling server, keeping at most batchTokens input tokens in flight. vLLM
// batches whatever is in flight.
type PoolingGPU struct {
	loop        *loop.Real
	sched       *sched.Scheduler
	batchTokens int
	call        func(context.Context, *sched.Request) (any, error)

	used     int
	inflight map[*sched.Request]context.CancelFunc
	busy     busy
	stats    gpu.Stats
}

func newPoolingGPU(lp *loop.Real, batchTokens int, call func(context.Context, *sched.Request) (any, error)) *PoolingGPU {
	return &PoolingGPU{loop: lp, batchTokens: batchTokens, call: call, inflight: map[*sched.Request]context.CancelFunc{}}
}

func (g *PoolingGPU) Attach(s *sched.Scheduler) { g.sched = s }

func (g *PoolingGPU) CanAdmit(r *sched.Request) bool { return g.used+r.InputTokens <= g.batchTokens }

func (g *PoolingGPU) Admit(r *sched.Request) {
	g.used += r.InputTokens
	ctx, cancel := context.WithCancel(context.Background())
	g.inflight[r] = cancel
	g.busy.inc(g.loop.Now())
	g.stats.Steps++
	g.stats.Tokens += int64(r.InputTokens)
	r.OnUsage(r.InputTokens, 0)
	go func() {
		res, err := g.call(ctx, r)
		g.loop.Post(func() {
			if r.State() != sched.StateRunning {
				return
			}
			r.Result, r.Err = res, err
			g.release(r)
			g.sched.Finish(r)
			g.sched.CapacityFreed()
		})
	}()
}

func (g *PoolingGPU) release(r *sched.Request) {
	if cancel, ok := g.inflight[r]; ok {
		cancel()
		delete(g.inflight, r)
		g.used -= r.InputTokens
		g.busy.dec(g.loop.Now())
	}
}

func (g *PoolingGPU) Abort(r *sched.Request) {
	g.release(r)
	g.sched.Finish(r)
}

func (g *PoolingGPU) Stats() gpu.Stats {
	s := g.stats
	s.Busy = g.busy.busyTime(g.loop.Now())
	s.UsedTokens = g.used
	s.CapacityTokens = g.batchTokens
	return s
}

// AddFlags registers command-line flags for the vLLM backend.
func (c *Config) AddFlags(fs *flag.FlagSet) {
	fs.StringVar(&c.LLMURL, "llm-url", c.LLMURL, "vLLM server for the LLM")
	fs.StringVar(&c.EmbedURL, "embed-url", c.EmbedURL, "vLLM server for embeddings")
	fs.StringVar(&c.RerankURL, "rerank-url", c.RerankURL, "vLLM server for rerank")
	fs.StringVar(&c.LLMModel, "llm-model", c.LLMModel, "LLM model name")
	fs.StringVar(&c.EmbedModel, "embed-model", c.EmbedModel, "embedding model name")
	fs.StringVar(&c.RerankModel, "rerank-model", c.RerankModel, "rerank model name")
	fs.IntVar(&c.KVCacheTokens, "kv-tokens", c.KVCacheTokens,
		"cap the LLM KV budget in tokens (0 = use what vLLM reports); lower it to force contention")
	fs.IntVar(&c.LLMMaxBatch, "llm-max-batch", c.LLMMaxBatch, "max concurrent LLM sequences (keep <= vLLM --max-num-seqs)")
	fs.IntVar(&c.EmbedBatchTokens, "embed-batch-tokens", c.EmbedBatchTokens, "max embed input tokens in flight")
	fs.IntVar(&c.RerankBatchTokens, "rerank-batch-tokens", c.RerankBatchTokens, "max rerank input tokens in flight")
}
