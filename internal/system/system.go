// Package system wires the gateway, services, schedulers and GPUs together.
// Everything in it runs on the event loop: callers outside the loop must use
// loop.Post / Real.Do.
package system

import (
	"fmt"
	"time"

	"github.com/DressingPluto84/Model-Scheduler/internal/gpu"
	"github.com/DressingPluto84/Model-Scheduler/internal/ledger"
	"github.com/DressingPluto84/Model-Scheduler/internal/loop"
	"github.com/DressingPluto84/Model-Scheduler/internal/ratelimit"
	"github.com/DressingPluto84/Model-Scheduler/internal/sched"
	"github.com/DressingPluto84/Model-Scheduler/internal/tenant"
)

// Config configures the whole system.
type Config struct {
	// Policy is the scheduling policy: one of sched.Policies.
	Policy string
	// PolicyOptions configures the drr policy.
	PolicyOptions sched.PolicyOptions
	// DispatchDepth is the size of each scheduler's dispatch queue.
	DispatchDepth int
	// IdleTimeout is how long a tenant with nothing queued or running must be
	// quiet before it counts as having left.
	IdleTimeout time.Duration
	// BillingWindow is the rolling billing period (24h).
	BillingWindow time.Duration
	// BillingEnabled turns the billing cap check on.
	BillingEnabled bool
	// RateLimitWindow is the gateway's sliding window (1 minute).
	RateLimitWindow time.Duration
	// DefaultMaxTokens is used for LLM requests that don't set max_tokens.
	DefaultMaxTokens int
	// RerankPairs splits each rerank request into one scheduler request per
	// doc-query pair. Off by default: a rerank request is one job.
	RerankPairs bool
	// RatioReset resets a model's vruntimes when premium's share drifts too
	// far above the weight ratio. Off when Factor is 0 (the default).
	RatioReset sched.RatioReset

	LLM    gpu.LLMConfig
	Embed  gpu.BatchConfig
	Rerank gpu.BatchConfig
}

// DefaultConfig returns the MVP defaults. The simulated GPUs are calibrated
// to one L40S running Qwen2.5-7B, bge-base and bge-reranker-v2-m3 with a
// 70/10/10 memory split (measured with deploy/runpod/profile.py):
//
//   - Embed and rerank keep up to 16,384 input tokens in flight, like the vLLM
//     backend, and take Base + PerToken × tokens.
//   - The LLM's KV budget and batch cap are what vLLM reported. StepBase and
//     PrefillPerToken are from the profile. DecodePerSeq is fitted to a run
//     with all three models busy (122ms per step at 256 sequences): the models
//     share one GPU, and alone a step at 256 sequences takes only 40ms.
func DefaultConfig() Config {
	return Config{
		Policy:           "eevdf",
		PolicyOptions:    sched.PolicyOptions{DRRTokenQuantum: 4096},
		DispatchDepth:    4,
		IdleTimeout:      30 * time.Second,
		BillingWindow:    24 * time.Hour,
		BillingEnabled:   true,
		RateLimitWindow:  time.Minute,
		DefaultMaxTokens: 512,
		LLM: gpu.LLMConfig{
			KVCacheTokens:   266547,
			MaxBatch:        256,
			StepBase:        20600 * time.Microsecond,
			DecodePerSeq:    320 * time.Microsecond,
			PrefillPerToken: 80 * time.Microsecond,
			FlushEvery:      32,
			OutputMedian:    200,
			OutputSigma:     0.8,
			Seed:            1,
		},
		Embed: gpu.BatchConfig{
			MaxBatchTokens: 16384,
			Base:           14 * time.Millisecond,
			PerToken:       5700 * time.Nanosecond,
		},
		Rerank: gpu.BatchConfig{
			MaxBatchTokens: 16384,
			Base:           37 * time.Millisecond,
			PerToken:       4500 * time.Nanosecond,
		},
	}
}

// System is the single-process MVP: gateway, services, schedulers, GPUs.
type System struct {
	cfg     Config
	loop    loop.Loop
	ledger  *ledger.Ledger
	tenants *tenant.Registry
	limiter *ratelimit.Limiter

	scheds map[sched.Model]*sched.Scheduler
	gpus   map[sched.Model]GPU

	nextID uint64
}

// GPU is a scheduler-facing GPU that also reports statistics.
type GPU interface {
	sched.GPU
	Stats() gpu.Stats
}

// GPUs is the GPU for each model. A nil field gets a simulated GPU built
// from Config.
type GPUs struct {
	LLM, Embed, Rerank GPU
}

// New builds a system on lp. Pass a zero GPUs to use simulated GPUs.
func New(lp loop.Loop, tenants *tenant.Registry, cfg Config, gpus GPUs) (*System, error) {
	if gpus.LLM == nil {
		if err := cfg.LLM.Validate(); err != nil {
			return nil, err
		}
		gpus.LLM = gpu.NewLLM(lp, cfg.LLM)
	}
	if gpus.Embed == nil {
		if err := cfg.Embed.Validate(); err != nil {
			return nil, fmt.Errorf("embed: %w", err)
		}
		gpus.Embed = gpu.NewBatch(lp, cfg.Embed)
	}
	if gpus.Rerank == nil {
		if err := cfg.Rerank.Validate(); err != nil {
			return nil, fmt.Errorf("rerank: %w", err)
		}
		gpus.Rerank = gpu.NewBatch(lp, cfg.Rerank)
	}
	if cfg.DefaultMaxTokens < 1 || cfg.DefaultMaxTokens > tenant.MaxOutputTokens {
		return nil, fmt.Errorf("DefaultMaxTokens must be in [1, %d]", tenant.MaxOutputTokens)
	}
	s := &System{
		cfg:     cfg,
		loop:    lp,
		ledger:  ledger.New(cfg.IdleTimeout, cfg.BillingWindow),
		tenants: tenants,
		limiter: ratelimit.New(cfg.RateLimitWindow),
		scheds:  make(map[sched.Model]*sched.Scheduler),
		gpus: map[sched.Model]GPU{
			tenant.LLM:    gpus.LLM,
			tenant.Embed:  gpus.Embed,
			tenant.Rerank: gpus.Rerank,
		},
	}
	for _, m := range tenant.Endpoints {
		p, err := sched.NewPolicy(cfg.Policy, m, cfg.PolicyOptions)
		if err != nil {
			return nil, err
		}
		s.scheds[m] = sched.New(m, lp, s.ledger, p, s.gpus[m], cfg.DispatchDepth)
		s.scheds[m].SetRatioReset(cfg.RatioReset)
	}
	return s, nil
}

// Loop returns the system's event loop.
func (s *System) Loop() loop.Loop { return s.loop }

// Ledger returns the accounting ledger.
func (s *System) Ledger() *ledger.Ledger { return s.ledger }

// Tenants returns the tenant registry.
func (s *System) Tenants() *tenant.Registry { return s.tenants }

// Scheduler returns the scheduler for a model.
func (s *System) Scheduler(m sched.Model) *sched.Scheduler { return s.scheds[m] }

// GPUStats returns cumulative stats for a model's GPU.
func (s *System) GPUStats(m sched.Model) gpu.Stats { return s.gpus[m].Stats() }

// Config returns the system configuration.
func (s *System) Config() Config { return s.cfg }

func (s *System) newRequest(t *tenant.Tenant, m sched.Model, input int, payload any) *sched.Request {
	s.nextID++
	r := &sched.Request{ID: s.nextID, Tenant: t, Model: m, InputTokens: input, Payload: payload,
		Arrival: s.loop.Now()}
	// Services are the source of truth for token counts: the GPU reports
	// usage to the service, which records it in the ledger the schedulers
	// read from.
	r.OnUsage = func(in, out int) { s.ledger.Charge(t, m, in, out, s.loop.Now()) }
	return r
}
