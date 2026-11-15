// Package gpu contains simulated GPUs. They model memory capacity and timing
// only; there is no real model behind them.
package gpu

import (
	"fmt"
	"math"
	"math/rand"
	"time"

	"github.com/DressingPluto84/Model-Scheduler/internal/loop"
	"github.com/DressingPluto84/Model-Scheduler/internal/sched"
	"github.com/DressingPluto84/Model-Scheduler/internal/tenant"
)

// LLMConfig configures the simulated LLM GPU.
type LLMConfig struct {
	// KVCacheTokens is the KV-cache budget. Requests reserve input + max_tokens.
	KVCacheTokens int
	// MaxBatch caps the number of sequences decoding at once.
	MaxBatch int
	// A step takes StepBase + DecodePerSeq × sequences + PrefillPerToken ×
	// tokens being prefilled this step.
	StepBase        time.Duration
	DecodePerSeq    time.Duration
	PrefillPerToken time.Duration
	// FlushEvery is k: output usage is reported in batches of this many tokens.
	FlushEvery int
	// Output lengths are sampled from a lognormal distribution with this
	// median and sigma, then capped at the request's max_tokens.
	OutputMedian float64
	OutputSigma  float64
	Seed         int64
}

// Validate checks the config can serve every allowed request.
func (c LLMConfig) Validate() error {
	if c.KVCacheTokens < tenant.MaxInputTokens+tenant.MaxOutputTokens {
		return fmt.Errorf("llm: KVCacheTokens %d is smaller than the largest reservation %d",
			c.KVCacheTokens, tenant.MaxInputTokens+tenant.MaxOutputTokens)
	}
	if c.MaxBatch < 1 || c.FlushEvery < 1 || c.OutputMedian < 1 {
		return fmt.Errorf("llm: MaxBatch, FlushEvery and OutputMedian must be positive")
	}
	return nil
}

// LLM is a continuous-batching LLM GPU.
type LLM struct {
	cfg   LLMConfig
	loop  loop.Loop
	sched *sched.Scheduler
	rng   *rand.Rand

	used     int
	prefill  []*sched.Request // admitted, waiting for the next step to prefill
	running  []*sched.Request // decoding
	stepping bool
	kicked   bool

	stats Stats
}

// NewLLM returns a simulated LLM GPU.
func NewLLM(lp loop.Loop, cfg LLMConfig) *LLM {
	return &LLM{cfg: cfg, loop: lp, rng: rand.New(rand.NewSource(cfg.Seed))}
}

func (g *LLM) Attach(s *sched.Scheduler) { g.sched = s }

func (g *LLM) CanAdmit(r *sched.Request) bool {
	return g.used+r.Reservation() <= g.cfg.KVCacheTokens &&
		len(g.prefill)+len(g.running) < g.cfg.MaxBatch
}

func (g *LLM) Admit(r *sched.Request) {
	g.used += r.Reservation()
	if r.OutputLen <= 0 {
		r.OutputLen = g.sampleOutputLen()
	}
	if r.OutputLen > r.MaxTokens {
		r.OutputLen = r.MaxTokens
	}
	if r.OutputLen < 1 {
		r.OutputLen = 1
	}
	g.prefill = append(g.prefill, r)
	r.OnUsage(r.InputTokens, 0) // input is charged when admitted to the GPU
	g.kick()
}

func (g *LLM) sampleOutputLen() int {
	v := g.cfg.OutputMedian * math.Exp(g.cfg.OutputSigma*g.rng.NormFloat64())
	return int(math.Max(1, math.Round(v)))
}

// kick starts a step once the current callback finishes, so every request
// admitted in the same pump is prefilled together.
func (g *LLM) kick() {
	if g.stepping || g.kicked {
		return
	}
	g.kicked = true
	g.loop.Post(func() {
		g.kicked = false
		g.startStep()
	})
}

func (g *LLM) startStep() {
	if g.stepping || (len(g.prefill) == 0 && len(g.running) == 0) {
		return
	}
	prefillTokens := 0
	for _, r := range g.prefill {
		prefillTokens += r.InputTokens
	}
	g.running = append(g.running, g.prefill...)
	g.prefill = nil
	batch := append([]*sched.Request(nil), g.running...)

	dur := g.cfg.StepBase +
		time.Duration(len(batch))*g.cfg.DecodePerSeq +
		time.Duration(prefillTokens)*g.cfg.PrefillPerToken
	g.stepping = true
	g.stats.Busy += dur
	g.stats.Steps++
	g.stats.SeqSteps += int64(len(batch))
	g.loop.After(dur, func() { g.endStep(batch) })
}

func (g *LLM) endStep(batch []*sched.Request) {
	g.stepping = false
	freed := false
	for _, r := range batch {
		if r.State() != sched.StateRunning {
			continue // aborted during the step
		}
		idx := r.Generated
		r.Generated++
		g.addUnflushed(r)
		g.stats.Tokens++
		if r.OnToken != nil {
			r.OnToken(idx, "")
		}
		if r.State() != sched.StateRunning {
			continue // cancelled from inside OnToken
		}
		if r.Generated >= r.OutputLen {
			g.release(r)
			g.sched.Finish(r)
			freed = true
		}
	}
	if freed {
		g.sched.CapacityFreed()
	}
	g.startStep()
}

func (g *LLM) addUnflushed(r *sched.Request) {
	if n := r.AddUnflushed(1); n >= g.cfg.FlushEvery {
		r.FlushUsage()
	}
}

// release frees r's reservation, flushes its remaining output usage and
// removes it from the batch.
func (g *LLM) release(r *sched.Request) {
	r.FlushUsage()
	g.used -= r.Reservation()
	g.running = remove(g.running, r)
	g.prefill = remove(g.prefill, r)
}

func (g *LLM) Abort(r *sched.Request) {
	g.release(r)
	g.sched.Finish(r)
}

// Stats returns cumulative GPU statistics.
func (g *LLM) Stats() Stats {
	s := g.stats
	s.UsedTokens = g.used
	s.CapacityTokens = g.cfg.KVCacheTokens
	return s
}

func remove(rs []*sched.Request, r *sched.Request) []*sched.Request {
	for i, q := range rs {
		if q == r {
			return append(rs[:i], rs[i+1:]...)
		}
	}
	return rs
}

// Stats are cumulative counters for a simulated GPU.
type Stats struct {
	Busy           time.Duration
	Steps          int64
	SeqSteps       int64 // LLM: Σ batch size over steps; batch GPUs: requests processed
	Tokens         int64 // LLM: output tokens; batch GPUs: input tokens
	UsedTokens     int
	CapacityTokens int
}
