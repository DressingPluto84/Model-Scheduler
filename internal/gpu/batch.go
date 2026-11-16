package gpu

import (
	"fmt"
	"time"

	"github.com/DressingPluto84/Model-Scheduler/internal/loop"
	"github.com/DressingPluto84/Model-Scheduler/internal/sched"
	"github.com/DressingPluto84/Model-Scheduler/internal/tenant"
)

// BatchConfig configures a simulated embed or rerank GPU.
type BatchConfig struct {
	// MaxBatchTokens is the token budget for one batch of in-flight work.
	MaxBatchTokens int
	// A batch takes Base + PerToken × total input tokens.
	Base     time.Duration
	PerToken time.Duration
}

// Validate checks the config can serve every allowed request.
func (c BatchConfig) Validate() error {
	if c.MaxBatchTokens < tenant.MaxInputTokens {
		return fmt.Errorf("batch gpu: MaxBatchTokens %d is smaller than the max request size %d",
			c.MaxBatchTokens, tenant.MaxInputTokens)
	}
	return nil
}

// Batch is a GPU that runs one batch of single-pass jobs (embeddings or
// doc-query rerank pairs) at a time. Jobs are admitted while no batch is
// running and the token budget allows; the batch starts once the current
// admission round finishes.
type Batch struct {
	cfg   BatchConfig
	loop  loop.Loop
	sched *sched.Scheduler

	used    int
	forming []*sched.Request
	running []*sched.Request
	busy    bool
	kicked  bool

	stats Stats
}

// NewBatch returns a simulated batch GPU.
func NewBatch(lp loop.Loop, cfg BatchConfig) *Batch {
	return &Batch{cfg: cfg, loop: lp}
}

func (g *Batch) Attach(s *sched.Scheduler) { g.sched = s }

// CanAdmit is true while no batch is running and r fits in the token budget.
// A job bigger than the whole budget (a large rerank) runs alone in a batch
// of its own rather than never fitting.
func (g *Batch) CanAdmit(r *sched.Request) bool {
	if g.busy {
		return false
	}
	return g.used+r.InputTokens <= g.cfg.MaxBatchTokens || g.used == 0
}

func (g *Batch) Admit(r *sched.Request) {
	g.used += r.InputTokens
	g.forming = append(g.forming, r)
	r.OnUsage(r.InputTokens, 0)
	if !g.kicked {
		g.kicked = true
		g.loop.Post(func() {
			g.kicked = false
			g.start()
		})
	}
}

func (g *Batch) start() {
	if g.busy || len(g.forming) == 0 {
		return
	}
	tokens := 0
	for _, r := range g.forming {
		tokens += r.InputTokens
	}
	g.running, g.forming = g.forming, nil
	g.busy = true
	dur := g.cfg.Base + time.Duration(tokens)*g.cfg.PerToken
	g.stats.Busy += dur
	g.stats.Steps++
	g.stats.SeqSteps += int64(len(g.running))
	g.stats.Tokens += int64(tokens)
	g.loop.After(dur, g.end)
}

func (g *Batch) end() {
	batch := g.running
	g.running = nil
	g.busy = false
	g.used = 0
	for _, r := range batch {
		if r.State() == sched.StateRunning {
			g.sched.Finish(r)
		}
	}
	g.sched.CapacityFreed()
}

// Abort finishes r right away. If r is part of a running batch its memory is
// released when the batch ends; otherwise it is released now.
func (g *Batch) Abort(r *sched.Request) {
	for i, q := range g.forming {
		if q == r {
			g.forming = append(g.forming[:i], g.forming[i+1:]...)
			g.used -= r.InputTokens
			break
		}
	}
	g.sched.Finish(r)
}

// Stats returns cumulative GPU statistics.
func (g *Batch) Stats() Stats {
	s := g.stats
	s.UsedTokens = g.used
	s.CapacityTokens = g.cfg.MaxBatchTokens
	return s
}
