package sched

import (
	"fmt"

	"github.com/DressingPluto84/Model-Scheduler/internal/ledger"
	"github.com/DressingPluto84/Model-Scheduler/internal/rbtree"
	"github.com/DressingPluto84/Model-Scheduler/internal/tenant"
)

// Env is the scheduler state a policy can read when making decisions.
type Env struct {
	Ledger *ledger.Ledger
	// Model is the model this scheduler serves; vruntime is kept per model.
	Model Model
	// Active counts each tenant's requests on this scheduler that are queued,
	// dispatched or running. A tenant with a positive count is "enqueued" for
	// the eligibility sums.
	Active map[*tenant.Tenant]int
	// Pending is input tokens picked into the dispatch queue but not yet
	// admitted (and so not yet charged). Policies add it to vruntime so a
	// tenant cannot fill the dispatch queue before its first charge lands.
	Pending map[*tenant.Tenant]int64
	// ActiveTenants counts, per tier, the tenants with a positive Active count.
	ActiveTenants map[tenant.Tier]int
}

// EffectiveVRuntime is the tenant's vruntime on this model plus provisional
// pending input.
func (e *Env) EffectiveVRuntime(t *tenant.Tenant) float64 {
	return e.Ledger.VRuntime(t, e.Model) + float64(e.Pending[t])
}

// Policy orders queued requests.
type Policy interface {
	Name() string
	// Add enqueues r. waking is true when the tenant is coming back after
	// having left (see ledger.IsAway).
	Add(r *Request, env *Env, waking bool)
	// Remove takes a queued request out (e.g. on cancellation).
	Remove(r *Request, env *Env)
	// Pick removes and returns the next request to dispatch, or nil.
	Pick(env *Env) *Request
	Len() int
}

// UsageObserver is a policy that wants to hear about tokens served (input
// when a request is admitted, output as it is generated).
type UsageObserver interface {
	OnUsage(r *Request, input, output int, env *Env)
}

// PolicyOptions configures the policies that need it.
type PolicyOptions struct {
	// TierShare is how the Tiered policy splits service between tiers:
	// TierSharePerTenant (default) or TierSharePerTier.
	TierShare TierShare
	// DRRTokenQuantum is the DRR quantum in tokens for embed and rerank
	// (about one full batch). LLM DRR counts requests with a quantum of 1.
	DRRTokenQuantum int
}

// Policies lists the policy names NewPolicy accepts.
var Policies = []string{"fcfs", "eevdf", "tiered", "mixed", "drr"}

// NewPolicy returns a policy by name for a model's scheduler:
//
//	fcfs    first come, first served (baseline)
//	eevdf   one EEVDF queue over all tenants
//	tiered  a premium queue and a free queue, each EEVDF over its tenants,
//	        with service split between them by tier weight
//	mixed   two EEVDF queues holding both tiers (A mostly premium, B mostly
//	        free, chosen at random per request), splitting service 50/50
//	drr     Cohere-style deficit round robin: tiers in priority order, DRR
//	        between the tenants of a tier, FIFO within a tenant
func NewPolicy(name string, model Model, opts PolicyOptions) (Policy, error) {
	switch name {
	case "fcfs":
		return NewFCFS(), nil
	case "eevdf":
		return NewEEVDF(), nil
	case "tiered":
		return NewTiered(opts.TierShare)
	case "mixed":
		return NewMixed(), nil
	case "drr":
		return NewDRR(model, opts.DRRTokenQuantum), nil
	}
	return nil, fmt.Errorf("unknown policy %q (want one of %v)", name, Policies)
}

// --- FCFS -----------------------------------------------------------------

// FCFS dispatches requests in arrival order. It is the benchmark baseline.
type FCFS struct {
	tree *rbtree.Tree[*Request]
}

func NewFCFS() *FCFS {
	return &FCFS{tree: rbtree.New(func(a, b *Request) bool { return a.seq < b.seq })}
}

func (f *FCFS) Name() string { return "fcfs" }
func (f *FCFS) Len() int     { return f.tree.Len() }

func (f *FCFS) Add(r *Request, _ *Env, _ bool) { r.node = f.tree.Insert(r) }

func (f *FCFS) Remove(r *Request, _ *Env) {
	f.tree.Delete(r.node)
	r.node = nil
}

func (f *FCFS) Pick(_ *Env) *Request {
	n := f.tree.Min()
	if n == nil {
		return nil
	}
	f.tree.Delete(n)
	n.Value.node = nil
	return n.Value
}
