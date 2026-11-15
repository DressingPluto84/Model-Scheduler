package sched

import (
	"github.com/DressingPluto84/Model-Scheduler/internal/ledger"
	"github.com/DressingPluto84/Model-Scheduler/internal/rbtree"
	"github.com/DressingPluto84/Model-Scheduler/internal/tenant"
)

// Env is the scheduler state a policy can read when making decisions.
type Env struct {
	Ledger *ledger.Ledger
	// Active counts each tenant's requests on this scheduler that are queued,
	// dispatched or running. A tenant with a positive count is "enqueued" for
	// the eligibility sums.
	Active map[*tenant.Tenant]int
	// Pending is input tokens picked into the dispatch queue but not yet
	// admitted (and so not yet charged). Policies add it to vruntime so a
	// tenant cannot fill the dispatch queue before its first charge lands.
	Pending map[*tenant.Tenant]int64
}

// EffectiveVRuntime is vruntime plus provisional pending input.
func (e *Env) EffectiveVRuntime(t *tenant.Tenant) float64 {
	return e.Ledger.VRuntime(t) + float64(e.Pending[t])
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

// NewPolicy returns a policy by name ("eevdf" or "fcfs").
func NewPolicy(name string) (Policy, bool) {
	switch name {
	case "eevdf":
		return NewEEVDF(), true
	case "fcfs":
		return NewFCFS(), true
	}
	return nil, false
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
