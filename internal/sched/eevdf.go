package sched

import (
	"math"

	"github.com/DressingPluto84/Model-Scheduler/internal/rbtree"
	"github.com/DressingPluto84/Model-Scheduler/internal/tenant"
)

// EEVDF is the earliest eligible virtual deadline first policy.
//
//	vruntime  = input + 2 × output, for all tokens processed all time (global)
//	used      = vruntime / weight
//	eligible -> vruntime <= Σ vruntime × (w_i / Σ w)   (sums over enqueued tenants)
//	deadline  = used + input / weight
//	pick the smallest deadline among eligible requests
//
// Requests are kept in an RB tree ordered by (deadline, arrival). A request's
// deadline is fixed when it is enqueued. When a tenant has several requests
// queued at once, "used" for each one also includes the input of that
// tenant's requests already queued ahead of it, so a burst gets staggered
// deadlines instead of all sharing the same one.
type EEVDF struct {
	tree   *rbtree.Tree[*Request]
	queued map[*tenant.Tenant]int64
}

func NewEEVDF() *EEVDF {
	return &EEVDF{
		tree: rbtree.New(func(a, b *Request) bool {
			if a.deadline != b.deadline {
				return a.deadline < b.deadline
			}
			return a.seq < b.seq
		}),
		queued: make(map[*tenant.Tenant]int64),
	}
}

func (e *EEVDF) Name() string { return "eevdf" }
func (e *EEVDF) Len() int     { return e.tree.Len() }

func (e *EEVDF) Add(r *Request, env *Env, waking bool) {
	t := r.Tenant
	w := t.Weight()
	if waking {
		// Come back at max(current vruntime, minimum eligible value).
		if v, ok := MinEligible(t, env); ok {
			env.Ledger.LiftVRuntime(t, v)
		}
	}
	used := (env.EffectiveVRuntime(t) + float64(e.queued[t])) / w
	r.deadline = used + float64(r.InputTokens)/w
	e.queued[t] += int64(r.InputTokens)
	r.node = e.tree.Insert(r)
}

func (e *EEVDF) Remove(r *Request, _ *Env) {
	e.tree.Delete(r.node)
	r.node = nil
	e.unqueue(r)
}

func (e *EEVDF) unqueue(r *Request) {
	t := r.Tenant
	e.queued[t] -= int64(r.InputTokens)
	if e.queued[t] <= 0 {
		delete(e.queued, t)
	}
}

func (e *EEVDF) Pick(env *Env) *Request {
	if e.tree.Len() == 0 {
		return nil
	}
	sumS, sumW := activeSums(env, nil)
	eligible := make(map[*tenant.Tenant]bool)
	var pick *rbtree.Node[*Request]
	for n := e.tree.Min(); n != nil; n = e.tree.Next(n) {
		t := n.Value.Tenant
		ok, seen := eligible[t]
		if !seen {
			ok = isEligible(env.EffectiveVRuntime(t), t.Weight(), sumS, sumW)
			eligible[t] = ok
		}
		if ok {
			pick = n
			break
		}
	}
	if pick == nil {
		// Eligibility is relative to all enqueued tenants, including ones whose
		// work is already dispatched or running. If every tenant with queued
		// work is ineligible, stay work-conserving and take the earliest
		// deadline.
		pick = e.tree.Min()
	}
	r := pick.Value
	e.tree.Delete(pick)
	r.node = nil
	e.unqueue(r)
	return r
}

// Deadline exposes a queued request's virtual deadline (for tests/debugging).
func (r *Request) Deadline() float64 { return r.deadline }

// MinEligible returns Σ vruntime × w_t / Σ w over the other enqueued tenants:
// the smallest vruntime at which t would no longer be ahead of its fair share.
// ok is false when no other tenant is enqueued.
func MinEligible(t *tenant.Tenant, env *Env) (float64, bool) {
	sumS, sumW := activeSums(env, t)
	if sumW == 0 {
		return 0, false
	}
	// Rounded up so vruntime stays integral and sums stay exact (and
	// therefore deterministic regardless of map iteration order).
	return math.Ceil(sumS * t.Weight() / sumW), true
}

func activeSums(env *Env, exclude *tenant.Tenant) (sumS, sumW float64) {
	for t, n := range env.Active {
		if n <= 0 || t == exclude {
			continue
		}
		sumS += env.EffectiveVRuntime(t)
		sumW += t.Weight()
	}
	return sumS, sumW
}

func isEligible(s, w, sumS, sumW float64) bool {
	if sumW == 0 {
		return true
	}
	// s <= sumS × w / sumW, rearranged to avoid division.
	lhs, rhs := s*sumW, sumS*w
	return lhs <= rhs+1e-9*(1+rhs)
}
