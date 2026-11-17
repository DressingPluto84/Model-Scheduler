package sched

import (
	"math"
	"sort"

	"github.com/DressingPluto84/Model-Scheduler/internal/rbtree"
	"github.com/DressingPluto84/Model-Scheduler/internal/tenant"
)

// EEVDF is the earliest eligible virtual deadline first policy.
//
//	vruntime  = input + 2 × output, for all tokens processed on this model
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
//
// The parts of a Group (rerank pairs) are the exception: they all share the
// deadline used + (whole request's input) / weight, and once one part is
// picked the remaining parts are picked next without another eligibility
// check, so a rerank request finishes together instead of being spread out
// behind other tenants' work.
type EEVDF struct {
	tree   *rbtree.Tree[*Request]
	queued map[*tenant.Tenant]int64
	// started holds groups that have had a part picked and still have parts
	// queued; their parts are picked before anything else.
	started []*Group
}

func NewEEVDF() *EEVDF {
	return &EEVDF{tree: rbtree.New(byDeadline), queued: make(map[*tenant.Tenant]int64)}
}

func byDeadline(a, b *Request) bool {
	if a.deadline != b.deadline {
		return a.deadline < b.deadline
	}
	return a.seq < b.seq
}

// Reset recomputes every queued request's deadline from the current
// vruntimes (called right after they were reset to 0), keeping arrival
// order. The rest of a rerank request that has already started still goes
// first.
func (e *EEVDF) Reset(env *Env) {
	var rs []*Request
	for n := e.tree.Min(); n != nil; n = e.tree.Next(n) {
		rs = append(rs, n.Value)
	}
	sort.Slice(rs, func(i, j int) bool { return rs[i].seq < rs[j].seq })
	for _, r := range rs {
		if g := r.Group; g != nil {
			g.queued = nil
			if !g.started {
				g.deadlineSet = false
			}
		}
	}
	e.tree = rbtree.New(byDeadline)
	e.queued = make(map[*tenant.Tenant]int64)
	for _, r := range rs {
		r.node = nil
		e.Add(r, env, false)
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
			env.Ledger.LiftVRuntime(t, env.Model, v)
		}
	}
	if g := r.Group; g != nil && g.deadlineSet {
		r.deadline = g.deadline
	} else {
		size := r.InputTokens
		if g != nil {
			size = g.InputTokens
		}
		used := (env.EffectiveVRuntime(t) + float64(e.queued[t])) / w
		r.deadline = used + float64(size)/w
		if g != nil {
			g.deadline, g.deadlineSet = r.deadline, true
		}
	}
	e.queued[t] += int64(r.InputTokens)
	r.node = e.tree.Insert(r)
	if r.Group != nil {
		r.Group.queued = append(r.Group.queued, r)
	}
}

func (e *EEVDF) Remove(r *Request, _ *Env) {
	e.tree.Delete(r.node)
	r.node = nil
	e.unqueue(r)
}

func (e *EEVDF) unqueue(r *Request) {
	if g := r.Group; g != nil {
		for i, q := range g.queued {
			if q == r {
				g.queued = append(g.queued[:i], g.queued[i+1:]...)
				break
			}
		}
	}
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
	// Finish a request that has already started before anything else.
	for len(e.started) > 0 {
		g := e.started[0]
		if len(g.queued) == 0 {
			e.started = e.started[1:]
			continue
		}
		return e.take(g.queued[0].node)
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
	return e.take(pick)
}

// take removes n from the queue and returns its request.
func (e *EEVDF) take(n *rbtree.Node[*Request]) *Request {
	r := n.Value
	e.tree.Delete(n)
	r.node = nil
	e.unqueue(r)
	if g := r.Group; g != nil && !g.started {
		g.started = true
		if len(g.queued) > 0 {
			e.started = append(e.started, g)
		}
	}
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

// activeSums sums effective vruntime and weight over enqueued tenants,
// leaving out exclude.
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
