package sched

import "github.com/DressingPluto84/Model-Scheduler/internal/tenant"

// tierOrder lists the tiers from highest priority to lowest.
var tierOrder = []tenant.Tier{tenant.Premium, tenant.Free}

// DRR is deficit round robin as described in Cohere's "serving fairness"
// post (https://cohere.com/blog/serving-fairness):
//
//  1. Tier: higher tiers are considered first. The post says higher tiers get
//     "greater processing priority" and lower tiers are served "as capacity
//     allows", so this implementation serves free only when no premium
//     request is queued.
//  2. Tenant: within a tier, tenants take turns. On its turn a tenant's
//     deficit grows by one quantum, and it dispatches requests while the
//     deficit covers the next request's cost. Then the turn passes on.
//  3. Within a tenant: arrival order (our requests carry no priority or
//     client deadline).
//
// Cost and quantum depend on the model, as in the post: LLM (streaming) costs
// 1 per request with a quantum of 1, so tenants rotate after about one
// request each. Embed and rerank cost their input tokens, with a quantum of
// about one full batch. A tenant whose queue empties leaves the rotation and
// loses its leftover deficit (standard DRR; the post doesn't say). Tenant
// weights play no part, only the tier order.
type DRR struct {
	tokenCost bool
	quantum   float64
	rings     map[tenant.Tier]*drrRing
	n         int
}

type drrRing struct {
	active   []*drrQueue // tenants with queued requests, in rotation order
	cur      int
	byTenant map[*tenant.Tenant]*drrQueue
}

type drrQueue struct {
	reqs    []*Request
	deficit float64
	inTurn  bool // its quantum for the current turn has been added
}

// NewDRR returns a DRR policy for model. tokenQuantum is the quantum in
// tokens for embed and rerank (default 4096).
func NewDRR(model Model, tokenQuantum int) *DRR {
	d := &DRR{rings: map[tenant.Tier]*drrRing{}}
	if model == tenant.LLM {
		d.quantum = 1
	} else {
		d.tokenCost = true
		if tokenQuantum <= 0 {
			tokenQuantum = 4096
		}
		d.quantum = float64(tokenQuantum)
	}
	for _, tier := range tierOrder {
		d.rings[tier] = &drrRing{byTenant: map[*tenant.Tenant]*drrQueue{}}
	}
	return d
}

func (d *DRR) Name() string { return "drr" }
func (d *DRR) Len() int     { return d.n }

func (d *DRR) cost(r *Request) float64 {
	if d.tokenCost {
		return float64(r.InputTokens)
	}
	return 1
}

func (d *DRR) Add(r *Request, _ *Env, _ bool) {
	ring := d.rings[r.Tenant.Tier]
	q := ring.byTenant[r.Tenant]
	if q == nil {
		q = &drrQueue{}
		ring.byTenant[r.Tenant] = q
	}
	if len(q.reqs) == 0 {
		ring.active = append(ring.active, q)
	}
	q.reqs = append(q.reqs, r)
	d.n++
}

func (d *DRR) Remove(r *Request, _ *Env) {
	ring := d.rings[r.Tenant.Tier]
	q := ring.byTenant[r.Tenant]
	if q == nil {
		return
	}
	for i, x := range q.reqs {
		if x == r {
			q.reqs = append(q.reqs[:i], q.reqs[i+1:]...)
			d.n--
			break
		}
	}
	if len(q.reqs) == 0 {
		ring.deactivate(q)
	}
}

func (d *DRR) Pick(_ *Env) *Request {
	for _, tier := range tierOrder {
		if ring := d.rings[tier]; len(ring.active) > 0 {
			d.n--
			return ring.next(d)
		}
	}
	return nil
}

// next serves the tenant whose turn it is, moving the turn on when its
// deficit can't cover its next request.
func (ring *drrRing) next(d *DRR) *Request {
	for {
		q := ring.active[ring.cur]
		if !q.inTurn {
			q.deficit += d.quantum
			q.inTurn = true
		}
		r := q.reqs[0]
		if c := d.cost(r); q.deficit >= c {
			q.deficit -= c
			q.reqs = q.reqs[1:]
			if len(q.reqs) == 0 {
				ring.deactivate(q)
			}
			return r
		}
		q.inTurn = false
		ring.cur = (ring.cur + 1) % len(ring.active)
	}
}

// deactivate takes an empty queue out of the rotation and clears its deficit.
func (ring *drrRing) deactivate(q *drrQueue) {
	q.deficit, q.inTurn = 0, false
	for i, x := range ring.active {
		if x != q {
			continue
		}
		ring.active = append(ring.active[:i], ring.active[i+1:]...)
		if i < ring.cur {
			ring.cur--
		}
		if ring.cur >= len(ring.active) {
			ring.cur = 0
		}
		return
	}
}
