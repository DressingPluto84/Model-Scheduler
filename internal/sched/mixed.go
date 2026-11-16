package sched

import (
	"math/rand"

	"github.com/DressingPluto84/Model-Scheduler/internal/ledger"
	"github.com/DressingPluto84/Model-Scheduler/internal/tenant"
)

// Mixed keeps two EEVDF queues that each hold both tiers. Each request picks
// a queue at random: premium goes to queue A with probability 0.75 (else B),
// free goes to A with probability 0.25 (else B), so A is mostly premium and
// B mostly free. The queues split service equally by tokens: the next
// request comes from the non-empty queue that has been served less. Inside a
// queue it is ordinary EEVDF with the tier weights.
//
// All pairs of a rerank request go to the same queue, and a rerank request
// that has started finishes before switching queues.
type Mixed struct {
	queues [2]*EEVDF
	served [2]float64
	rng    *rand.Rand
	lane   map[*Group]int // queue chosen for each rerank request
}

// premiumToA and freeToA are the chances a request goes to queue A.
const (
	premiumToA = 0.75
	freeToA    = 0.25
)

func NewMixed() *Mixed {
	return &Mixed{
		queues: [2]*EEVDF{NewEEVDF(), NewEEVDF()},
		rng:    rand.New(rand.NewSource(1)),
		lane:   map[*Group]int{},
	}
}

func (p *Mixed) Name() string { return "mixed" }
func (p *Mixed) Len() int     { return p.queues[0].Len() + p.queues[1].Len() }

func (p *Mixed) chooseQueue(r *Request) int {
	if g := r.Group; g != nil {
		if q, ok := p.lane[g]; ok {
			return q
		}
	}
	toA := freeToA
	if r.Tenant.Tier == tenant.Premium {
		toA = premiumToA
	}
	q := 1
	if p.rng.Float64() < toA {
		q = 0
	}
	if r.Group != nil {
		p.lane[r.Group] = q
	}
	return q
}

func (p *Mixed) Add(r *Request, env *Env, waking bool) {
	q := p.chooseQueue(r)
	r.lane = q
	if p.queues[q].Len() == 0 && p.queues[1-q].Len() > 0 && p.served[1-q] > p.served[q] {
		// An idle queue comes back level instead of catching up.
		p.served[q] = p.served[1-q]
	}
	p.queues[q].Add(r, env, waking)
}

func (p *Mixed) Remove(r *Request, env *Env) { p.queues[r.lane].Remove(r, env) }

func (p *Mixed) Pick(env *Env) *Request {
	q := -1
	for i, e := range p.queues {
		if e.hasStartedGroup() {
			q = i
		}
	}
	if q < 0 {
		for i, e := range p.queues {
			if e.Len() > 0 && (q < 0 || p.served[i] < p.served[q]) {
				q = i
			}
		}
	}
	if q < 0 {
		return nil
	}
	r := p.queues[q].Pick(env)
	p.served[q] += float64(r.InputTokens)
	if g := r.Group; g != nil && len(g.queued) == 0 {
		delete(p.lane, g)
	}
	return r
}

// OnUsage counts generated output against the request's queue.
func (p *Mixed) OnUsage(r *Request, _, output int, _ *Env) {
	p.served[r.lane] += float64(ledger.OutputWeight * output)
}
