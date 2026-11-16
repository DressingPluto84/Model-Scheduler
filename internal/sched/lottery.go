package sched

import (
	"math/rand"

	"github.com/DressingPluto84/Model-Scheduler/internal/tenant"
)

// Lottery keeps a premium queue and a free queue, each ordered by EEVDF over
// that tier's tenants, like Tiered. Instead of tracking how much each tier
// has been served, every pick is a coin flip: the premium queue with
// probability 0.75, the free queue with 0.25. If the chosen queue is empty
// the other one is used, so no GPU time is wasted.
//
// A rerank request that has started (pairs mode) finishes before the other
// queue is drawn.
type Lottery struct {
	tiers map[tenant.Tier]*EEVDF
	rng   *rand.Rand
}

// premiumChance is the chance a pick draws the premium queue.
const premiumChance = 0.75

func NewLottery() *Lottery {
	p := &Lottery{tiers: map[tenant.Tier]*EEVDF{}, rng: rand.New(rand.NewSource(1))}
	for _, tier := range tierOrder {
		e := NewEEVDF()
		e.tier = tier
		p.tiers[tier] = e
	}
	return p
}

func (p *Lottery) Name() string { return "lottery" }

func (p *Lottery) Len() int { return p.tiers[tenant.Premium].Len() + p.tiers[tenant.Free].Len() }

func (p *Lottery) Add(r *Request, env *Env, waking bool) {
	p.tiers[r.Tenant.Tier].Add(r, env, waking)
}

func (p *Lottery) Remove(r *Request, env *Env) { p.tiers[r.Tenant.Tier].Remove(r, env) }

func (p *Lottery) Pick(env *Env) *Request {
	prem, free := p.tiers[tenant.Premium], p.tiers[tenant.Free]
	var q *EEVDF
	switch {
	case prem.hasStartedGroup():
		q = prem
	case free.hasStartedGroup():
		q = free
	case prem.Len() == 0 && free.Len() == 0:
		return nil
	case free.Len() == 0:
		q = prem
	case prem.Len() == 0:
		q = free
	case p.rng.Float64() < premiumChance:
		q = prem
	default:
		q = free
	}
	return q.Pick(env)
}
