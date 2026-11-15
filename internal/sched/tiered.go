package sched

import (
	"fmt"

	"github.com/DressingPluto84/Model-Scheduler/internal/ledger"
	"github.com/DressingPluto84/Model-Scheduler/internal/tenant"
)

// TierShare is how the Tiered policy splits service between tiers.
type TierShare string

const (
	// TierSharePerTenant keeps the weight ratio per tenant: a tier's share
	// is its weight × its number of active tenants. With 3:1 weights, 1,500
	// premium and 3,764 free tenants split 54% / 46%.
	TierSharePerTenant TierShare = "per-tenant"
	// TierSharePerTier applies the weight ratio to the tiers as a whole:
	// premium gets 75% and free 25% with 3:1 weights, however many tenants
	// each has.
	TierSharePerTier TierShare = "per-tier"
)

// Tiered keeps one queue per tier, each ordered by EEVDF over that tier's
// tenants, and splits service between the queues by tokens.
//
// Each tier counts the tokens it has been served. The next request comes from
// the non-empty queue whose tier has the least served / share, where share
// (see TierShare) uses the tier's active tenant count right now, so the split
// follows tenants joining and leaving at once and holds even when there are
// far more tenants than requests served. A tier's input tokens are counted
// when its request is picked, so the dispatch queue can't fill up with one
// tier; output is counted as it is generated. A tier that was idle comes back
// level with the others instead of catching up on time it missed.
type Tiered struct {
	share  TierShare
	tiers  map[tenant.Tier]*EEVDF
	served map[tenant.Tier]float64
}

var tierOrder = []tenant.Tier{tenant.Premium, tenant.Free}

// NewTiered returns a tiered policy. An empty share means per-tenant.
func NewTiered(share TierShare) (*Tiered, error) {
	if share == "" {
		share = TierSharePerTenant
	}
	if share != TierSharePerTenant && share != TierSharePerTier {
		return nil, fmt.Errorf("unknown tier share %q (want %q or %q)", share, TierSharePerTenant, TierSharePerTier)
	}
	t := &Tiered{share: share, tiers: map[tenant.Tier]*EEVDF{}, served: map[tenant.Tier]float64{}}
	for _, tier := range tierOrder {
		e := NewEEVDF()
		e.tier = tier
		t.tiers[tier] = e
	}
	return t, nil
}

func (p *Tiered) Name() string { return "tiered" }

func (p *Tiered) Len() int {
	n := 0
	for _, e := range p.tiers {
		n += e.Len()
	}
	return n
}

// shareOf is a tier's share of service right now.
func (p *Tiered) shareOf(tier tenant.Tier, env *Env) float64 {
	w := tenant.TierWeights[tier]
	if p.share == TierSharePerTenant {
		w *= float64(max(1, env.ActiveTenants[tier]))
	}
	return w
}

func (p *Tiered) Add(r *Request, env *Env, waking bool) {
	tier := r.Tenant.Tier
	if env.ActiveTenants[tier] == 0 {
		// The tier was idle: bring it level with the other active tiers so
		// it doesn't get a burst to make up for the time it was away.
		for _, other := range tierOrder {
			if other == tier || env.ActiveTenants[other] == 0 {
				continue
			}
			level := p.served[other] / p.shareOf(other, env) * p.shareOf(tier, env)
			if level > p.served[tier] {
				p.served[tier] = level
			}
		}
	}
	p.tiers[tier].Add(r, env, waking)
}

func (p *Tiered) Remove(r *Request, env *Env) { p.tiers[r.Tenant.Tier].Remove(r, env) }

func (p *Tiered) Pick(env *Env) *Request {
	best := p.pickTier(env)
	if best == "" {
		return nil
	}
	r := p.tiers[best].Pick(env)
	p.served[best] += float64(r.InputTokens)
	return r
}

// pickTier returns the non-empty tier furthest behind, except that a rerank
// request that has started is finished before switching tiers.
func (p *Tiered) pickTier(env *Env) tenant.Tier {
	for _, tier := range tierOrder {
		if p.tiers[tier].hasStartedGroup() {
			return tier
		}
	}
	var best tenant.Tier
	var bestKey float64
	for _, tier := range tierOrder {
		if p.tiers[tier].Len() == 0 {
			continue
		}
		key := p.served[tier] / p.shareOf(tier, env)
		if best == "" || key < bestKey {
			best, bestKey = tier, key
		}
	}
	return best
}

// OnUsage counts generated output against the tier (input was counted when
// the request was picked).
func (p *Tiered) OnUsage(r *Request, _, output int, _ *Env) {
	p.served[r.Tenant.Tier] += float64(ledger.OutputWeight * output)
}
