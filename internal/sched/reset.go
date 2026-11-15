package sched

import (
	"math"

	"github.com/DressingPluto84/Model-Scheduler/internal/tenant"
)

// RatioReset configures the vruntime reset: each scheduler tracks how much
// service a premium tenant gets compared with a free tenant (tokens served
// per tenant of each tier since the last reset). When that ratio goes above
// Factor × the weight ratio (e.g. 4.5:1 for 3:1 weights with Factor 1.5),
// every tenant's vruntime on that model is reset to 0 and queued requests get
// fresh deadlines.
type RatioReset struct {
	// Factor is how far above the weight ratio the observed ratio may go
	// before a reset. 0 turns resets off.
	Factor float64
	// MinTokens is how much service (input + 2 × output, both tiers together)
	// must happen after a reset before the ratio is checked again, so a
	// handful of requests can't trigger one.
	MinTokens int64
}

// Resetter is a policy whose queued deadlines depend on vruntime and must be
// recomputed after a reset.
type Resetter interface {
	Reset(env *Env)
}

// SetRatioReset turns the vruntime reset on (Factor > 0) or off. It only has
// an effect for policies that use vruntime (EEVDF).
func (s *Scheduler) SetRatioReset(cfg RatioReset) { s.ratio.cfg = cfg }

type ratioTracker struct {
	cfg    RatioReset
	served map[tenant.Tier]int64                   // since the last reset
	seen   map[tenant.Tier]map[*tenant.Tenant]bool // tenants active since the last reset
	total  int64
	resets int
}

func (rt *ratioTracker) reset(active map[*tenant.Tenant]int) {
	rt.served = map[tenant.Tier]int64{}
	rt.seen = map[tenant.Tier]map[*tenant.Tenant]bool{tenant.Premium: {}, tenant.Free: {}}
	rt.total = 0
	for t, n := range active {
		if n > 0 {
			rt.see(t)
		}
	}
}

func (rt *ratioTracker) see(t *tenant.Tenant) { rt.seen[t.Tier][t] = true }

func (rt *ratioTracker) add(t *tenant.Tenant, amount int64) {
	rt.served[t.Tier] += amount
	rt.total += amount
}

// observed returns premium tokens per premium tenant divided by free tokens
// per free tenant since the last reset. ok is false when there isn't enough
// data to judge yet.
func (rt *ratioTracker) observed() (ratio float64, ok bool) {
	np, nf := len(rt.seen[tenant.Premium]), len(rt.seen[tenant.Free])
	if np == 0 || nf == 0 || rt.total < rt.cfg.MinTokens {
		return 0, false
	}
	perP := float64(rt.served[tenant.Premium]) / float64(np)
	perF := float64(rt.served[tenant.Free]) / float64(nf)
	if perF == 0 {
		if perP == 0 {
			return 0, false
		}
		return math.Inf(1), true
	}
	return perP / perF, true
}

func (s *Scheduler) maybeResetVRuntimes() {
	rt := &s.ratio
	if rt.cfg.Factor <= 0 {
		return
	}
	resetter, ok := s.policy.(Resetter)
	if !ok {
		return
	}
	ratio, ok := rt.observed()
	target := tenant.TierWeights[tenant.Premium] / tenant.TierWeights[tenant.Free]
	if !ok || ratio <= rt.cfg.Factor*target {
		return
	}
	s.ledger.ResetVRuntimes(s.model)
	resetter.Reset(s.env)
	rt.resets++
	rt.reset(s.env.Active)
}
