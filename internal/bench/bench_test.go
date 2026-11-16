package bench

import (
	"testing"
	"time"

	"github.com/DressingPluto84/Model-Scheduler/internal/system"
	"github.com/DressingPluto84/Model-Scheduler/internal/tenant"
)

func run(t *testing.T, premium, free int) *Report {
	t.Helper()
	cfg := system.DefaultConfig()
	cfg.BillingEnabled = false
	w := DefaultWorkload()
	w.Duration = 2 * time.Minute
	w.Premium, w.Free, w.Sleeper = premium, free, false
	rep, err := Run(cfg, w, Backend{})
	if err != nil {
		t.Fatal(err)
	}
	return rep
}

func goodput(rep *Report, m tenant.Endpoint) float64 {
	var met, n int
	for _, tier := range []tenant.Tier{tenant.Premium, tenant.Free} {
		s := rep.Stats[statKey{m, tier}]
		met += s.SLOMet
		n += s.SLOMet + s.SLOMissed
	}
	return float64(met) / float64(n)
}

func TestGoodputHighWhenLightlyLoaded(t *testing.T) {
	rep := run(t, 1, 2)
	for _, m := range tenant.Endpoints {
		if g := goodput(rep, m); g < 0.9 {
			t.Errorf("%s goodput %.2f with 3 tenants, want >= 0.9", m, g)
		}
	}
}

func TestGoodputLowWhenOverloaded(t *testing.T) {
	rep := run(t, 30, 90)
	if g := goodput(rep, tenant.LLM); g > 0.2 {
		t.Errorf("llm goodput %.2f at heavy overload, want <= 0.2", g)
	}
}

func TestTierLagSignUnderStrictPriority(t *testing.T) {
	cfg := system.DefaultConfig()
	cfg.Policy = "drr"
	cfg.BillingEnabled = false
	w := DefaultWorkload()
	w.Duration = 2 * time.Minute
	w.Premium, w.Free, w.Sleeper = 30, 60, false
	rep, err := Run(cfg, w, Backend{})
	if err != nil {
		t.Fatal(err)
	}
	// Premium alone overloads the LLM, so strict priority puts premium ahead
	// of its 3:1 share and leaves free owed.
	p := meanF(rep.Stats[statKey{tenant.LLM, tenant.Premium}].TierLagSamples)
	f := meanF(rep.Stats[statKey{tenant.LLM, tenant.Free}].TierLagSamples)
	if p >= 0 || f <= 0 {
		t.Fatalf("tier lag premium %+.0f free %+.0f, want premium < 0 < free", p, f)
	}
}
