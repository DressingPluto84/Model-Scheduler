package bench

import (
	"math"
	"math/rand"
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
	rep := run(t, 64, 144) // about 8× what one L40S serves within SLO
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
	w.Premium, w.Free, w.Sleeper = 150, 60, false
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

func TestGammaGapsKeepMeanAndSetBurstiness(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	for _, b := range []float64{0.5, 2, 3} {
		k := 1 / (b * b)
		var sum, sq float64
		const n = 200000
		for i := 0; i < n; i++ {
			x := gammaRand(rng, k) / k // mean 1
			sum += x
			sq += x * x
		}
		mean := sum / n
		cv := math.Sqrt(sq/n-mean*mean) / mean
		if math.Abs(mean-1) > 0.03 || math.Abs(cv-b)/b > 0.05 {
			t.Errorf("burstiness %g: gap mean %.3f (want 1), CV %.2f (want %g)", b, mean, cv, b)
		}
	}
}

func TestSkewKeepsAverageAndConcentratesTraffic(t *testing.T) {
	w := DefaultWorkload()
	w.Skew = 1
	xs := intensities(w, 100)
	var sum, top float64
	for i, x := range xs {
		sum += x
		if i < 10 {
			top += x
		}
	}
	mean := w.HeavyFraction*w.HeavyIntensity + (1-w.HeavyFraction)*w.LightIntensity
	if math.Abs(sum/100-mean) > 1e-9 {
		t.Fatalf("average intensity %.3f, want %.3f", sum/100, mean)
	}
	// Zipf(1) over 100 tenants: the top 10 send about 56% of the traffic.
	if share := top / sum; share < 0.5 || share > 0.62 {
		t.Fatalf("top 10%% of tenants send %.0f%%, want about 56%%", 100*share)
	}
	big := &tenant.Tenant{Tier: tenant.Premium, QuotaScale: xs[0]}
	if got, base := big.RateLimit(tenant.LLM), tenant.RateLimitPerMinute(tenant.Premium, tenant.LLM); got <= base {
		t.Fatalf("largest tenant's limit %d, want above the tier's %d", got, base)
	}
}
