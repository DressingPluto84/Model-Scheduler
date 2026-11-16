package sched

import (
	"math"
	"testing"
	"time"

	"github.com/DressingPluto84/Model-Scheduler/internal/tenant"
)

func drrOrder(d *DRR, n int) string {
	out := ""
	for i := 0; i < n; i++ {
		r := d.Pick(nil)
		if r == nil {
			break
		}
		out += r.Tenant.ID
	}
	return out
}

func TestDRRRotatesTenantsPerRequestForLLM(t *testing.T) {
	d := NewDRR(tenant.LLM, 0)
	a := &tenant.Tenant{ID: "A", Tier: tenant.Free}
	b := &tenant.Tenant{ID: "B", Tier: tenant.Free}
	for i := 0; i < 3; i++ {
		d.Add(&Request{Tenant: a, InputTokens: 4000}, nil, false)
	}
	for i := 0; i < 3; i++ {
		d.Add(&Request{Tenant: b, InputTokens: 10}, nil, false)
	}
	if got := drrOrder(d, 10); got != "ABABAB" {
		t.Fatalf("order %q, want ABABAB (one request per turn, whatever its size)", got)
	}
}

func TestDRRChargesTokensForEmbed(t *testing.T) {
	d := NewDRR(tenant.Embed, 1000)
	a := &tenant.Tenant{ID: "A", Tier: tenant.Free}
	b := &tenant.Tenant{ID: "B", Tier: tenant.Free}
	for i := 0; i < 4; i++ {
		d.Add(&Request{Tenant: a, InputTokens: 600}, nil, false)
	}
	for i := 0; i < 6; i++ {
		d.Add(&Request{Tenant: b, InputTokens: 100}, nil, false)
	}
	// A: 1000 covers one 600 (400 left). B: 1000 covers all six 100s and
	// leaves the rotation. A: 400 + 1000 covers two, then one more turn.
	if got := drrOrder(d, 20); got != "ABBBBBBAAA" {
		t.Fatalf("order %q, want ABBBBBBAAA", got)
	}
	if d.Len() != 0 {
		t.Fatalf("len %d after draining", d.Len())
	}
}

func TestDRRServesHigherTierFirst(t *testing.T) {
	d := NewDRR(tenant.LLM, 0)
	f := &tenant.Tenant{ID: "f", Tier: tenant.Free}
	p := &tenant.Tenant{ID: "p", Tier: tenant.Premium}
	d.Add(&Request{Tenant: f}, nil, false)
	d.Add(&Request{Tenant: f}, nil, false)
	d.Add(&Request{Tenant: p}, nil, false)
	if got := drrOrder(d, 5); got != "pff" {
		t.Fatalf("order %q, want premium first", got)
	}
}

func TestDRRRemove(t *testing.T) {
	d := NewDRR(tenant.LLM, 0)
	a := &tenant.Tenant{ID: "A", Tier: tenant.Free}
	b := &tenant.Tenant{ID: "B", Tier: tenant.Free}
	ra := &Request{Tenant: a}
	d.Add(ra, nil, false)
	d.Add(&Request{Tenant: b}, nil, false)
	d.Remove(ra, nil)
	if got := drrOrder(d, 5); got != "B" || d.Len() != 0 {
		t.Fatalf("order %q len %d", got, d.Len())
	}
}

// tieredShare floods one premium and three free tenants with equal requests
// and returns premium's fraction of the service.
func tieredShare(t *testing.T, share TierShare) float64 {
	t.Helper()
	p, err := NewTiered(share)
	if err != nil {
		t.Fatal(err)
	}
	h := newHarness(p, 100, 1)
	prem := &tenant.Tenant{ID: "p", Tier: tenant.Premium}
	frees := []*tenant.Tenant{{ID: "f1", Tier: tenant.Free}, {ID: "f2", Tier: tenant.Free}, {ID: "f3", Tier: tenant.Free}}
	for i := 0; i < 200; i++ {
		h.submit(prem, 100)
		for _, f := range frees {
			h.submit(f, 100)
		}
	}
	h.lp.Run(h.lp.Now().Add(8 * time.Second)) // 80 requests of 100ms each
	st := h.s.Stats()
	var pt, total float64
	for tn, n := range st.Served {
		total += float64(n)
		if tn == prem {
			pt += float64(n)
		}
	}
	return pt / total
}

func TestTieredPerTenantShare(t *testing.T) {
	// One premium tenant (3) vs three free tenants (1 each): 50 / 50.
	if got := tieredShare(t, TierSharePerTenant); math.Abs(got-0.5) > 0.05 {
		t.Fatalf("premium share %.2f, want 0.50", got)
	}
}

func TestTieredPerTierShare(t *testing.T) {
	// The tiers as a whole split 3:1.
	if got := tieredShare(t, TierSharePerTier); math.Abs(got-0.75) > 0.05 {
		t.Fatalf("premium share %.2f, want 0.75", got)
	}
}

func TestNewPolicyNames(t *testing.T) {
	for _, name := range Policies {
		if _, err := NewPolicy(name, tenant.LLM, PolicyOptions{}); err != nil {
			t.Fatalf("%s: %v", name, err)
		}
	}
	if _, err := NewPolicy("nope", tenant.LLM, PolicyOptions{}); err == nil {
		t.Fatal("unknown policy accepted")
	}
}

func TestMixedQueuesAndEqualSplit(t *testing.T) {
	p := NewMixed()
	h := newHarness(p, 100, 1)
	prem := &tenant.Tenant{ID: "p", Tier: tenant.Premium}
	free := &tenant.Tenant{ID: "f", Tier: tenant.Free}
	var inA = map[tenant.Tier]int{}
	for i := 0; i < 400; i++ {
		for _, tn := range []*tenant.Tenant{prem, free} {
			if r := h.submit(tn, 100); r.lane == 0 {
				inA[tn.Tier]++
			}
		}
	}
	if f := float64(inA[tenant.Premium]) / 400; math.Abs(f-0.75) > 0.06 {
		t.Fatalf("%.2f of premium requests in queue A, want 0.75", f)
	}
	if f := float64(inA[tenant.Free]) / 400; math.Abs(f-0.25) > 0.06 {
		t.Fatalf("%.2f of free requests in queue A, want 0.25", f)
	}
	h.lp.Run(h.lp.Now().Add(20 * time.Second)) // 200 requests
	if a, b := p.served[0], p.served[1]; math.Abs(a-b) > 200 {
		t.Fatalf("queues served %.0f and %.0f, want an even split", a, b)
	}
}
