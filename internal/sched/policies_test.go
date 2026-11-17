package sched

import (
	"testing"

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
