package sched

import (
	"math"
	"testing"
	"time"

	"github.com/DressingPluto84/Model-Scheduler/internal/ledger"
	"github.com/DressingPluto84/Model-Scheduler/internal/loop"
	"github.com/DressingPluto84/Model-Scheduler/internal/tenant"
)

// fakeGPU has a token budget and runs each request for 1ms per input token.
type fakeGPU struct {
	lp       *loop.Sim
	s        *Scheduler
	capacity int
	used     int
	admitted []*Request
}

func (g *fakeGPU) Attach(s *Scheduler)      { g.s = s }
func (g *fakeGPU) CanAdmit(r *Request) bool { return g.used+r.InputTokens <= g.capacity }
func (g *fakeGPU) Abort(r *Request)         { g.used -= r.InputTokens; g.s.Finish(r) }
func (g *fakeGPU) Admit(r *Request) {
	g.used += r.InputTokens
	g.admitted = append(g.admitted, r)
	r.OnUsage(r.InputTokens, 0)
	g.lp.After(time.Duration(r.InputTokens)*time.Millisecond, func() {
		if r.State() != StateRunning {
			return
		}
		g.used -= r.InputTokens
		g.s.Finish(r)
		g.s.CapacityFreed()
	})
}

type harness struct {
	lp  *loop.Sim
	l   *ledger.Ledger
	gpu *fakeGPU
	s   *Scheduler
	id  uint64
}

func newHarness(policy Policy, capacity, depth int) *harness {
	lp := loop.NewSim(time.Unix(0, 0))
	l := ledger.New(30*time.Second, 24*time.Hour)
	g := &fakeGPU{lp: lp, capacity: capacity}
	return &harness{lp: lp, l: l, gpu: g, s: New(tenant.Embed, lp, l, policy, g, depth)}
}

func (h *harness) submit(t *tenant.Tenant, input int) *Request {
	h.id++
	r := &Request{ID: h.id, Tenant: t, Model: tenant.Embed, InputTokens: input}
	r.OnUsage = func(in, out int) { h.l.Charge(t, tenant.Embed, in, out, h.lp.Now()) }
	h.s.Submit(r)
	return r
}

func TestEEVDFWeightedShare(t *testing.T) {
	h := newHarness(NewEEVDF(), 100, 1)
	prem := &tenant.Tenant{ID: "p", Tier: tenant.Premium}
	free := &tenant.Tenant{ID: "f", Tier: tenant.Free}
	for i := 0; i < 60; i++ {
		h.submit(free, 100)
		h.submit(prem, 100)
	}
	// One request at a time, 100ms each: 40 requests complete in 4s.
	h.lp.Run(h.lp.Now().Add(4 * time.Second))
	p, f := h.l.VRuntime(prem, tenant.Embed), h.l.VRuntime(free, tenant.Embed)
	ratio := p / f
	if ratio < 2.5 || ratio > 3.5 {
		t.Fatalf("premium:free service = %.0f:%.0f (ratio %.2f), want about 3", p, f, ratio)
	}
}

func TestFCFSIgnoresWeights(t *testing.T) {
	h := newHarness(NewFCFS(), 100, 1)
	prem := &tenant.Tenant{ID: "p", Tier: tenant.Premium}
	free := &tenant.Tenant{ID: "f", Tier: tenant.Free}
	for i := 0; i < 60; i++ {
		h.submit(free, 100)
		h.submit(prem, 100)
	}
	h.lp.Run(h.lp.Now().Add(4 * time.Second))
	if p, f := h.l.VRuntime(prem, tenant.Embed), h.l.VRuntime(free, tenant.Embed); math.Abs(p-f) > 100 {
		t.Fatalf("fcfs should alternate: premium %.0f free %.0f", p, f)
	}
}

func TestWakingTenantIsLifted(t *testing.T) {
	h := newHarness(NewEEVDF(), 100, 1)
	a := &tenant.Tenant{ID: "a", Tier: tenant.Free}
	b := &tenant.Tenant{ID: "b", Tier: tenant.Premium}
	for i := 0; i < 20; i++ {
		h.submit(a, 100)
	}
	h.lp.Run(h.lp.Now().Add(1 * time.Second)) // a has used ~1000
	// a is the only other enqueued tenant: Σ S × w_b / Σ w = S_a × 3 / 1.
	env := h.s.env
	want := math.Ceil(env.EffectiveVRuntime(a) * 3)
	h.submit(b, 100)
	if got := h.l.VRuntime(b, tenant.Embed); got != want {
		t.Fatalf("b came back at vruntime %.0f, want %.0f", got, want)
	}
	// A tenant that is still active (not away) is not lifted again.
	before := h.l.VRuntime(b, tenant.Embed)
	h.submit(b, 100)
	if h.l.VRuntime(b, tenant.Embed) != before {
		t.Fatalf("active tenant was lifted again")
	}
}

func TestIdleTimeoutMakesTenantAway(t *testing.T) {
	h := newHarness(NewEEVDF(), 100, 1)
	a := &tenant.Tenant{ID: "a", Tier: tenant.Free}
	h.submit(a, 100)
	h.lp.Run(h.lp.Now().Add(time.Second))
	if h.l.IsAway(a, tenant.Embed, h.lp.Now()) {
		t.Fatalf("tenant away right after finishing")
	}
	if !h.l.IsAway(a, tenant.Embed, h.lp.Now().Add(30*time.Second)) {
		t.Fatalf("tenant not away after idle timeout")
	}
}

func TestHeadOfLineWaitsUntilItFits(t *testing.T) {
	h := newHarness(NewFCFS(), 100, 4)
	a := &tenant.Tenant{ID: "a", Tier: tenant.Free}
	h.submit(a, 60) // runs
	big := h.submit(a, 50)
	small := h.submit(a, 10)
	if big.State() != StateDispatched || small.State() != StateDispatched {
		t.Fatalf("expected both waiting in dispatch queue, got %v %v", big.State(), small.State())
	}
	h.lp.Run(h.lp.Now().Add(61 * time.Millisecond))
	if big.State() != StateRunning || small.State() != StateRunning {
		t.Fatalf("expected both running after the first finished, got %v %v", big.State(), small.State())
	}
	if h.gpu.admitted[1] != big {
		t.Fatalf("small request skipped ahead of the head of the dispatch queue")
	}
}

func TestInputChargedOnlyOnAdmission(t *testing.T) {
	h := newHarness(NewEEVDF(), 100, 1)
	a := &tenant.Tenant{ID: "a", Tier: tenant.Free}
	h.submit(a, 100)      // running
	d := h.submit(a, 100) // dispatched
	q := h.submit(a, 100) // queued
	var reasons []DoneReason
	q.OnDone = func(r DoneReason) { reasons = append(reasons, r) }
	d.OnDone = func(r DoneReason) { reasons = append(reasons, r) }
	if d.State() != StateDispatched || q.State() != StateQueued {
		t.Fatalf("unexpected states %v %v", d.State(), q.State())
	}
	h.s.Cancel(q, d)
	if got := h.l.VRuntime(a, tenant.Embed); got != 100 {
		t.Fatalf("vruntime %.0f, want only the admitted request's 100", got)
	}
	if len(reasons) != 2 || reasons[0] != Cancelled || reasons[1] != Cancelled {
		t.Fatalf("cancel reasons %v", reasons)
	}
	if h.l.Inflight(a) != 1 {
		t.Fatalf("inflight %d, want 1", h.l.Inflight(a))
	}
}

func TestBurstGetsStaggeredDeadlines(t *testing.T) {
	e := NewEEVDF()
	l := ledger.New(time.Minute, time.Hour)
	env := &Env{Ledger: l, Active: map[*tenant.Tenant]int{}, Pending: map[*tenant.Tenant]int64{}}
	a := &tenant.Tenant{ID: "a", Tier: tenant.Free}
	r1 := &Request{Tenant: a, InputTokens: 100, seq: 1}
	r2 := &Request{Tenant: a, InputTokens: 100, seq: 2}
	e.Add(r1, env, false)
	e.Add(r2, env, false)
	if r1.Deadline() != 100 || r2.Deadline() != 200 {
		t.Fatalf("deadlines %v %v, want 100 200", r1.Deadline(), r2.Deadline())
	}
}

func TestRerankGroupSharesDeadlineAndRunsTogether(t *testing.T) {
	h := newHarness(NewEEVDF(), 100, 1)
	a := &tenant.Tenant{ID: "a", Tier: tenant.Free}
	b := &tenant.Tenant{ID: "b", Tier: tenant.Free}
	for i := 0; i < 6; i++ {
		h.submit(b, 100)
	}
	g := &Group{InputTokens: 500}
	var pairs []*Request
	for i := 0; i < 5; i++ {
		h.id++
		r := &Request{ID: h.id, Tenant: a, Model: tenant.Embed, InputTokens: 100, Group: g}
		r.OnUsage = func(in, out int) { h.l.Charge(a, tenant.Embed, in, out, h.lp.Now()) }
		h.s.Submit(r)
		pairs = append(pairs, r)
	}
	for _, r := range pairs[1:] {
		if r.State() == StateQueued && r.Deadline() != pairs[len(pairs)-1].Deadline() {
			t.Fatalf("pairs have different deadlines")
		}
	}
	h.lp.Run(h.lp.Now().Add(2 * time.Second))
	first := -1
	for i, r := range h.gpu.admitted {
		if r.Tenant == a {
			first = i
			break
		}
	}
	if first < 0 {
		t.Fatal("no pair ran")
	}
	for i := first; i < first+5; i++ {
		if h.gpu.admitted[i].Tenant != a {
			t.Fatalf("pairs were split up by other tenants' work: admission %d is %s", i, h.gpu.admitted[i].Tenant.ID)
		}
	}
}

func TestVRuntimeIsPerModel(t *testing.T) {
	h := newHarness(NewEEVDF(), 100, 1)
	a := &tenant.Tenant{ID: "a", Tier: tenant.Free}
	h.l.Charge(a, tenant.LLM, 1000, 0, h.lp.Now())
	h.submit(a, 100)
	h.lp.Run(h.lp.Now().Add(time.Second))
	if got := h.l.VRuntime(a, tenant.Embed); got != 100 {
		t.Fatalf("embed vruntime %.0f, want 100 (LLM usage must not count)", got)
	}
	if got := h.l.BillingUsage(a, h.lp.Now()); got != 1100 {
		t.Fatalf("billing %d, want 1100 summed over models", got)
	}
}

func TestRatioResetZeroesVRuntimeAndDeadlines(t *testing.T) {
	h := newHarness(NewEEVDF(), 100, 1)
	h.s.SetRatioReset(RatioReset{Factor: 1.5, MinTokens: 1000})
	prem := &tenant.Tenant{ID: "p", Tier: tenant.Premium}
	free := &tenant.Tenant{ID: "f", Tier: tenant.Free}
	h.l.Charge(free, tenant.Embed, 5000, 0, h.lp.Now())

	h.submit(prem, 100) // runs
	h.submit(free, 100) // dispatched
	queued := h.submit(free, 100)
	if queued.State() != StateQueued || queued.Deadline() < 5000 {
		t.Fatalf("setup: state %v deadline %.0f", queued.State(), queued.Deadline())
	}
	if h.s.Stats().Resets != 0 {
		t.Fatal("reset before the ratio was off")
	}

	// Premium has been served far more than 4.5× free per tenant.
	h.s.ratio.add(prem, 100000)
	h.submit(prem, 100)

	if got := h.s.Stats().Resets; got != 1 {
		t.Fatalf("resets = %d, want 1", got)
	}
	if v := h.l.VRuntime(free, tenant.Embed); v != 0 {
		t.Fatalf("free vruntime %.0f after reset, want 0", v)
	}
	if queued.Deadline() >= 5000 {
		t.Fatalf("queued deadline %.0f was not recomputed", queued.Deadline())
	}
	if h.l.BillingUsage(free, h.lp.Now()) < 5000 {
		t.Fatal("billing must not be reset")
	}
}

func TestRatioResetOffByDefault(t *testing.T) {
	h := newHarness(NewEEVDF(), 100, 1)
	prem := &tenant.Tenant{ID: "p", Tier: tenant.Premium}
	free := &tenant.Tenant{ID: "f", Tier: tenant.Free}
	h.submit(free, 100)
	h.s.ratio.add(prem, 1000000)
	h.submit(prem, 100)
	if h.s.Stats().Resets != 0 {
		t.Fatal("reset happened with Factor 0")
	}
}
