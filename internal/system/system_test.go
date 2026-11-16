package system

import (
	"net/http"
	"testing"
	"time"

	"github.com/DressingPluto84/Model-Scheduler/internal/loop"
	"github.com/DressingPluto84/Model-Scheduler/internal/sched"
	"github.com/DressingPluto84/Model-Scheduler/internal/tenant"
)

func newTestSystem(t *testing.T) (*System, *loop.Sim) {
	t.Helper()
	return newTestSystemWith(t, DefaultConfig())
}

func newTestSystemWith(t *testing.T, cfg Config) (*System, *loop.Sim) {
	t.Helper()
	lp := loop.NewSim(time.Unix(0, 0))
	reg, err := tenant.NewRegistry([]*tenant.Tenant{
		{ID: "prem", Tier: tenant.Premium},
		{ID: "free", Tier: tenant.Free},
	})
	if err != nil {
		t.Fatal(err)
	}
	sys, err := New(lp, reg, cfg, GPUs{})
	if err != nil {
		t.Fatal(err)
	}
	return sys, lp
}

func TestUnknownTenant(t *testing.T) {
	sys, _ := newTestSystem(t)
	_, err := sys.Embed("nobody", EmbedParams{InputTokens: 10}, nil)
	if err == nil || err.Status != http.StatusUnauthorized {
		t.Fatalf("got %v, want 401", err)
	}
}

func TestRateLimitPerTier(t *testing.T) {
	sys, lp := newTestSystem(t)
	for i := 0; i < 5; i++ {
		if _, err := sys.LLM("free", LLMParams{InputTokens: 10, OutputLen: 1}, LLMCallbacks{}); err != nil {
			t.Fatalf("request %d rejected: %v", i, err)
		}
	}
	_, err := sys.LLM("free", LLMParams{InputTokens: 10}, LLMCallbacks{})
	if err == nil || err.Status != http.StatusTooManyRequests || err.Code != CodeRateLimited {
		t.Fatalf("6th free llm request: got %v, want rate-limit 429", err)
	}
	// Premium gets 10/min.
	for i := 0; i < 10; i++ {
		if _, err := sys.LLM("prem", LLMParams{InputTokens: 10, OutputLen: 1}, LLMCallbacks{}); err != nil {
			t.Fatalf("premium request %d rejected: %v", i, err)
		}
	}
	// The window slides.
	lp.Run(lp.Now().Add(61 * time.Second))
	if _, err := sys.LLM("free", LLMParams{InputTokens: 10, OutputLen: 1}, LLMCallbacks{}); err != nil {
		t.Fatalf("after a minute: %v", err)
	}
}

func TestRequestSizeLimits(t *testing.T) {
	sys, _ := newTestSystem(t)
	if _, err := sys.Embed("prem", EmbedParams{InputTokens: tenant.MaxInputTokens + 1}, nil); err == nil || err.Code != CodeInputTooLarge {
		t.Fatalf("embed too large: got %v", err)
	}
	if _, err := sys.LLM("prem", LLMParams{InputTokens: 10, MaxTokens: tenant.MaxOutputTokens + 1}, LLMCallbacks{}); err == nil || err.Code != CodeMaxTokens {
		t.Fatalf("max_tokens too large: got %v", err)
	}
	if _, err := sys.Rerank("prem", RerankParams{QueryTokens: 100, DocTokens: []int{10, tenant.MaxInputTokens}}, nil); err == nil || err.Code != CodeInputTooLarge {
		t.Fatalf("rerank pair too large: got %v", err)
	}
}

func TestBillingCapRollingWindow(t *testing.T) {
	sys, lp := newTestSystem(t)
	free := sys.Tenants().Get("free")
	sys.Ledger().Charge(free, tenant.LLM, int(tenant.BillingCap(tenant.Free)), 0, lp.Now())

	_, err := sys.Embed("free", EmbedParams{InputTokens: 10}, nil)
	if err == nil || err.Status != http.StatusTooManyRequests || err.Code != CodeBillingLimited {
		t.Fatalf("got %v, want billing 429", err)
	}
	if err.Code == CodeRateLimited || err.Message == "" {
		t.Fatalf("billing 429 should be distinguishable from rate-limit 429")
	}
	if err.RetryAfter <= 23*time.Hour {
		t.Fatalf("retry after %v, want about 24h", err.RetryAfter)
	}
	// The cap is on the sum over all models: LLM usage blocks embed too.
	// Premium is unaffected.
	if _, err := sys.Embed("prem", EmbedParams{InputTokens: 10}, nil); err != nil {
		t.Fatalf("premium: %v", err)
	}
	lp.Run(lp.Now().Add(24*time.Hour + time.Second))
	if _, err := sys.Embed("free", EmbedParams{InputTokens: 10}, nil); err != nil {
		t.Fatalf("after the window rolled: %v", err)
	}
}

func TestLLMStreamAndBatchedAccounting(t *testing.T) {
	sys, lp := newTestSystem(t)
	prem := sys.Tenants().Get("prem")
	var tokens int
	var reason sched.DoneReason = -1
	var at33 float64
	_, err := sys.LLM("prem", LLMParams{InputTokens: 50, MaxTokens: 100, OutputLen: 70}, LLMCallbacks{
		OnToken: func(i int, _ string, _ time.Time) {
			tokens++
			if i == 32 {
				at33 = sys.Ledger().VRuntime(prem, tenant.LLM)
			}
		},
		OnDone: func(d LLMDone) { reason = d.Reason },
	})
	if err != nil {
		t.Fatal(err)
	}
	lp.Run(lp.Now().Add(time.Minute))
	if tokens != 70 || reason != sched.Completed {
		t.Fatalf("got %d tokens, reason %v", tokens, reason)
	}
	// After the 33rd token only one batch of k=32 has been written.
	if at33 != 50+2*32 {
		t.Fatalf("vruntime at token 33 = %.0f, want %d", at33, 50+2*32)
	}
	if got := sys.Ledger().VRuntime(prem, tenant.LLM); got != 50+2*70 {
		t.Fatalf("final vruntime %.0f, want %d", got, 50+2*70)
	}
}

func TestLLMCancelChargesTokensUsed(t *testing.T) {
	sys, lp := newTestSystem(t)
	prem := sys.Tenants().Get("prem")
	var h *Handle
	var reason sched.DoneReason = -1
	var generated int
	h, err := sys.LLM("prem", LLMParams{InputTokens: 50, MaxTokens: 500, OutputLen: 400}, LLMCallbacks{
		OnToken: func(i int, _ string, _ time.Time) {
			if i == 39 {
				lp.Post(h.Cancel)
			}
		},
		OnDone: func(d LLMDone) { reason, generated = d.Reason, d.Generated },
	})
	if err != nil {
		t.Fatal(err)
	}
	lp.Run(lp.Now().Add(time.Minute))
	if reason != sched.Cancelled || generated != 40 {
		t.Fatalf("reason %v generated %d, want cancelled after 40", reason, generated)
	}
	if got := sys.Ledger().VRuntime(prem, tenant.LLM); got != 50+2*40 {
		t.Fatalf("vruntime %.0f, want %d", got, 50+2*40)
	}
	if st := sys.GPUStats(tenant.LLM); st.UsedTokens != 0 {
		t.Fatalf("KV cache still holds %d tokens after cancel", st.UsedTokens)
	}
}

func TestRerankIsOneJob(t *testing.T) {
	sys, lp := newTestSystem(t)
	prem := sys.Tenants().Get("prem")
	calls := 0
	var res RerankResult
	_, err := sys.Rerank("prem", RerankParams{QueryTokens: 10, DocTokens: []int{90, 190, 290}},
		func(r RerankResult) { calls++; res = r })
	if err != nil {
		t.Fatal(err)
	}
	if st := sys.Scheduler(tenant.Rerank).Stats(); st.Queued+st.Dispatched > 1 {
		t.Fatalf("rerank became %d scheduler requests, want 1", st.Queued+st.Dispatched)
	}
	lp.Run(lp.Now().Add(time.Minute))
	if calls != 1 || res.Reason != sched.Completed || res.Pairs != 3 {
		t.Fatalf("done %d times: %+v", calls, res)
	}
	// Input is the sum over pairs: (10+90) + (10+190) + (10+290).
	if got := sys.Ledger().VRuntime(prem, tenant.Rerank); got != 600 {
		t.Fatalf("vruntime %.0f, want 600", got)
	}
}

func TestOversizedRerankRunsAlone(t *testing.T) {
	sys, lp := newTestSystem(t)
	docs := make([]int, 12) // 12 × (10 + 990) = 12,000 tokens, over the 4,096 batch budget
	for i := range docs {
		docs[i] = 990
	}
	done := false
	if _, err := sys.Rerank("prem", RerankParams{QueryTokens: 10, DocTokens: docs},
		func(r RerankResult) { done = r.Reason == sched.Completed }); err != nil {
		t.Fatal(err)
	}
	lp.Run(lp.Now().Add(time.Minute))
	if !done {
		t.Fatal("a rerank bigger than the batch budget never ran")
	}
}

func TestRerankCancelDropsUncomputedPairs(t *testing.T) {
	cfg := DefaultConfig()
	cfg.RerankPairs = true
	sys, lp := newTestSystemWith(t, cfg)
	prem := sys.Tenants().Get("prem")
	// Each pair is 1000 tokens; the rerank GPU fits 4 per batch.
	docs := make([]int, 12)
	for i := range docs {
		docs[i] = 990
	}
	calls := 0
	var res RerankResult
	h, err := sys.Rerank("prem", RerankParams{QueryTokens: 10, DocTokens: docs}, func(r RerankResult) { calls++; res = r })
	if err != nil {
		t.Fatal(err)
	}
	// Let the first batch finish, then disconnect.
	batch := DefaultConfig().Rerank
	lp.Run(lp.Now().Add(batch.Base + 4000*batch.PerToken + time.Millisecond))
	h.Cancel()
	lp.Run(lp.Now().Add(time.Minute))
	if calls != 1 || res.Reason != sched.Cancelled {
		t.Fatalf("done called %d times with %v", calls, res.Reason)
	}
	// First batch (4 pairs) finished and the second batch (4 pairs) was
	// already admitted; the remaining 4 pairs were never charged.
	if got := sys.Ledger().VRuntime(prem, tenant.Rerank); got != 8000 {
		t.Fatalf("vruntime %.0f, want 8000", got)
	}
	if sys.Ledger().Inflight(prem) != 0 {
		t.Fatalf("inflight %d after cancel", sys.Ledger().Inflight(prem))
	}
}
