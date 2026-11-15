package vllm

import (
	"context"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/DressingPluto84/Model-Scheduler/internal/loop"
	"github.com/DressingPluto84/Model-Scheduler/internal/sched"
	"github.com/DressingPluto84/Model-Scheduler/internal/system"
	"github.com/DressingPluto84/Model-Scheduler/internal/tenant"
	"github.com/DressingPluto84/Model-Scheduler/internal/vllm/fakevllm"
)

type env struct {
	fake *fakevllm.Server
	lp   *loop.Real
	sys  *system.System
	prem *tenant.Tenant
	tok  *ServerTokenizer
}

func newEnv(t *testing.T, fc fakevllm.Config, tweak func(*Config)) *env {
	t.Helper()
	fake := fakevllm.New(fc)
	srv := httptest.NewServer(fake.Handler())
	t.Cleanup(srv.Close)

	cfg := DefaultConfig()
	cfg.LLMURL, cfg.EmbedURL, cfg.RerankURL = srv.URL, srv.URL, srv.URL
	if tweak != nil {
		tweak(&cfg)
	}
	lp := loop.NewReal()
	lp.Start()
	t.Cleanup(lp.Stop)
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	gpus, err := NewGPUs(ctx, lp, cfg)
	if err != nil {
		t.Fatal(err)
	}
	prem := &tenant.Tenant{ID: "prem", Tier: tenant.Premium}
	reg, _ := tenant.NewRegistry([]*tenant.Tenant{prem})
	var sys *system.System
	lp.Do(func() { sys, err = system.New(lp, reg, system.DefaultConfig(), gpus) })
	if err != nil {
		t.Fatal(err)
	}
	return &env{fake: fake, lp: lp, sys: sys, prem: prem, tok: NewServerTokenizer(cfg)}
}

func (e *env) vruntime() float64 {
	var v float64
	e.lp.Do(func() { v = e.sys.Ledger().VRuntime(e.prem, tenant.LLM) })
	return v
}

func wait[T any](t *testing.T, ch <-chan T) T {
	t.Helper()
	select {
	case v := <-ch:
		return v
	case <-time.After(10 * time.Second):
		t.Fatal("timed out")
	}
	var zero T
	return zero
}

func TestKVBudgetFromMetrics(t *testing.T) {
	e := newEnv(t, fakevllm.DefaultConfig(), nil)
	var kv int
	e.lp.Do(func() { kv = e.sys.GPUStats(tenant.LLM).CapacityTokens })
	if want := 31129; kv != want { // 2048 blocks × 16 × (1 - 5% safety margin)
		t.Fatalf("KV budget %d, want %d", kv, want)
	}
	n, err := e.tok.Count(context.Background(), tenant.LLM, "one two three")
	if err != nil || n != 3 {
		t.Fatalf("tokenize = %d, %v", n, err)
	}
}

func TestLLMStreamsAndCharges(t *testing.T) {
	e := newEnv(t, fakevllm.DefaultConfig(), nil)
	done := make(chan system.LLMDone, 1)
	var texts []string
	e.lp.Do(func() {
		_, err := e.sys.LLM("prem", system.LLMParams{Prompt: "a b c d e", InputTokens: 5, MaxTokens: 100, OutputLen: 40},
			system.LLMCallbacks{
				OnToken: func(i int, text string, _ time.Time) { texts = append(texts, text) },
				OnDone:  func(d system.LLMDone) { done <- d },
			})
		if err != nil {
			t.Error(err)
		}
	})
	d := wait(t, done)
	if d.Reason != sched.Completed || d.Generated != 40 || len(texts) != 40 || texts[0] != " tok0" {
		t.Fatalf("done %+v with %d tokens (first %q)", d, len(texts), texts[0])
	}
	if v := e.vruntime(); v != 5+2*40 {
		t.Fatalf("vruntime %.0f, want %d", v, 5+2*40)
	}
	var st int
	e.lp.Do(func() { st = e.sys.GPUStats(tenant.LLM).UsedTokens })
	if st != 0 {
		t.Fatalf("KV still reserved: %d", st)
	}
}

func TestLLMCancelAbortsInVLLM(t *testing.T) {
	fc := fakevllm.DefaultConfig()
	fc.DecodePerToken = 5 * time.Millisecond
	e := newEnv(t, fc, nil)
	done := make(chan system.LLMDone, 1)
	e.lp.Do(func() {
		var h *system.Handle
		h, _ = e.sys.LLM("prem", system.LLMParams{Prompt: "hi", InputTokens: 1, MaxTokens: 500, OutputLen: 500},
			system.LLMCallbacks{
				OnToken: func(i int, _ string, _ time.Time) {
					if i == 9 {
						h.Cancel()
					}
				},
				OnDone: func(d system.LLMDone) { done <- d },
			})
	})
	d := wait(t, done)
	if d.Reason != sched.Cancelled || d.Generated != 10 {
		t.Fatalf("done %+v, want cancelled after 10 tokens", d)
	}
	if v := e.vruntime(); v != 1+2*10 {
		t.Fatalf("vruntime %.0f, want %d", v, 1+2*10)
	}
	deadline := time.Now().Add(2 * time.Second)
	for e.fake.Aborted.Load() == 0 && time.Now().Before(deadline) {
		time.Sleep(10 * time.Millisecond)
	}
	if e.fake.Aborted.Load() != 1 {
		t.Fatalf("vLLM did not see the abort")
	}
}

func TestKVBudgetLimitsConcurrency(t *testing.T) {
	fc := fakevllm.DefaultConfig()
	fc.DecodePerToken = time.Millisecond
	// Cap the budget so exactly two reservations of 10 + 4096 tokens fit.
	e := newEnv(t, fc, func(c *Config) { c.KVCacheTokens = 8192 + 2*(10+100) })
	done := make(chan system.LLMDone, 8)
	e.lp.Do(func() {
		for i := 0; i < 8; i++ {
			e.sys.LLM("prem", system.LLMParams{Prompt: "w w w w w w w w w w", InputTokens: 10, MaxTokens: 4096, OutputLen: 20},
				system.LLMCallbacks{OnDone: func(d system.LLMDone) { done <- d }})
		}
	})
	for i := 0; i < 8; i++ {
		if d := wait(t, done); d.Reason != sched.Completed {
			t.Fatalf("request %d: %+v", i, d)
		}
	}
	// Each request reserves 10 + 4096 tokens, so at most two fit at once.
	if m := e.fake.MaxActive.Load(); m > 2 {
		t.Fatalf("%d streams ran at once; the KV budget allows 2", m)
	}
}

func TestEmbedAndRerank(t *testing.T) {
	e := newEnv(t, fakevllm.DefaultConfig(), nil)
	embed := make(chan system.EmbedResult, 1)
	rerank := make(chan system.RerankResult, 1)
	e.lp.Do(func() {
		e.sys.Embed("prem", system.EmbedParams{Text: "hello world", InputTokens: 2},
			func(r system.EmbedResult) { embed <- r })
		e.sys.Rerank("prem", system.RerankParams{Query: "gpu fair", QueryTokens: 2,
			Docs: []string{"cats", "gpu fair share", "gpu"}, DocTokens: []int{1, 3, 1}},
			func(r system.RerankResult) { rerank <- r })
	})
	if r := wait(t, embed); r.Reason != sched.Completed || len(r.Embedding) != 8 {
		t.Fatalf("embed %+v", r)
	}
	r := wait(t, rerank)
	if r.Reason != sched.Completed || len(r.Scores) != 3 || r.Scores[1] != 1 || r.Scores[0] != 0 || r.Scores[2] != 0.5 {
		t.Fatalf("rerank %+v", r)
	}
}

func TestBackendFailure(t *testing.T) {
	fc := fakevllm.DefaultConfig()
	fc.FailPrompt = "boom"
	e := newEnv(t, fc, nil)
	done := make(chan system.LLMDone, 1)
	embed := make(chan system.EmbedResult, 1)
	e.lp.Do(func() {
		e.sys.LLM("prem", system.LLMParams{Prompt: "boom", InputTokens: 1, MaxTokens: 10},
			system.LLMCallbacks{OnDone: func(d system.LLMDone) { done <- d }})
		e.sys.Embed("prem", system.EmbedParams{Text: "boom", InputTokens: 1}, func(r system.EmbedResult) { embed <- r })
	})
	if d := wait(t, done); d.Reason != sched.Failed || d.Err == nil {
		t.Fatalf("llm %+v, want failed", d)
	}
	if r := wait(t, embed); r.Reason != sched.Failed || r.Err == nil {
		t.Fatalf("embed %+v, want failed", r)
	}
}
