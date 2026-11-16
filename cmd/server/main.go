// Command server runs the scheduler as a single process, on simulated GPUs
// (default) or real models served by vLLM.
package main

import (
	"context"
	"flag"
	"log"
	"net/http"
	"time"

	"github.com/DressingPluto84/Model-Scheduler/internal/httpapi"
	"github.com/DressingPluto84/Model-Scheduler/internal/loop"
	"github.com/DressingPluto84/Model-Scheduler/internal/sched"
	"github.com/DressingPluto84/Model-Scheduler/internal/system"
	"github.com/DressingPluto84/Model-Scheduler/internal/tenant"
	"github.com/DressingPluto84/Model-Scheduler/internal/vllm"
)

func main() {
	addr := flag.String("addr", ":8080", "listen address")
	tenantsPath := flag.String("tenants", "tenants.json", "JSON file listing tenants and their tiers")
	policy := flag.String("policy", "eevdf", "scheduling policy: fcfs, eevdf, tiered, lottery or drr")
	rerankPairs := flag.Bool("rerank-pairs", false, "split each rerank request into one scheduler request per doc-query pair (default: one job)")
	tierShare := flag.String("tier-share", string(sched.TierSharePerTenant), "tiered policy split: per-tenant or per-tier")
	backend := flag.String("backend", "sim", "GPU backend: sim or vllm")
	vcfg := vllm.DefaultConfig()
	vcfg.AddFlags(flag.CommandLine)
	flag.Parse()

	reg, err := tenant.LoadRegistry(*tenantsPath)
	if err != nil {
		log.Fatalf("load tenants: %v", err)
	}
	cfg := system.DefaultConfig()
	cfg.Policy = *policy
	cfg.RerankPairs = *rerankPairs
	cfg.PolicyOptions.TierShare = sched.TierShare(*tierShare)

	lp := loop.NewReal()
	lp.Start()
	defer lp.Stop()

	var gpus system.GPUs
	var tok vllm.Tokenizer = vllm.WordTokenizer{}
	switch *backend {
	case "sim":
	case "vllm":
		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Minute)
		log.Printf("waiting for vLLM servers at %s, %s, %s", vcfg.LLMURL, vcfg.EmbedURL, vcfg.RerankURL)
		gpus, err = vllm.NewGPUs(ctx, lp, vcfg)
		cancel()
		if err != nil {
			log.Fatalf("connect to vLLM: %v", err)
		}
		tok = vllm.NewServerTokenizer(vcfg)
	default:
		log.Fatalf("unknown backend %q", *backend)
	}

	var sys *system.System
	lp.Do(func() { sys, err = system.New(lp, reg, cfg, gpus) })
	if err != nil {
		log.Fatalf("build system: %v", err)
	}
	if *backend == "vllm" {
		var st struct{ kv int }
		lp.Do(func() { st.kv = sys.GPUStats(tenant.LLM).CapacityTokens })
		log.Printf("LLM KV budget: %d tokens", st.kv)
	}

	log.Printf("listening on %s (policy=%s, backend=%s, %d tenants)", *addr, cfg.Policy, *backend, len(reg.All()))
	log.Fatal(http.ListenAndServe(*addr, httpapi.New(sys, lp, tok).Handler()))
}
