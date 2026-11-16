// Command bench runs the same workload under each scheduling policy and prints a
// side-by-side comparison, on simulated GPUs (default) or real vLLM servers.
package main

import (
	"context"
	"flag"
	"log"
	"os"
	"strings"
	"time"

	"github.com/DressingPluto84/Model-Scheduler/internal/bench"
	"github.com/DressingPluto84/Model-Scheduler/internal/loop"
	"github.com/DressingPluto84/Model-Scheduler/internal/sched"
	"github.com/DressingPluto84/Model-Scheduler/internal/system"
	"github.com/DressingPluto84/Model-Scheduler/internal/tenant"
	"github.com/DressingPluto84/Model-Scheduler/internal/vllm"
)

func main() {
	var pw, fw float64
	w := bench.DefaultWorkload()
	flag.DurationVar(&w.Duration, "duration", w.Duration, "run length per policy (wall-clock time with -backend vllm)")
	flag.IntVar(&w.Premium, "premium", w.Premium, "number of premium tenants")
	flag.IntVar(&w.Free, "free", w.Free, "number of free tenants")
	flag.Float64Var(&w.HeavyIntensity, "heavy", w.HeavyIntensity, "heavy tenants' send rate as a multiple of their rate limit")
	flag.Float64Var(&w.LightIntensity, "light", w.LightIntensity, "light tenants' send rate as a multiple of their rate limit")
	flag.BoolVar(&w.Sleeper, "sleeper", w.Sleeper, "add a premium tenant that sleeps for the middle third of the run")
	flag.Float64Var(&w.CancelFraction, "cancel", w.CancelFraction, "fraction of LLM requests that disconnect early")
	flag.Int64Var(&w.Seed, "seed", w.Seed, "random seed")
	flag.IntVar(&w.RerankDocsMin, "rerank-docs-min", w.RerankDocsMin, "fewest documents per rerank request")
	flag.IntVar(&w.RerankDocsMax, "rerank-docs-max", w.RerankDocsMax, "most documents per rerank request")
	flag.DurationVar(&w.SLO.TTFT, "slo-ttft", w.SLO.TTFT, "goodput SLO: LLM time to first token")
	flag.DurationVar(&w.SLO.TPOT, "slo-tpot", w.SLO.TPOT, "goodput SLO: LLM average time per output token")
	flag.DurationVar(&w.SLO.Embed, "slo-embed", w.SLO.Embed, "goodput SLO: embed end-to-end latency")
	flag.DurationVar(&w.SLO.Rerank, "slo-rerank", w.SLO.Rerank, "goodput SLO: rerank end-to-end latency")
	// Billing is off by default: at rate-limit intensity the 24h caps run out
	// within minutes and most traffic becomes 429s, which hides the scheduler.
	billing := flag.Bool("billing", false, "enforce the 24h billing caps")
	rerankPairs := flag.Bool("rerank-pairs", false, "split each rerank request into one scheduler request per doc-query pair (default: one job)")
	rerankPerToken := flag.Duration("rerank-per-token", system.DefaultConfig().Rerank.PerToken, "simulated rerank GPU time per input token")
	depth := flag.Int("dispatch-depth", system.DefaultConfig().DispatchDepth, "dispatch queue depth")
	flag.Float64Var(&pw, "premium-weight", tenant.TierWeights[tenant.Premium], "EEVDF weight for premium tenants")
	flag.Float64Var(&fw, "free-weight", tenant.TierWeights[tenant.Free], "EEVDF weight for free tenants")
	policies := flag.String("policies", "fcfs,eevdf,tiered,lottery,drr", "comma-separated policies to compare: "+strings.Join(sched.Policies, ", "))
	tierShare := flag.String("tier-share", string(sched.TierSharePerTenant),
		"tiered policy split between tiers: per-tenant (weight ratio per tenant) or per-tier (weight ratio between whole tiers)")
	drrQuantum := flag.Int("drr-quantum", 4096, "drr quantum in tokens for embed and rerank (LLM uses 1 request)")
	resetFactor := flag.Float64("reset-factor", 0,
		"reset a model's vruntimes when premium:free service per tenant exceeds this × the weight ratio (0 = off)")
	resetMin := flag.Int64("reset-min-tokens", 20000, "tokens served on a model after a reset before the ratio is checked again")
	backend := flag.String("backend", "sim", "GPU backend: sim or vllm")
	drain := flag.Duration("drain", 15*time.Second, "pause between runs on vllm so the GPUs go idle")
	vcfg := vllm.DefaultConfig()
	vcfg.AddFlags(flag.CommandLine)
	flag.Parse()
	if pw <= 0 || fw <= 0 {
		log.Fatal("weights must be positive")
	}
	if w.RerankDocsMin < 1 || w.RerankDocsMax < w.RerankDocsMin {
		log.Fatal("need 1 <= -rerank-docs-min <= -rerank-docs-max")
	}
	tenant.TierWeights[tenant.Premium], tenant.TierWeights[tenant.Free] = pw, fw

	var b bench.Backend
	switch *backend {
	case "sim":
	case "vllm":
		b.Drain = *drain
		b.NewGPUs = func(lp *loop.Real) (system.GPUs, error) {
			ctx, cancel := context.WithTimeout(context.Background(), 10*time.Minute)
			defer cancel()
			return vllm.NewGPUs(ctx, lp, vcfg)
		}
		log.Printf("running %s for %s each against vLLM", *policies, w.Duration)
	default:
		log.Fatalf("unknown backend %q", *backend)
	}

	var reports []*bench.Report
	for _, policy := range strings.Split(*policies, ",") {
		policy = strings.TrimSpace(policy)
		cfg := system.DefaultConfig()
		cfg.Policy = policy
		cfg.BillingEnabled = *billing
		cfg.DispatchDepth = *depth
		cfg.RerankPairs = *rerankPairs
		cfg.Rerank.PerToken = *rerankPerToken
		cfg.RatioReset = sched.RatioReset{Factor: *resetFactor, MinTokens: *resetMin}
		cfg.PolicyOptions = sched.PolicyOptions{TierShare: sched.TierShare(*tierShare), DRRTokenQuantum: *drrQuantum}
		if b.NewGPUs != nil {
			log.Printf("starting %s run", policy)
		}
		rep, err := bench.Run(cfg, w, b)
		if err != nil {
			log.Fatal(err)
		}
		reports = append(reports, rep)
	}
	bench.Print(os.Stdout, reports...)
}
