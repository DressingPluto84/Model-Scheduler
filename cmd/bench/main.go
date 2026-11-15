// Command bench runs the same workload under FCFS and EEVDF and prints a
// side-by-side comparison, on simulated GPUs (default) or real vLLM servers.
package main

import (
	"context"
	"flag"
	"log"
	"os"
	"time"

	"github.com/DressingPluto84/Model-Scheduler/internal/bench"
	"github.com/DressingPluto84/Model-Scheduler/internal/loop"
	"github.com/DressingPluto84/Model-Scheduler/internal/system"
	"github.com/DressingPluto84/Model-Scheduler/internal/vllm"
)

func main() {
	w := bench.DefaultWorkload()
	flag.DurationVar(&w.Duration, "duration", w.Duration, "run length per policy (wall-clock time with -backend vllm)")
	flag.IntVar(&w.Premium, "premium", w.Premium, "number of premium tenants")
	flag.IntVar(&w.Free, "free", w.Free, "number of free tenants")
	flag.Float64Var(&w.HeavyIntensity, "heavy", w.HeavyIntensity, "heavy tenants' send rate as a multiple of their rate limit")
	flag.Float64Var(&w.LightIntensity, "light", w.LightIntensity, "light tenants' send rate as a multiple of their rate limit")
	flag.BoolVar(&w.Sleeper, "sleeper", w.Sleeper, "add a premium tenant that sleeps for the middle third of the run")
	flag.Float64Var(&w.CancelFraction, "cancel", w.CancelFraction, "fraction of LLM requests that disconnect early")
	flag.Int64Var(&w.Seed, "seed", w.Seed, "random seed")
	// Billing is off by default: at rate-limit intensity the 24h caps run out
	// within minutes and most traffic becomes 429s, which hides the scheduler.
	billing := flag.Bool("billing", false, "enforce the 24h billing caps")
	depth := flag.Int("dispatch-depth", system.DefaultConfig().DispatchDepth, "dispatch queue depth")
	backend := flag.String("backend", "sim", "GPU backend: sim or vllm")
	drain := flag.Duration("drain", 15*time.Second, "pause between runs on vllm so the GPUs go idle")
	vcfg := vllm.DefaultConfig()
	vcfg.AddFlags(flag.CommandLine)
	flag.Parse()

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
		log.Printf("running FCFS then EEVDF for %s each against vLLM", w.Duration)
	default:
		log.Fatalf("unknown backend %q", *backend)
	}

	var reports []*bench.Report
	for _, policy := range []string{"fcfs", "eevdf"} {
		cfg := system.DefaultConfig()
		cfg.Policy = policy
		cfg.BillingEnabled = *billing
		cfg.DispatchDepth = *depth
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
