package bench

import (
	"bufio"
	"fmt"
	"io"
	"time"

	"github.com/DressingPluto84/Model-Scheduler/internal/tenant"
)

// WriteSamples writes every latency sample of the runs as CSV, one row per
// sample: policy, model, tier, metric, ms. Metrics are ttft and itl for the
// LLM and e2e for embed and rerank. scripts/dist.py plots them.
func WriteSamples(out io.Writer, reports ...*Report) error {
	w := bufio.NewWriter(out)
	fmt.Fprintln(w, "policy,model,tier,metric,ms")
	for _, r := range reports {
		for _, m := range tenant.Endpoints {
			for _, tier := range []tenant.Tier{tenant.Premium, tenant.Free} {
				s := r.Stats[statKey{m, tier}]
				for _, set := range []struct {
					name string
					ds   []time.Duration
				}{{"ttft", s.TTFT}, {"itl", s.ITL}, {"e2e", s.E2E}} {
					for _, d := range set.ds {
						fmt.Fprintf(w, "%s,%s,%s,%s,%.3f\n", r.Policy, m, tier, set.name, float64(d)/float64(time.Millisecond))
					}
				}
			}
		}
	}
	return w.Flush()
}
