package bench

import (
	"fmt"
	"io"
	"math"
	"sort"
	"strings"
	"text/tabwriter"
	"time"

	"github.com/DressingPluto84/Model-Scheduler/internal/sched"
	"github.com/DressingPluto84/Model-Scheduler/internal/system"
	"github.com/DressingPluto84/Model-Scheduler/internal/tenant"
)

type statKey struct {
	Model tenant.Endpoint
	Tier  tenant.Tier
}

// Stats are the metrics for one (model, tier) pair.
type Stats struct {
	Submitted int
	Rejected  map[string]int // by error code
	Completed int
	Cancelled int
	Failed    int // backend errors

	// SLOMet / SLOMissed count requests that met or missed their latency
	// target (see Workload.SLO). Client cancellations and requests still in
	// flight at the end that could still have made it are in neither.
	SLOMet, SLOMissed int

	// LLM
	TTFT []time.Duration
	ITL  []time.Duration
	// Embed / rerank
	E2E []time.Duration

	// Fairness: |lag| samples (weighted tokens) for backlogged tenants.
	LagSamples []float64
	// TierLagSamples are the tier's total signed lag (the sum over its
	// backlogged tenants), sampled once a second: positive when the tier as a
	// whole has received less than its weighted share, negative when more.
	TierLagSamples []float64

	// ServedTokens is the weighted service (input + 2 × output) this tier
	// received on this model during the run.
	ServedTokens int64
	Tenants      int
}

// Report holds the results of one run.
type Report struct {
	Policy   string
	Backend  string
	Billing  bool
	Workload Workload
	Stats    map[statKey]*Stats
	GPUBusy  map[tenant.Endpoint]float64 // fraction of wall time the GPU was busy
	Resets   map[tenant.Endpoint]int     // vruntime resets per model
	// TierRatio is premium tokens per premium tenant / free tokens per free
	// tenant, per model.
	TierRatio map[tenant.Endpoint]float64

	Reset   sched.RatioReset
	Sleeper *SleeperStats

	// closed is set when the run ends; later events are not recorded.
	closed bool
}

// SleeperStats describes the tenant that left and came back.
type SleeperStats struct {
	VRuntime float64
}

func newReport(policy string, w Workload) *Report {
	r := &Report{Policy: policy, Workload: w, Stats: make(map[statKey]*Stats),
		GPUBusy: make(map[tenant.Endpoint]float64), Resets: make(map[tenant.Endpoint]int),
		TierRatio: make(map[tenant.Endpoint]float64)}
	for _, m := range tenant.Endpoints {
		for _, tier := range []tenant.Tier{tenant.Premium, tenant.Free} {
			r.Stats[statKey{m, tier}] = &Stats{Rejected: make(map[string]int)}
		}
	}
	return r
}

func (r *Report) get(k statKey) *Stats { return r.Stats[k] }

func (r *Report) submitted(k statKey) { r.get(k).Submitted++ }

func (r *Report) rejected(k statKey, err *system.Error) {
	if err != nil && !r.closed {
		r.get(k).Rejected[err.Code]++
	}
}

func (r *Report) e2eDone(k statKey, reason sched.DoneReason, d time.Duration) {
	if r.closed {
		return
	}
	r.count(k, reason)
	if reason == sched.Completed {
		r.get(k).E2E = append(r.get(k).E2E, d)
	}
}

func (r *Report) llmDone(k statKey, reason sched.DoneReason) {
	if !r.closed {
		r.count(k, reason)
	}
}

func (r *Report) count(k statKey, reason sched.DoneReason) {
	s := r.get(k)
	switch reason {
	case sched.Completed:
		s.Completed++
	case sched.Cancelled:
		s.Cancelled++
	case sched.Failed:
		s.Failed++
	}
}

func (r *Report) slo(k statKey, met bool) {
	if r.closed {
		return
	}
	if met {
		r.get(k).SLOMet++
	} else {
		r.get(k).SLOMissed++
	}
}

func (r *Report) ttft(k statKey, d time.Duration) {
	if !r.closed {
		r.get(k).TTFT = append(r.get(k).TTFT, d)
	}
}

func (r *Report) itl(k statKey, d time.Duration) {
	if !r.closed {
		r.get(k).ITL = append(r.get(k).ITL, d)
	}
}

func (r *Report) finish(sys *system.System, all []*tenant.Tenant, dur time.Duration) {
	for _, m := range tenant.Endpoints {
		st := sys.Scheduler(m).Stats()
		for _, t := range all {
			s := r.get(statKey{m, t.Tier})
			s.ServedTokens += st.Served[t]
		}
		r.GPUBusy[m] = float64(sys.GPUStats(m).Busy) / float64(dur)
		r.Resets[m] = st.Resets
	}
	for _, t := range all {
		for _, m := range tenant.Endpoints {
			r.get(statKey{m, t.Tier}).Tenants++
		}
		if t.ID == "premium-sleeper" {
			r.Sleeper = &SleeperStats{VRuntime: sys.Ledger().TotalVRuntime(t)}
		}
	}

	for _, m := range tenant.Endpoints {
		p, f := r.get(statKey{m, tenant.Premium}), r.get(statKey{m, tenant.Free})
		if p.Tenants > 0 && f.Tenants > 0 && f.ServedTokens > 0 {
			r.TierRatio[m] = (float64(p.ServedTokens) / float64(p.Tenants)) / (float64(f.ServedTokens) / float64(f.Tenants))
		}
	}
}

// --- Lag -------------------------------------------------------------------------

// lagTracker measures each tenant's lag per model: the weighted service it
// was entitled to while backlogged (its w_i / Σ w share of everything served)
// minus the service it actually received. Lag resets when a tenant stops
// being backlogged.
type lagTracker struct {
	sys  *system.System
	all  []*tenant.Tenant
	rep  *Report
	prev map[tenant.Endpoint]map[*tenant.Tenant]int64
	lag  map[tenant.Endpoint]map[*tenant.Tenant]float64
}

func newLagTracker(sys *system.System, all []*tenant.Tenant, rep *Report) *lagTracker {
	lt := &lagTracker{sys: sys, all: all, rep: rep,
		prev: make(map[tenant.Endpoint]map[*tenant.Tenant]int64),
		lag:  make(map[tenant.Endpoint]map[*tenant.Tenant]float64)}
	for _, m := range tenant.Endpoints {
		lt.prev[m] = make(map[*tenant.Tenant]int64)
		lt.lag[m] = make(map[*tenant.Tenant]float64)
	}
	return lt
}

func (lt *lagTracker) sample() {
	for _, m := range tenant.Endpoints {
		st := lt.sys.Scheduler(m).Stats()
		var total, sumW float64
		delta := make(map[*tenant.Tenant]float64, len(lt.all))
		var backlogged []*tenant.Tenant
		for _, t := range lt.all {
			d := float64(st.Served[t] - lt.prev[m][t])
			lt.prev[m][t] = st.Served[t]
			delta[t] = d
			total += d
			if st.Active[t] > 0 || d > 0 {
				backlogged = append(backlogged, t)
				sumW += t.Weight()
			}
		}
		tierLag := map[tenant.Tier]float64{}
		tierIn := map[tenant.Tier]bool{}
		for _, t := range lt.all {
			in := st.Active[t] > 0 || delta[t] > 0
			if !in {
				lt.lag[m][t] = 0
				continue
			}
			lt.lag[m][t] += total*t.Weight()/sumW - delta[t]
			s := lt.rep.get(statKey{m, t.Tier})
			s.LagSamples = append(s.LagSamples, math.Abs(lt.lag[m][t]))
			tierLag[t.Tier] += lt.lag[m][t]
			tierIn[t.Tier] = true
		}
		for tier := range tierIn {
			s := lt.rep.get(statKey{m, tier})
			s.TierLagSamples = append(s.TierLagSamples, tierLag[tier])
		}
	}
}

// --- Printing ------------------------------------------------------------------

var tiers = []tenant.Tier{tenant.Premium, tenant.Free}

// Print writes a side-by-side comparison of reports (e.g. FCFS vs EEVDF) as a
// series of small tables.
func Print(out io.Writer, reports ...*Report) {
	if len(reports) == 0 {
		return
	}
	w := reports[0].Workload
	sleeper := ""
	if w.Sleeper {
		sleeper = " + 1 sleeper"
	}
	billing := "off"
	if reports[0].Billing {
		billing = "on"
	}
	reset := "off"
	if rr := reports[0].Reset; rr.Factor > 0 {
		reset = fmt.Sprintf("above %g:1 (after %d tokens)",
			rr.Factor*tenant.TierWeights[tenant.Premium]/tenant.TierWeights[tenant.Free], rr.MinTokens)
	}
	fmt.Fprintf(out, "%s on %s GPUs · %d premium + %d free tenants%s · weights %g:%g · seed %d · billing %s · vruntime reset %s\n",
		w.Duration, reports[0].Backend, w.Premium, w.Free, sleeper,
		tenant.TierWeights[tenant.Premium], tenant.TierWeights[tenant.Free], w.Seed, billing, reset)

	policies := make([]string, len(reports))
	for i, r := range reports {
		policies[i] = r.Policy
	}

	// perTier prints one row per tier for a single model.
	perTier := func(title string, m tenant.Endpoint, cell func(*Stats) string) {
		t := newTable(title, policies)
		for _, tier := range tiers {
			t.row(string(tier), reports, func(r *Report) string { return cell(r.get(statKey{m, tier})) })
		}
		t.print(out)
	}
	// perModelTier prints one row per (model, tier).
	perModelTier := func(title string, cell func(*Stats) string) {
		t := newTable(title, policies)
		for _, m := range tenant.Endpoints {
			for _, tier := range tiers {
				t.row(fmt.Sprintf("%s %s", m, tier), reports, func(r *Report) string { return cell(r.get(statKey{m, tier})) })
			}
		}
		t.print(out)
	}

	gpu := newTable("GPU busy", policies)
	for _, m := range tenant.Endpoints {
		gpu.row(string(m), reports, func(r *Report) string { return fmt.Sprintf("%.0f%%", 100*r.GPUBusy[m]) })
	}
	gpu.print(out)

	if reports[0].Reset.Factor > 0 {
		rt := newTable("Vruntime resets", policies)
		for _, m := range tenant.Endpoints {
			rt.row(string(m), reports, func(r *Report) string { return fmt.Sprintf("%d", r.Resets[m]) })
		}
		rt.print(out)
	}

	slo := w.SLO
	perModelTier(fmt.Sprintf("Goodput: requests within SLO (LLM TTFT ≤ %s and TPOT ≤ %s, embed ≤ %s, rerank ≤ %s)",
		dur(slo.TTFT), dur(slo.TPOT), dur(slo.Embed), dur(slo.Rerank)), func(s *Stats) string {
		n := s.SLOMet + s.SLOMissed
		if n == 0 {
			return "-"
		}
		return fmt.Sprintf("%.0f%% (%.2f/s)", 100*float64(s.SLOMet)/float64(n), float64(s.SLOMet)/w.Duration.Seconds())
	})

	latency := func(ds []time.Duration) string { return dur(pct(ds, 50)) + " / " + dur(pct(ds, 99)) }
	perTier("LLM time to first token (p50 / p99)", tenant.LLM, func(s *Stats) string { return latency(s.TTFT) })
	perTier("LLM inter-token latency (p50 / p99)", tenant.LLM, func(s *Stats) string { return latency(s.ITL) })
	perTier("Embed latency, end to end (p50 / p99)", tenant.Embed, func(s *Stats) string { return latency(s.E2E) })
	perTier("Rerank latency, end to end (p50 / p99)", tenant.Rerank, func(s *Stats) string { return latency(s.E2E) })

	target := tenant.TierWeights[tenant.Premium] / tenant.TierWeights[tenant.Free]
	rt := newTable(fmt.Sprintf("Premium:free tokens per tenant (weights say %g)", target), policies)
	for _, m := range tenant.Endpoints {
		rt.row(string(m), reports, func(r *Report) string {
			if r.TierRatio[m] == 0 {
				return "-"
			}
			return fmt.Sprintf("%.2f", r.TierRatio[m])
		})
	}
	rt.print(out)

	perModelTier("Fairness: mean |lag| per tenant in tokens (lower is fairer)", func(s *Stats) string {
		return fmt.Sprintf("%.0f", meanF(s.LagSamples))
	})
	perModelTier("Tier lag: mean signed lag of the whole tier in tokens (+ owed service / behind its share, − ahead)", func(s *Stats) string {
		if len(s.TierLagSamples) == 0 {
			return "-"
		}
		return fmt.Sprintf("%+.0f", meanF(s.TierLagSamples))
	})
	perModelTier("Tokens served per tenant", func(s *Stats) string {
		if s.Tenants == 0 {
			return "-"
		}
		return fmt.Sprintf("%d", s.ServedTokens/int64(s.Tenants))
	})
	perModelTier("Requests: completed / cancelled / rejected", func(s *Stats) string {
		rejected := 0
		for _, n := range s.Rejected {
			rejected += n
		}
		return fmt.Sprintf("%d / %d / %d", s.Completed, s.Cancelled, rejected)
	})

	failed := false
	for _, r := range reports {
		for _, st := range r.Stats {
			failed = failed || st.Failed > 0
		}
	}
	if failed {
		perModelTier("Backend errors (failed requests)", func(s *Stats) string { return fmt.Sprintf("%d", s.Failed) })
	}

	if reports[0].Sleeper != nil {
		t := newTable("Sleeper tenant", policies)
		t.row("final vruntime", reports, func(r *Report) string { return fmt.Sprintf("%.0f", r.Sleeper.VRuntime) })
		t.print(out)
	}
}

type table struct {
	title string
	rows  [][]string
}

func newTable(title string, columns []string) *table {
	return &table{title: title, rows: [][]string{append([]string{""}, columns...)}}
}

func (t *table) row(label string, reports []*Report, cell func(*Report) string) {
	r := []string{label}
	for _, rep := range reports {
		r = append(r, cell(rep))
	}
	t.rows = append(t.rows, r)
}

func (t *table) print(out io.Writer) {
	fmt.Fprintf(out, "\n%s\n", t.title)
	tw := tabwriter.NewWriter(out, 0, 0, 3, ' ', 0)
	for _, r := range t.rows {
		fmt.Fprintln(tw, "  "+strings.Join(r, "\t"))
	}
	tw.Flush()
}

func pct(ds []time.Duration, p float64) time.Duration {
	if len(ds) == 0 {
		return 0
	}
	s := append([]time.Duration(nil), ds...)
	sort.Slice(s, func(i, j int) bool { return s[i] < s[j] })
	idx := int(math.Ceil(p/100*float64(len(s)))) - 1
	return s[max(0, min(len(s)-1, idx))]
}

func meanF(xs []float64) float64 {
	if len(xs) == 0 {
		return 0
	}
	var sum float64
	for _, x := range xs {
		sum += x
	}
	return sum / float64(len(xs))
}

// dur formats a latency as milliseconds below one second, seconds above.
func dur(d time.Duration) string {
	if d < time.Second {
		return fmt.Sprintf("%dms", d.Milliseconds())
	}
	return fmt.Sprintf("%.1fs", d.Seconds())
}
