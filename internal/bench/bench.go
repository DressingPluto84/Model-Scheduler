// Package bench runs a simulated workload against the system on the
// discrete-event loop and collects latency and fairness metrics.
package bench

import (
	"fmt"
	"math"
	"math/rand"
	"strings"
	"time"

	"github.com/DressingPluto84/Model-Scheduler/internal/sched"

	"github.com/DressingPluto84/Model-Scheduler/internal/loop"
	"github.com/DressingPluto84/Model-Scheduler/internal/system"
	"github.com/DressingPluto84/Model-Scheduler/internal/tenant"
)

// Workload describes the simulated traffic.
type Workload struct {
	Duration time.Duration
	Premium  int // number of premium tenants
	Free     int // number of free tenants
	// HeavyFraction of tenants send at HeavyIntensity × their rate limit; the
	// rest send at LightIntensity × their rate limit.
	HeavyFraction  float64
	HeavyIntensity float64
	LightIntensity float64
	// Sleeper adds one premium tenant that goes quiet for the middle third of
	// the run and then comes back.
	Sleeper bool
	// CancelFraction of LLM requests disconnect after a few tokens.
	CancelFraction float64
	// Skew, when above 0, replaces the heavy/light split with a Zipf spread
	// of traffic across each tier's tenants, as in production where a few
	// large customers send most requests: the tenant ranked i sends in
	// proportion to 1 / i^Skew, scaled so the tier's average intensity stays
	// the same. Each tenant's quota grows with its traffic, so large tenants
	// aren't simply rate limited away.
	Skew float64
	// Burstiness is the coefficient of variation of the gaps between a
	// tenant's requests. 1 is Poisson (evenly random); above 1 the same
	// average rate arrives in bursts separated by quiet spells, as real
	// traffic does. Gaps are gamma distributed with shape 1 / Burstiness².
	Burstiness float64

	LLMInputMedian    float64
	LLMMaxTokens      int
	LLMOutputMedian   float64
	LLMOutputSigma    float64
	EmbedInputMedian  float64
	RerankQueryMedian float64
	RerankDocMedian   float64
	RerankDocsMin     int
	RerankDocsMax     int

	Seed int64

	// SLO is the latency target a request must meet to count toward goodput.
	SLO SLO
}

// SLO is a per-request latency target. An LLM request meets it when its time
// to first token is within TTFT and its average time per output token after
// the first is within TPOT; embed and rerank requests when they finish within
// Embed / Rerank of arriving.
type SLO struct {
	TTFT, TPOT    time.Duration
	Embed, Rerank time.Duration
}

// DefaultWorkload offers more load than the default simulated GPUs can serve,
// so every GPU stays saturated and queues build up (the regime where the
// scheduling policy matters).
func DefaultWorkload() Workload {
	return Workload{
		Duration:          10 * time.Minute,
		Premium:           10,
		Free:              30,
		HeavyFraction:     0.5,
		HeavyIntensity:    1.2,
		LightIntensity:    0.3,
		Sleeper:           true,
		CancelFraction:    0.05,
		Burstiness:        1,
		LLMInputMedian:    300,
		LLMMaxTokens:      512,
		LLMOutputMedian:   200,
		LLMOutputSigma:    0.8,
		EmbedInputMedian:  200,
		RerankQueryMedian: 20,
		RerankDocMedian:   150,
		RerankDocsMin:     4,
		RerankDocsMax:     16,
		Seed:              42,
		SLO:               SLO{TTFT: 2 * time.Second, TPOT: 100 * time.Millisecond, Embed: 3 * time.Second, Rerank: 2 * time.Second},
	}
}

type simTenant struct {
	t         *tenant.Tenant
	intensity float64
	sleeper   bool
}

// Backend selects where the benchmark's requests run.
type Backend struct {
	// NewGPUs builds real GPUs (e.g. vLLM) on a wall-clock loop. When nil,
	// the run uses simulated GPUs on the simulated clock.
	NewGPUs func(lp *loop.Real) (system.GPUs, error)
	// Drain is how long a real run waits after cancelling outstanding work,
	// so the next run starts on idle GPUs.
	Drain time.Duration
}

// Run executes the workload under cfg and returns the collected metrics.
//
// With simulated GPUs the run takes about a second. With a real backend it
// takes w.Duration of wall-clock time; at the end, outstanding requests are
// cancelled (and not counted) so the GPUs are idle for the next run.
func Run(cfg system.Config, w Workload, b Backend) (*Report, error) {
	tenants, all := makeTenants(w)
	reg, err := tenant.NewRegistry(all)
	if err != nil {
		return nil, err
	}
	rep := newReport(cfg.Policy, w)
	rep.Billing = cfg.BillingEnabled
	rep.Reset = cfg.RatioReset
	rep.Backend = "simulated"
	if b.NewGPUs != nil {
		rep.Backend = "vLLM"
	}

	if b.NewGPUs == nil {
		lp := loop.NewSim(time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC))
		run, err := start(lp, reg, cfg, system.GPUs{}, w, tenants, all, rep, false)
		if err != nil {
			return nil, err
		}
		lp.Run(run.end)
		run.stop()
		return rep, nil
	}

	lp := loop.NewReal()
	lp.Start()
	defer lp.Stop()
	gpus, err := b.NewGPUs(lp)
	if err != nil {
		return nil, err
	}
	var run *runState
	lp.Do(func() { run, err = start(lp, reg, cfg, gpus, w, tenants, all, rep, true) })
	if err != nil {
		return nil, err
	}
	time.Sleep(w.Duration)
	lp.Do(run.stop)
	time.Sleep(b.Drain)
	return rep, nil
}

func makeTenants(w Workload) ([]*simTenant, []*tenant.Tenant) {
	var tenants []*simTenant
	var all []*tenant.Tenant
	add := func(id string, tier tenant.Tier, intensity float64, sleeper bool) {
		t := &tenant.Tenant{ID: id, Tier: tier}
		tenants = append(tenants, &simTenant{t: t, intensity: intensity, sleeper: sleeper})
		all = append(all, t)
	}
	for i, x := range intensities(w, w.Premium) {
		add(fmt.Sprintf("premium-%02d", i), tenant.Premium, x, false)
	}
	for i, x := range intensities(w, w.Free) {
		add(fmt.Sprintf("free-%02d", i), tenant.Free, x, false)
	}
	for _, st := range tenants {
		if w.Skew > 0 && st.intensity > 1 {
			st.t.QuotaScale = st.intensity
		}
	}
	if w.Sleeper {
		add("premium-sleeper", tenant.Premium, w.HeavyIntensity, true)
	}
	return tenants, all
}

// runState is one benchmark run. It is only touched on the event loop.
type runState struct {
	sys     *system.System
	rep     *Report
	all     []*tenant.Tenant
	dur     time.Duration
	end     time.Time
	handles map[*system.Handle]*inflight
}

// inflight is a request that hasn't finished yet, kept so the run's end can
// tell whether it had already missed its SLO.
type inflight struct {
	key        statKey
	arrival    time.Time
	firstToken time.Time // LLM only; zero until the first token arrives
}

func start(lp loop.Loop, reg *tenant.Registry, cfg system.Config, gpus system.GPUs, w Workload,
	tenants []*simTenant, all []*tenant.Tenant, rep *Report, withText bool) (*runState, error) {
	sys, err := system.New(lp, reg, cfg, gpus)
	if err != nil {
		return nil, err
	}
	begin := lp.Now()
	run := &runState{sys: sys, rep: rep, all: all, dur: w.Duration, end: begin.Add(w.Duration),
		handles: make(map[*system.Handle]*inflight)}

	// One independent random stream per (tenant, endpoint), and every request
	// parameter (including the LLM output length) is drawn at arrival, so
	// FCFS and EEVDF see exactly the same requests.
	for ti, st := range tenants {
		for ei, ep := range tenant.Endpoints {
			seed := w.Seed + int64(ti)*1000 + int64(ei)
			g := &generator{run: run, sys: sys, lp: lp, rng: rand.New(rand.NewSource(seed)), st: st, ep: ep,
				rate: st.intensity * float64(tenant.RateLimitPerMinute(st.t.Tier, ep)) / 60, // per second
				w:    w, rep: rep, start: begin, end: run.end}
			if withText {
				g.textRng = rand.New(rand.NewSource(-seed))
			}
			g.scheduleNext()
		}
	}

	lag := newLagTracker(sys, all, rep)
	var tick func()
	tick = func() {
		if rep.closed {
			return
		}
		lag.sample()
		lp.After(time.Second, tick)
	}
	lp.After(time.Second, tick)
	return run, nil
}

// stop records the final metrics and cancels everything still in flight.
//
// A request still in flight counts as an SLO miss if it has already missed
// (waited longer than its target for its first token or its result).
// Otherwise it could still have made it, so it isn't counted either way.
func (run *runState) stop() {
	slo := run.rep.Workload.SLO
	now := run.sys.Loop().Now()
	for _, f := range run.handles {
		target := slo.Embed
		switch f.key.Model {
		case tenant.Rerank:
			target = slo.Rerank
		case tenant.LLM:
			target = slo.TTFT
			if !f.firstToken.IsZero() {
				if f.firstToken.Sub(f.arrival) > target {
					run.rep.slo(f.key, false)
				}
				continue
			}
		}
		if now.Sub(f.arrival) > target {
			run.rep.slo(f.key, false)
		}
	}
	run.rep.finish(run.sys, run.all, run.dur)
	run.rep.closed = true
	for h := range run.handles {
		h.Cancel()
	}
}

// intensities returns the send rate of each of a tier's n tenants as a
// multiple of the tier's rate limit (see Workload.Skew).
func intensities(w Workload, n int) []float64 {
	out := make([]float64, n)
	if w.Skew <= 0 {
		for i := range out {
			out[i] = intensity(w, i, n)
		}
		return out
	}
	mean := w.HeavyFraction*w.HeavyIntensity + (1-w.HeavyFraction)*w.LightIntensity
	var sum float64
	for i := range out {
		out[i] = 1 / math.Pow(float64(i+1), w.Skew)
		sum += out[i]
	}
	for i := range out {
		out[i] *= mean * float64(n) / sum
	}
	return out
}

func intensity(w Workload, i, n int) float64 {
	if float64(i) < w.HeavyFraction*float64(n) {
		return w.HeavyIntensity
	}
	return w.LightIntensity
}

type generator struct {
	run        *runState
	sys        *system.System
	lp         loop.Loop
	rng        *rand.Rand
	textRng    *rand.Rand // nil on simulated GPUs, which don't need text
	st         *simTenant
	ep         tenant.Endpoint
	rate       float64
	w          Workload
	rep        *Report
	start, end time.Time
}

func (g *generator) scheduleNext() {
	if g.rate <= 0 {
		return
	}
	mean := 1 / g.rate // seconds
	var gap time.Duration
	if b := g.w.Burstiness; b <= 0 || b == 1 {
		gap = time.Duration(g.rng.ExpFloat64() * mean * float64(time.Second))
	} else {
		k := 1 / (b * b) // gamma shape; mean k × scale = mean, CV 1/√k = b
		gap = time.Duration(gammaRand(g.rng, k) / k * mean * float64(time.Second))
	}
	if !g.lp.Now().Add(gap).Before(g.end) {
		return
	}
	g.lp.After(gap, func() {
		if g.rep.closed {
			return
		}
		g.fire()
		g.scheduleNext()
	})
}

// gammaRand draws from a gamma distribution with shape k and scale 1
// (Marsaglia and Tsang; for k < 1, boosted from k + 1).
func gammaRand(rng *rand.Rand, k float64) float64 {
	if k < 1 {
		return gammaRand(rng, k+1) * math.Pow(rng.Float64(), 1/k)
	}
	d := k - 1.0/3
	c := 1 / math.Sqrt(9*d)
	for {
		x := rng.NormFloat64()
		v := 1 + c*x
		if v <= 0 {
			continue
		}
		v = v * v * v
		u := rng.Float64()
		if math.Log(u) < 0.5*x*x+d-d*v+d*math.Log(v) {
			return d * v
		}
	}
}

func (g *generator) asleep() bool {
	if !g.st.sleeper {
		return false
	}
	el := g.lp.Now().Sub(g.start)
	return el >= g.w.Duration/3 && el < 2*g.w.Duration/3
}

func lognormal(rng *rand.Rand, median, sigma float64, lo, hi int) int {
	v := int(math.Round(median * math.Exp(sigma*rng.NormFloat64())))
	return max(lo, min(hi, v))
}

// words are common English words that most tokenizers encode as a single
// token each, so a text of n words is about n tokens.
var words = strings.Fields(`the of and to in is was for on that with as by at from this are be or an
	have not but had his they you were her all she there would their we him been has when who will
	more no if out so said what up its about into than them can only other new some could time these
	two may then do first any like now my such make over our even most made after also did many off
	before must well back through years much where your way down should because each just those people
	how too little state good very world still own see men work long here get both between life being
	under never day same another know while last might great old year come since against go came right
	used take three small place house water city number part school point light country end open`)

func (g *generator) text(n int) string {
	if g.textRng == nil {
		return ""
	}
	var b strings.Builder
	for i := 0; i < n; i++ {
		if i > 0 {
			b.WriteByte(' ')
		}
		b.WriteString(words[g.textRng.Intn(len(words))])
	}
	return b.String()
}

func (g *generator) track(h *system.Handle, key statKey) *inflight {
	f := &inflight{key: key, arrival: g.lp.Now()}
	if h != nil {
		g.run.handles[h] = f
	}
	return f
}

// e2eSLO records whether a finished embed or rerank request met its target.
// Requests the client cancelled don't count.
func (g *generator) e2eSLO(key statKey, reason sched.DoneReason, d, target time.Duration) {
	if reason != sched.Cancelled {
		g.rep.slo(key, reason == sched.Completed && d <= target)
	}
}

func (g *generator) fire() {
	// Draw every parameter up front, even if the request is skipped, so each
	// stream stays aligned across runs.
	t := g.st.t
	key := statKey{Model: g.ep, Tier: t.Tier}
	switch g.ep {
	case tenant.Embed:
		in := lognormal(g.rng, g.w.EmbedInputMedian, 0.7, 1, 6000)
		if g.asleep() {
			return
		}
		g.rep.submitted(key)
		var h *system.Handle
		h, err := g.sys.Embed(t.ID, system.EmbedParams{Text: g.text(in), InputTokens: in}, func(res system.EmbedResult) {
			delete(g.run.handles, h)
			g.rep.e2eDone(key, res.Reason, res.Done.Sub(res.Arrival))
			g.e2eSLO(key, res.Reason, res.Done.Sub(res.Arrival), g.w.SLO.Embed)
		})
		g.track(h, key)
		g.rep.rejected(key, err)

	case tenant.Rerank:
		q := lognormal(g.rng, g.w.RerankQueryMedian, 0.5, 1, 200)
		nd := g.w.RerankDocsMin + g.rng.Intn(g.w.RerankDocsMax-g.w.RerankDocsMin+1)
		docTokens := make([]int, nd)
		for i := range docTokens {
			docTokens[i] = lognormal(g.rng, g.w.RerankDocMedian, 0.6, 1, 4500)
		}
		if g.asleep() {
			return
		}
		p := system.RerankParams{Query: g.text(q), QueryTokens: q, DocTokens: docTokens}
		if g.textRng != nil {
			for _, d := range docTokens {
				p.Docs = append(p.Docs, g.text(d))
			}
		}
		g.rep.submitted(key)
		var h *system.Handle
		h, err := g.sys.Rerank(t.ID, p, func(res system.RerankResult) {
			delete(g.run.handles, h)
			g.rep.e2eDone(key, res.Reason, res.Done.Sub(res.Arrival))
			g.e2eSLO(key, res.Reason, res.Done.Sub(res.Arrival), g.w.SLO.Rerank)
		})
		g.track(h, key)
		g.rep.rejected(key, err)

	case tenant.LLM:
		in := lognormal(g.rng, g.w.LLMInputMedian, 0.6, 1, 6000)
		out := lognormal(g.rng, g.w.LLMOutputMedian, g.w.LLMOutputSigma, 1, 1<<20)
		cancelAfter := -1
		if g.rng.Float64() < g.w.CancelFraction {
			cancelAfter = 1 + g.rng.Intn(50)
		}
		if g.asleep() {
			return
		}
		g.rep.submitted(key)
		arrival := g.lp.Now()
		var first, last time.Time
		var f *inflight
		var h *system.Handle
		h, err := g.sys.LLM(t.ID,
			system.LLMParams{Prompt: g.text(in), InputTokens: in, MaxTokens: g.w.LLMMaxTokens, OutputLen: out},
			system.LLMCallbacks{
				OnToken: func(i int, _ string, at time.Time) {
					if i == 0 {
						first = at
						if f != nil {
							f.firstToken = at
						}
						g.rep.ttft(key, at.Sub(arrival))
					} else {
						g.rep.itl(key, at.Sub(last))
					}
					last = at
					if i+1 == cancelAfter {
						g.lp.Post(h.Cancel) // client disconnects
					}
				},
				OnDone: func(d system.LLMDone) {
					delete(g.run.handles, h)
					g.rep.llmDone(key, d.Reason)
					switch d.Reason {
					case sched.Failed:
						g.rep.slo(key, false)
					case sched.Completed:
						met := !first.IsZero() && first.Sub(arrival) <= g.w.SLO.TTFT
						if d.Generated > 1 {
							met = met && last.Sub(first)/time.Duration(d.Generated-1) <= g.w.SLO.TPOT
						}
						g.rep.slo(key, met)
					}
				},
			})
		f = g.track(h, key)
		g.rep.rejected(key, err)
	}
}
