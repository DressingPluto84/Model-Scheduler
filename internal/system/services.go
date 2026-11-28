package system

import (
	"fmt"
	"net/http"
	"time"

	"github.com/DressingPluto84/Model-Scheduler/internal/ratelimit"
	"github.com/DressingPluto84/Model-Scheduler/internal/sched"
	"github.com/DressingPluto84/Model-Scheduler/internal/tenant"
)

// Error is a request rejected by the gateway or a service.
type Error struct {
	Status     int
	Code       string
	Message    string
	RetryAfter time.Duration
}

func (e *Error) Error() string { return fmt.Sprintf("%d %s: %s", e.Status, e.Code, e.Message) }

// Error codes.
const (
	CodeUnknownTenant  = "unknown_tenant"
	CodeRateLimited    = "rate_limit_exceeded"
	CodeBillingLimited = "billing_limit_exceeded"
	CodeInputTooLarge  = "input_too_large"
	CodeMaxTokens      = "max_tokens_too_large"
	CodeInvalid        = "invalid_request"
)

// Handle lets the caller cancel a submitted request (client disconnect).
type Handle struct {
	sys  *System
	reqs []*sched.Request
}

// Cancel aborts every part of the request that has not finished. Tokens
// already processed still count towards vruntime and billing. Must be called
// on the event loop.
func (h *Handle) Cancel() {
	if len(h.reqs) == 0 {
		return
	}
	// All parts of one request run on the same model.
	h.sys.scheds[h.reqs[0].Model].Cancel(h.reqs...)
}

// --- API gateway ------------------------------------------------------------

// gateway authenticates the tenant and applies the per-(tenant, endpoint)
// sliding-window rate limit.
func (s *System) gateway(tenantID string, ep tenant.Endpoint) (*tenant.Tenant, *Error) {
	t := s.tenants.Get(tenantID)
	if t == nil {
		return nil, &Error{Status: http.StatusUnauthorized, Code: CodeUnknownTenant,
			Message: fmt.Sprintf("unknown tenant %q", tenantID)}
	}
	limit := t.RateLimit(ep)
	key := ratelimit.Key{Tenant: t.ID, Endpoint: string(ep)}
	if ok, retry := s.limiter.Allow(key, limit, s.loop.Now()); !ok {
		return nil, &Error{Status: http.StatusTooManyRequests, Code: CodeRateLimited, RetryAfter: retry,
			Message: fmt.Sprintf("rate limit exceeded: %s allows %d /%s requests per minute", t.ID, limit, ep)}
	}
	return t, nil
}

// --- Service-side checks ----------------------------------------------------

// checkBilling enforces the rolling 24 hour token cap for the tenant's tier.
func (s *System) checkBilling(t *tenant.Tenant) *Error {
	if !s.cfg.BillingEnabled {
		return nil
	}
	now := s.loop.Now()
	limit := tenant.BillingCap(t.Tier)
	used := s.ledger.BillingUsage(t, now)
	if used < limit {
		return nil
	}
	return &Error{Status: http.StatusTooManyRequests, Code: CodeBillingLimited,
		RetryAfter: s.ledger.BillingResetIn(t, limit, now),
		Message: fmt.Sprintf("billing limit exceeded: %d of %d tokens used in the last 24 hours (%s tier)",
			used, limit, t.Tier)}
}

func checkInput(n int, what string) *Error {
	if n < 1 {
		return &Error{Status: http.StatusBadRequest, Code: CodeInvalid, Message: what + " is empty"}
	}
	if n > tenant.MaxInputTokens {
		return &Error{Status: http.StatusBadRequest, Code: CodeInputTooLarge,
			Message: fmt.Sprintf("%s has %d tokens; the maximum is %d", what, n, tenant.MaxInputTokens)}
	}
	return nil
}

// --- Embed -------------------------------------------------------------------

// EmbedParams describes an embed request.
type EmbedParams struct {
	Text        string
	InputTokens int
}

// EmbedResult describes a finished embed request.
type EmbedResult struct {
	Reason      sched.DoneReason
	Err         error
	InputTokens int
	// Embedding is nil when running on simulated GPUs.
	Embedding []float32
	Arrival   time.Time
	Done      time.Time
}

// Embed submits an embedding request.
func (s *System) Embed(tenantID string, p EmbedParams, done func(EmbedResult)) (*Handle, *Error) {
	t, err := s.gateway(tenantID, tenant.Embed)
	if err != nil {
		return nil, err
	}
	if err := checkInput(p.InputTokens, "input"); err != nil {
		return nil, err
	}
	if err := s.checkBilling(t); err != nil {
		return nil, err
	}
	r := s.newRequest(t, tenant.Embed, p.InputTokens, sched.EmbedPayload{Text: p.Text})
	r.OnDone = func(reason sched.DoneReason) {
		if done == nil {
			return
		}
		res := EmbedResult{Reason: reason, Err: r.Err, InputTokens: p.InputTokens, Arrival: r.Arrival, Done: s.loop.Now()}
		if v, ok := r.Result.(sched.EmbedResult); ok {
			res.Embedding = v
		}
		done(res)
	}
	s.scheds[tenant.Embed].Submit(r)
	return &Handle{sys: s, reqs: []*sched.Request{r}}, nil
}

// --- Rerank ------------------------------------------------------------------

// RerankParams describes a rerank request. Docs may be nil on simulated GPUs;
// DocTokens always gives one entry per document.
type RerankParams struct {
	Query       string
	QueryTokens int
	Docs        []string
	DocTokens   []int
}

// RerankResult describes a finished rerank request.
type RerankResult struct {
	// Reason is Cancelled if any pair was cancelled, else Failed if any pair
	// failed.
	Reason sched.DoneReason
	Err    error
	Pairs  int
	// Scores holds one relevance score per document, in input order. It is
	// nil when running on simulated GPUs.
	Scores  []float64
	Arrival time.Time
	Done    time.Time
}

// Rerank submits a rerank request. By default it is scheduled and run as one
// job whose input is the sum over its doc-query pairs (the query counts once
// per document, as a cross-encoder reads it). With Config.RerankPairs it is
// split into one request per pair instead, and done is called once every
// pair has finished; if the client disconnects, pairs not yet computed are
// removed.
func (s *System) Rerank(tenantID string, p RerankParams, done func(RerankResult)) (*Handle, *Error) {
	t, err := s.gateway(tenantID, tenant.Rerank)
	if err != nil {
		return nil, err
	}
	if len(p.DocTokens) == 0 {
		return nil, &Error{Status: http.StatusBadRequest, Code: CodeInvalid, Message: "no documents"}
	}
	if err := checkInput(p.QueryTokens, "query"); err != nil {
		return nil, err
	}
	for i, d := range p.DocTokens {
		if err := checkInput(p.QueryTokens+d, fmt.Sprintf("query + document %d", i)); err != nil {
			return nil, err
		}
	}
	if err := s.checkBilling(t); err != nil {
		return nil, err
	}

	if !s.cfg.RerankPairs {
		return s.rerankOneJob(t, p, done), nil
	}

	h := &Handle{sys: s}
	res := RerankResult{Reason: sched.Completed, Pairs: len(p.DocTokens), Arrival: s.loop.Now()}
	group := &sched.Group{}
	for _, d := range p.DocTokens {
		group.InputTokens += p.QueryTokens + d
	}
	remaining := len(p.DocTokens)
	for i, d := range p.DocTokens {
		var payload sched.RerankPayload
		if i < len(p.Docs) {
			payload = sched.RerankPayload{Query: p.Query, Doc: p.Docs[i]}
		}
		r := s.newRequest(t, tenant.Rerank, p.QueryTokens+d, payload)
		r.Group = group
		r.OnDone = func(rs sched.DoneReason) {
			switch {
			case rs == sched.Cancelled:
				res.Reason = sched.Cancelled
			case rs == sched.Failed && res.Reason != sched.Cancelled:
				res.Reason, res.Err = sched.Failed, r.Err
			}
			if score, ok := r.Result.(sched.RerankScore); ok {
				if res.Scores == nil {
					res.Scores = make([]float64, len(p.DocTokens))
				}
				res.Scores[i] = float64(score)
			}
			if remaining--; remaining == 0 && done != nil {
				res.Done = s.loop.Now()
				done(res)
			}
		}
		h.reqs = append(h.reqs, r)
	}
	for _, r := range h.reqs {
		s.scheds[tenant.Rerank].Submit(r)
	}
	return h, nil
}

// rerankOneJob submits the whole rerank request as a single job.
func (s *System) rerankOneJob(t *tenant.Tenant, p RerankParams, done func(RerankResult)) *Handle {
	input := 0
	for _, d := range p.DocTokens {
		input += p.QueryTokens + d
	}
	r := s.newRequest(t, tenant.Rerank, input, sched.RerankAllPayload{Query: p.Query, Docs: p.Docs})
	r.OnDone = func(reason sched.DoneReason) {
		if done == nil {
			return
		}
		res := RerankResult{Reason: reason, Err: r.Err, Pairs: len(p.DocTokens), Arrival: r.Arrival, Done: s.loop.Now()}
		if scores, ok := r.Result.(sched.RerankScores); ok {
			res.Scores = scores
		}
		done(res)
	}
	s.scheds[tenant.Rerank].Submit(r)
	return &Handle{sys: s, reqs: []*sched.Request{r}}
}

// --- LLM ---------------------------------------------------------------------

// LLMParams describes an LLM generation request.
type LLMParams struct {
	Prompt      string
	InputTokens int
	// MaxTokens defaults to Config.DefaultMaxTokens when 0.
	MaxTokens int
	// OutputLen forces the output length (benchmarks and tests). 0 lets the
	// model decide (or, on simulated GPUs, samples a length).
	OutputLen int
}

// LLMDone describes a finished LLM request.
type LLMDone struct {
	Reason    sched.DoneReason
	Err       error
	Generated int
	At        time.Time
}

// LLMCallbacks receive the stream of generated tokens.
type LLMCallbacks struct {
	// OnToken gets each generated token. text is empty on simulated GPUs.
	OnToken func(index int, text string, at time.Time)
	OnDone  func(LLMDone)
}

// LLM submits a streaming generation request.
func (s *System) LLM(tenantID string, p LLMParams, cb LLMCallbacks) (*Handle, *Error) {
	t, err := s.gateway(tenantID, tenant.LLM)
	if err != nil {
		return nil, err
	}
	if err := checkInput(p.InputTokens, "prompt"); err != nil {
		return nil, err
	}
	if p.MaxTokens == 0 {
		p.MaxTokens = s.cfg.DefaultMaxTokens
	}
	if p.MaxTokens < 0 || p.MaxTokens > tenant.MaxOutputTokens {
		return nil, &Error{Status: http.StatusBadRequest, Code: CodeMaxTokens,
			Message: fmt.Sprintf("max_tokens must be between 1 and %d", tenant.MaxOutputTokens)}
	}
	if err := s.checkBilling(t); err != nil {
		return nil, err
	}
	r := s.newRequest(t, tenant.LLM, p.InputTokens, sched.LLMPayload{Prompt: p.Prompt})
	r.MaxTokens = p.MaxTokens
	r.OutputLen = min(p.OutputLen, p.MaxTokens)
	r.OnToken = func(i int, text string) {
		if cb.OnToken != nil {
			cb.OnToken(i, text, s.loop.Now())
		}
	}
	r.OnDone = func(reason sched.DoneReason) {
		if cb.OnDone != nil {
			cb.OnDone(LLMDone{Reason: reason, Err: r.Err, Generated: r.Generated, At: s.loop.Now()})
		}
	}
	s.scheds[tenant.LLM].Submit(r)
	return &Handle{sys: s, reqs: []*sched.Request{r}}, nil
}
