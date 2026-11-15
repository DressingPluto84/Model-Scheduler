// Package httpapi exposes the system over HTTP. Embed and rerank are plain
// JSON request/response; the LLM endpoint streams tokens over a websocket.
//
// The tenant is identified by the X-Tenant-ID header (or a ?tenant= query
// parameter, for websocket clients that can't set headers).
package httpapi

import (
	"encoding/json"
	"fmt"
	"hash/fnv"
	"log"
	"math"
	"net/http"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/gorilla/websocket"

	"github.com/DressingPluto84/Model-Scheduler/internal/loop"
	"github.com/DressingPluto84/Model-Scheduler/internal/sched"
	"github.com/DressingPluto84/Model-Scheduler/internal/system"
	"github.com/DressingPluto84/Model-Scheduler/internal/tenant"
	"github.com/DressingPluto84/Model-Scheduler/internal/vllm"
)

// API serves HTTP requests against a system running on a real loop.
type API struct {
	sys      *system.System
	loop     *loop.Real
	tok      vllm.Tokenizer
	upgrader websocket.Upgrader
}

// New returns an API for sys, whose loop must be lp. tok counts tokens: the
// models' own tokenizers with vLLM, or vllm.WordTokenizer with simulated GPUs.
func New(sys *system.System, lp *loop.Real, tok vllm.Tokenizer) *API {
	return &API{sys: sys, loop: lp, tok: tok, upgrader: websocket.Upgrader{
		CheckOrigin: func(*http.Request) bool { return true },
	}}
}

// Handler returns the HTTP routes.
func (a *API) Handler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("POST /v1/embed", a.embed)
	mux.HandleFunc("POST /v1/rerank", a.rerank)
	mux.HandleFunc("GET /v1/llm", a.llm)
	mux.HandleFunc("GET /v1/usage", a.usage)
	mux.HandleFunc("GET /healthz", func(w http.ResponseWriter, _ *http.Request) { w.Write([]byte("ok\n")) })
	return mux
}

func tenantID(r *http.Request) string {
	if id := r.Header.Get("X-Tenant-ID"); id != "" {
		return id
	}
	return r.URL.Query().Get("tenant")
}

// errBackend is returned when the model servers fail.
func errBackend(err error) *system.Error {
	return &system.Error{Status: http.StatusBadGateway, Code: "backend_error", Message: err.Error()}
}

// --- /v1/embed ----------------------------------------------------------------

type embedRequest struct {
	Input string `json:"input"`
}

type embedResponse struct {
	Embedding []float64 `json:"embedding"`
	Usage     usage     `json:"usage"`
}

type usage struct {
	InputTokens  int `json:"input_tokens"`
	OutputTokens int `json:"output_tokens,omitempty"`
}

func (a *API) embed(w http.ResponseWriter, r *http.Request) {
	var req embedRequest
	if !decode(w, r, &req) {
		return
	}
	n, err := a.tok.Count(r.Context(), tenant.Embed, req.Input)
	if err != nil {
		writeError(w, errBackend(err))
		return
	}
	resCh := make(chan system.EmbedResult, 1)
	var h *system.Handle
	var apiErr *system.Error
	a.loop.Do(func() {
		h, apiErr = a.sys.Embed(tenantID(r), system.EmbedParams{Text: req.Input, InputTokens: n},
			func(res system.EmbedResult) { resCh <- res })
	})
	if apiErr != nil {
		writeError(w, apiErr)
		return
	}
	select {
	case res := <-resCh:
		switch res.Reason {
		case sched.Cancelled:
			return
		case sched.Failed:
			writeError(w, errBackend(res.Err))
			return
		}
		emb := fakeEmbedding(req.Input)
		if res.Embedding != nil {
			emb = make([]float64, len(res.Embedding))
			for i, v := range res.Embedding {
				emb[i] = float64(v)
			}
		}
		writeJSON(w, http.StatusOK, embedResponse{Embedding: emb, Usage: usage{InputTokens: n}})
	case <-r.Context().Done():
		a.loop.Post(h.Cancel)
	}
}

// fakeEmbedding returns a deterministic unit vector derived from the text.
func fakeEmbedding(s string) []float64 {
	const dim = 16
	v := make([]float64, dim)
	var norm float64
	for i := range v {
		h := fnv.New64a()
		fmt.Fprintf(h, "%d:%s", i, s)
		v[i] = float64(h.Sum64()%2001)/1000 - 1
		norm += v[i] * v[i]
	}
	norm = math.Sqrt(norm)
	for i := range v {
		v[i] /= norm
	}
	return v
}

// --- /v1/rerank ---------------------------------------------------------------

type rerankRequest struct {
	Query     string   `json:"query"`
	Documents []string `json:"documents"`
}

type rerankResult struct {
	Index int     `json:"index"`
	Score float64 `json:"score"`
}

type rerankResponse struct {
	Results []rerankResult `json:"results"`
	Usage   usage          `json:"usage"`
}

func (a *API) rerank(w http.ResponseWriter, r *http.Request) {
	var req rerankRequest
	if !decode(w, r, &req) {
		return
	}
	counts, err := a.countAll(r, tenant.Rerank, append([]string{req.Query}, req.Documents...))
	if err != nil {
		writeError(w, errBackend(err))
		return
	}
	q, docs := counts[0], counts[1:]
	total := 0
	for _, d := range docs {
		total += q + d
	}
	resCh := make(chan system.RerankResult, 1)
	var h *system.Handle
	var apiErr *system.Error
	a.loop.Do(func() {
		h, apiErr = a.sys.Rerank(tenantID(r),
			system.RerankParams{Query: req.Query, QueryTokens: q, Docs: req.Documents, DocTokens: docs},
			func(res system.RerankResult) { resCh <- res })
	})
	if apiErr != nil {
		writeError(w, apiErr)
		return
	}
	select {
	case res := <-resCh:
		switch res.Reason {
		case sched.Cancelled:
			return
		case sched.Failed:
			writeError(w, errBackend(res.Err))
			return
		}
		results := make([]rerankResult, len(req.Documents))
		for i, d := range req.Documents {
			score := overlapScore(req.Query, d)
			if res.Scores != nil {
				score = res.Scores[i]
			}
			results[i] = rerankResult{Index: i, Score: score}
		}
		sort.SliceStable(results, func(i, j int) bool { return results[i].Score > results[j].Score })
		writeJSON(w, http.StatusOK, rerankResponse{Results: results, Usage: usage{InputTokens: total}})
	case <-r.Context().Done():
		a.loop.Post(h.Cancel)
	}
}

// countAll tokenizes texts concurrently.
func (a *API) countAll(r *http.Request, model tenant.Endpoint, texts []string) ([]int, error) {
	counts := make([]int, len(texts))
	errs := make([]error, len(texts))
	var wg sync.WaitGroup
	for i, t := range texts {
		wg.Add(1)
		go func() {
			defer wg.Done()
			counts[i], errs[i] = a.tok.Count(r.Context(), model, t)
		}()
	}
	wg.Wait()
	for _, err := range errs {
		if err != nil {
			return nil, err
		}
	}
	return counts, nil
}

// overlapScore is a stand-in relevance score: the fraction of query words
// that appear in the document.
func overlapScore(query, doc string) float64 {
	qw := strings.Fields(strings.ToLower(query))
	if len(qw) == 0 {
		return 0
	}
	dw := make(map[string]bool)
	for _, w := range strings.Fields(strings.ToLower(doc)) {
		dw[w] = true
	}
	hit := 0
	for _, w := range qw {
		if dw[w] {
			hit++
		}
	}
	return float64(hit) / float64(len(qw))
}

// --- /v1/llm (websocket) ------------------------------------------------------

type llmRequest struct {
	Prompt    string `json:"prompt"`
	MaxTokens int    `json:"max_tokens"`
}

type llmEvent struct {
	Type    string `json:"type"` // "token", "done" or "error"
	Index   int    `json:"index,omitempty"`
	Token   string `json:"token,omitempty"`
	Reason  string `json:"reason,omitempty"`
	Usage   *usage `json:"usage,omitempty"`
	Status  int    `json:"status,omitempty"`
	Code    string `json:"code,omitempty"`
	Message string `json:"message,omitempty"`
}

func (a *API) llm(w http.ResponseWriter, r *http.Request) {
	conn, err := a.upgrader.Upgrade(w, r, nil)
	if err != nil {
		return // Upgrade already wrote the error response
	}
	defer conn.Close()

	conn.SetReadDeadline(time.Now().Add(30 * time.Second))
	var req llmRequest
	if err := conn.ReadJSON(&req); err != nil {
		conn.WriteJSON(llmEvent{Type: "error", Status: http.StatusBadRequest, Code: system.CodeInvalid,
			Message: "expected a JSON message {\"prompt\": ..., \"max_tokens\": ...}"})
		return
	}
	conn.SetReadDeadline(time.Time{})

	n, err := a.tok.Count(r.Context(), tenant.LLM, req.Prompt)
	if err != nil {
		conn.WriteJSON(errorEvent(errBackend(err)))
		return
	}
	maxTokens := req.MaxTokens
	if maxTokens == 0 {
		maxTokens = a.sys.Config().DefaultMaxTokens
	}
	// Buffered so the event loop never blocks: at most max_tokens tokens plus
	// one done event are ever sent.
	events := make(chan llmEvent, min(maxTokens, tenant.MaxOutputTokens)+2)
	var h *system.Handle
	var apiErr *system.Error
	a.loop.Do(func() {
		h, apiErr = a.sys.LLM(tenantID(r),
			system.LLMParams{Prompt: req.Prompt, InputTokens: n, MaxTokens: req.MaxTokens},
			system.LLMCallbacks{
				OnToken: func(i int, text string, _ time.Time) {
					if text == "" {
						text = fakeToken(req.Prompt, i)
					}
					events <- llmEvent{Type: "token", Index: i, Token: text}
				},
				OnDone: func(d system.LLMDone) {
					if d.Reason == sched.Failed {
						events <- errorEvent(errBackend(d.Err))
						return
					}
					events <- llmEvent{Type: "done", Reason: d.Reason.String(),
						Usage: &usage{InputTokens: n, OutputTokens: d.Generated}}
				},
			})
	})
	if apiErr != nil {
		conn.WriteJSON(errorEvent(apiErr))
		return
	}

	// Any read error (including a close frame) means the client went away.
	gone := make(chan struct{})
	go func() {
		defer close(gone)
		for {
			if _, _, err := conn.NextReader(); err != nil {
				return
			}
		}
	}()

	for {
		select {
		case ev := <-events:
			if err := conn.WriteJSON(ev); err != nil {
				a.loop.Post(h.Cancel)
				return
			}
			if ev.Type == "done" || ev.Type == "error" {
				conn.WriteMessage(websocket.CloseMessage,
					websocket.FormatCloseMessage(websocket.CloseNormalClosure, ""))
				return
			}
		case <-gone:
			a.loop.Post(h.Cancel)
			return
		}
	}
}

var vocab = strings.Fields("the a model token GPU fair share scheduler request tenant queue " +
	"batch latency throughput deadline eligible weight premium free cache decode")

func fakeToken(prompt string, i int) string {
	h := fnv.New32a()
	fmt.Fprintf(h, "%d:%s", i, prompt)
	return vocab[h.Sum32()%uint32(len(vocab))]
}

func errorEvent(e *system.Error) llmEvent {
	return llmEvent{Type: "error", Status: e.Status, Code: e.Code, Message: e.Message}
}

// --- /v1/usage ----------------------------------------------------------------

type usageResponse struct {
	Tenant       string             `json:"tenant"`
	Tier         string             `json:"tier"`
	VRuntime     map[string]float64 `json:"vruntime"` // per model
	BillingUsed  int64              `json:"billing_used_24h"`
	BillingLimit int64              `json:"billing_limit_24h"`
	Inflight     int                `json:"inflight"`
}

func (a *API) usage(w http.ResponseWriter, r *http.Request) {
	id := tenantID(r)
	var resp *usageResponse
	a.loop.Do(func() {
		t := a.sys.Tenants().Get(id)
		if t == nil {
			return
		}
		l := a.sys.Ledger()
		resp = &usageResponse{
			Tenant: t.ID, Tier: string(t.Tier),
			VRuntime:     map[string]float64{},
			BillingUsed:  l.BillingUsage(t, a.sys.Loop().Now()),
			BillingLimit: tenant.BillingCap(t.Tier),
			Inflight:     l.Inflight(t),
		}
		for _, m := range tenant.Endpoints {
			resp.VRuntime[string(m)] = l.VRuntime(t, m)
		}
	})
	if resp == nil {
		writeError(w, &system.Error{Status: http.StatusUnauthorized, Code: system.CodeUnknownTenant,
			Message: fmt.Sprintf("unknown tenant %q", id)})
		return
	}
	writeJSON(w, http.StatusOK, resp)
}

// --- helpers ------------------------------------------------------------------

func decode(w http.ResponseWriter, r *http.Request, v any) bool {
	dec := json.NewDecoder(http.MaxBytesReader(w, r.Body, 4<<20))
	dec.DisallowUnknownFields()
	if err := dec.Decode(v); err != nil {
		writeError(w, &system.Error{Status: http.StatusBadRequest, Code: system.CodeInvalid,
			Message: "invalid JSON body: " + err.Error()})
		return false
	}
	return true
}

func writeError(w http.ResponseWriter, e *system.Error) {
	if e.RetryAfter > 0 {
		secs := int(math.Ceil(e.RetryAfter.Seconds()))
		w.Header().Set("Retry-After", strconv.Itoa(secs))
	}
	writeJSON(w, e.Status, map[string]any{"error": map[string]string{"code": e.Code, "message": e.Message}})
}

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	if err := json.NewEncoder(w).Encode(v); err != nil {
		log.Printf("write response: %v", err)
	}
}
