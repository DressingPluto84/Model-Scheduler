// Package fakevllm is a stand-in for vLLM's OpenAI-compatible server, used to
// test the vLLM backend without a GPU. One server answers for all three
// models. Tokens are whitespace-separated words.
package fakevllm

import (
	"encoding/json"
	"fmt"
	"hash/fnv"
	"math/rand"
	"net/http"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

// Config sets the fake server's behaviour.
type Config struct {
	// KVBlocks × BlockSize is the KV cache size reported in /metrics.
	KVBlocks  int
	BlockSize int
	// PrefillPerToken delays the first token; DecodePerToken is the gap
	// between tokens.
	PrefillPerToken time.Duration
	DecodePerToken  time.Duration
	// PoolingPerToken is the latency of embed/rerank per input token.
	PoolingPerToken time.Duration
	// FailPrompt makes any request whose text contains it return HTTP 500.
	FailPrompt string
}

// DefaultConfig is fast enough for tests.
func DefaultConfig() Config {
	return Config{KVBlocks: 2048, BlockSize: 16, PrefillPerToken: 10 * time.Microsecond,
		DecodePerToken: 2 * time.Millisecond, PoolingPerToken: 5 * time.Microsecond}
}

// Server is the fake vLLM server.
type Server struct {
	cfg Config
	mu  sync.Mutex
	rng *rand.Rand

	// Active is the number of completion streams in progress.
	Active atomic.Int64
	// Aborted counts completion streams the client closed early.
	Aborted atomic.Int64
	// MaxActive is the largest Active seen.
	MaxActive atomic.Int64
}

// New returns a fake server.
func New(cfg Config) *Server { return &Server{cfg: cfg, rng: rand.New(rand.NewSource(1))} }

// Handler returns the HTTP routes.
func (s *Server) Handler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /health", func(w http.ResponseWriter, _ *http.Request) {})
	mux.HandleFunc("GET /metrics", s.metrics)
	mux.HandleFunc("POST /tokenize", s.tokenize)
	mux.HandleFunc("POST /v1/completions", s.completions)
	mux.HandleFunc("POST /v1/embeddings", s.embeddings)
	mux.HandleFunc("POST /rerank", s.rerank)
	return mux
}

func (s *Server) metrics(w http.ResponseWriter, _ *http.Request) {
	fmt.Fprintf(w, "# HELP vllm:cache_config_info Information of the LLMEngine CacheConfig\n")
	fmt.Fprintf(w, "vllm:cache_config_info{block_size=\"%d\",cache_dtype=\"auto\",num_gpu_blocks=\"%d\"} 1.0\n",
		s.cfg.BlockSize, s.cfg.KVBlocks)
}

func count(s string) int { return len(strings.Fields(s)) }

func decode(w http.ResponseWriter, r *http.Request, v any) bool {
	if err := json.NewDecoder(r.Body).Decode(v); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return false
	}
	return true
}

func (s *Server) fail(w http.ResponseWriter, texts ...string) bool {
	if s.cfg.FailPrompt == "" {
		return false
	}
	for _, t := range texts {
		if strings.Contains(t, s.cfg.FailPrompt) {
			http.Error(w, `{"error": "simulated failure"}`, http.StatusInternalServerError)
			return true
		}
	}
	return false
}

func (s *Server) tokenize(w http.ResponseWriter, r *http.Request) {
	var req struct {
		Prompt string `json:"prompt"`
	}
	if !decode(w, r, &req) {
		return
	}
	n := count(req.Prompt)
	tokens := make([]int, n)
	for i := range tokens {
		tokens[i] = i
	}
	json.NewEncoder(w).Encode(map[string]any{"count": n, "tokens": tokens, "max_model_len": 8192})
}

func (s *Server) completions(w http.ResponseWriter, r *http.Request) {
	var req struct {
		Prompt    string `json:"prompt"`
		MaxTokens int    `json:"max_tokens"`
		MinTokens int    `json:"min_tokens"`
		IgnoreEOS bool   `json:"ignore_eos"`
		Stream    bool   `json:"stream"`
	}
	if !decode(w, r, &req) || s.fail(w, req.Prompt) {
		return
	}
	n := req.MaxTokens
	if !req.IgnoreEOS {
		s.mu.Lock()
		n = max(req.MinTokens, 1+s.rng.Intn(max(1, req.MaxTokens)))
		s.mu.Unlock()
	}
	if !req.Stream {
		var text strings.Builder
		for i := 0; i < n; i++ {
			fmt.Fprintf(&text, " tok%d", i)
		}
		json.NewEncoder(w).Encode(map[string]any{
			"choices": []map[string]any{{"index": 0, "text": text.String(), "finish_reason": "length"}},
			"usage":   map[string]int{"prompt_tokens": count(req.Prompt), "completion_tokens": n},
		})
		return
	}
	active := s.Active.Add(1)
	defer s.Active.Add(-1)
	for {
		m := s.MaxActive.Load()
		if active <= m || s.MaxActive.CompareAndSwap(m, active) {
			break
		}
	}

	w.Header().Set("Content-Type", "text/event-stream")
	flusher, _ := w.(http.Flusher)
	send := func(v any) bool {
		b, _ := json.Marshal(v)
		if _, err := fmt.Fprintf(w, "data: %s\n\n", b); err != nil {
			return false
		}
		if flusher != nil {
			flusher.Flush()
		}
		return true
	}
	wait := func(d time.Duration) bool {
		select {
		case <-r.Context().Done():
			s.Aborted.Add(1)
			return false
		case <-time.After(d):
			return true
		}
	}
	if !wait(time.Duration(count(req.Prompt)) * s.cfg.PrefillPerToken) {
		return
	}
	for i := 0; i < n; i++ {
		if i > 0 && !wait(s.cfg.DecodePerToken) {
			return
		}
		finish := any(nil)
		if i == n-1 {
			finish = "length"
		}
		if !send(map[string]any{"choices": []map[string]any{{"index": 0, "text": fmt.Sprintf(" tok%d", i), "finish_reason": finish}}}) {
			s.Aborted.Add(1)
			return
		}
	}
	send(map[string]any{"choices": []any{}, "usage": map[string]int{
		"prompt_tokens": count(req.Prompt), "completion_tokens": n, "total_tokens": count(req.Prompt) + n}})
	fmt.Fprint(w, "data: [DONE]\n\n")
}

func (s *Server) pool(r *http.Request, tokens int) bool {
	select {
	case <-r.Context().Done():
		return false
	case <-time.After(time.Duration(tokens) * s.cfg.PoolingPerToken):
		return true
	}
}

func (s *Server) embeddings(w http.ResponseWriter, r *http.Request) {
	var req struct {
		Input string `json:"input"`
	}
	if !decode(w, r, &req) || s.fail(w, req.Input) || !s.pool(r, count(req.Input)) {
		return
	}
	v := make([]float32, 8)
	for i := range v {
		h := fnv.New32a()
		fmt.Fprintf(h, "%d:%s", i, req.Input)
		v[i] = float32(h.Sum32()%1000) / 1000
	}
	json.NewEncoder(w).Encode(map[string]any{"data": []any{map[string]any{"index": 0, "embedding": v}},
		"usage": map[string]int{"prompt_tokens": count(req.Input)}})
}

func (s *Server) rerank(w http.ResponseWriter, r *http.Request) {
	var req struct {
		Query     string   `json:"query"`
		Documents []string `json:"documents"`
	}
	if !decode(w, r, &req) || s.fail(w, append([]string{req.Query}, req.Documents...)...) {
		return
	}
	tokens := 0
	for _, d := range req.Documents {
		tokens += count(req.Query) + count(d)
	}
	if !s.pool(r, tokens) {
		return
	}
	results := make([]map[string]any, len(req.Documents))
	for i, d := range req.Documents {
		// Fraction of query words present in the document.
		dw := map[string]bool{}
		for _, x := range strings.Fields(d) {
			dw[x] = true
		}
		hit, qw := 0, strings.Fields(req.Query)
		for _, x := range qw {
			if dw[x] {
				hit++
			}
		}
		results[i] = map[string]any{"index": i, "relevance_score": float64(hit) / float64(max(1, len(qw)))}
	}
	json.NewEncoder(w).Encode(map[string]any{"results": results})
}
