// Package vllm backs the schedulers with real models served by vLLM's
// OpenAI-compatible servers: one server per model (LLM, embed, rerank).
//
// Our scheduler stays in charge of admission. vLLM is started with a high
// --max-num-seqs so it runs everything we admit right away, and the LLM GPU
// only admits a request when its input + max_tokens fits in the KV cache, so
// vLLM never has to queue or preempt on its own.
//
// HTTP calls happen on their own goroutines; results are posted back to the
// event loop, so these GPUs require a real (wall-clock) loop.
package vllm

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"regexp"
	"strconv"
	"strings"
	"time"
)

// Client talks to one vLLM server.
type Client struct {
	BaseURL string // e.g. http://127.0.0.1:8001
	Model   string // served model name
	HTTP    *http.Client
}

// NewClient returns a client with connection pooling sized for many
// concurrent streams.
func NewClient(baseURL, model string) *Client {
	tr := http.DefaultTransport.(*http.Transport).Clone()
	tr.MaxIdleConns = 2048
	tr.MaxIdleConnsPerHost = 2048
	return &Client{BaseURL: strings.TrimRight(baseURL, "/"), Model: model, HTTP: &http.Client{Transport: tr}}
}

func (c *Client) post(ctx context.Context, path string, body any) (*http.Response, error) {
	b, err := json.Marshal(body)
	if err != nil {
		return nil, err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.BaseURL+path, bytes.NewReader(b))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := c.HTTP.Do(req)
	if err != nil {
		return nil, err
	}
	if resp.StatusCode != http.StatusOK {
		defer resp.Body.Close()
		msg, _ := io.ReadAll(io.LimitReader(resp.Body, 2048))
		return nil, fmt.Errorf("%s%s: %s: %s", c.BaseURL, path, resp.Status, strings.TrimSpace(string(msg)))
	}
	return resp, nil
}

func (c *Client) postJSON(ctx context.Context, path string, body, out any) error {
	resp, err := c.post(ctx, path, body)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if err := json.NewDecoder(resp.Body).Decode(out); err != nil {
		return fmt.Errorf("%s%s: decode response: %w", c.BaseURL, path, err)
	}
	return nil
}

// Tokenize returns the number of tokens text has under the model's tokenizer.
func (c *Client) Tokenize(ctx context.Context, text string) (int, error) {
	var out struct {
		Count int `json:"count"`
	}
	err := c.postJSON(ctx, "/tokenize", map[string]any{"model": c.Model, "prompt": text}, &out)
	return out.Count, err
}

// WaitReady polls /health until the server answers or ctx ends.
func (c *Client) WaitReady(ctx context.Context) error {
	for {
		req, _ := http.NewRequestWithContext(ctx, http.MethodGet, c.BaseURL+"/health", nil)
		resp, err := c.HTTP.Do(req)
		if err == nil {
			resp.Body.Close()
			if resp.StatusCode == http.StatusOK {
				return nil
			}
		}
		select {
		case <-ctx.Done():
			return fmt.Errorf("%s not ready: %w", c.BaseURL, ctx.Err())
		case <-time.After(time.Second):
		}
	}
}

var cacheInfoLabel = regexp.MustCompile(`(\w+)="([^"]*)"`)

// KVCacheTokens reads the KV-cache size (num_gpu_blocks × block_size) from
// the vllm:cache_config_info metric.
func (c *Client) KVCacheTokens(ctx context.Context) (int, error) {
	req, _ := http.NewRequestWithContext(ctx, http.MethodGet, c.BaseURL+"/metrics", nil)
	resp, err := c.HTTP.Do(req)
	if err != nil {
		return 0, err
	}
	defer resp.Body.Close()
	sc := bufio.NewScanner(resp.Body)
	sc.Buffer(make([]byte, 1<<20), 1<<20)
	for sc.Scan() {
		line := sc.Text()
		if !strings.HasPrefix(line, "vllm:cache_config_info{") {
			continue
		}
		labels := map[string]string{}
		for _, m := range cacheInfoLabel.FindAllStringSubmatch(line, -1) {
			labels[m[1]] = m[2]
		}
		blocks, err1 := strconv.Atoi(labels["num_gpu_blocks"])
		size, err2 := strconv.Atoi(labels["block_size"])
		if err1 != nil || err2 != nil || blocks <= 0 || size <= 0 {
			return 0, fmt.Errorf("cache_config_info has no usable num_gpu_blocks/block_size: %s", line)
		}
		return blocks * size, nil
	}
	if err := sc.Err(); err != nil {
		return 0, err
	}
	return 0, fmt.Errorf("%s/metrics has no vllm:cache_config_info", c.BaseURL)
}

// --- LLM streaming ----------------------------------------------------------------

// Usage is vLLM's token accounting for a request.
type Usage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
}

// CompletionRequest is a streaming /v1/completions call.
type CompletionRequest struct {
	Prompt    string
	MaxTokens int
	// ExactTokens forces exactly this many output tokens (min_tokens +
	// ignore_eos), used by benchmarks. 0 lets the model stop on its own.
	ExactTokens int
}

// StreamCompletion streams a completion, calling onText for every chunk that
// carries text. It returns vLLM's final usage (nil if the server didn't send
// one). Cancelling ctx closes the connection, which aborts the request in vLLM.
func (c *Client) StreamCompletion(ctx context.Context, cr CompletionRequest, onText func(string)) (*Usage, error) {
	body := map[string]any{
		"model":          c.Model,
		"prompt":         cr.Prompt,
		"max_tokens":     cr.MaxTokens,
		"stream":         true,
		"stream_options": map[string]any{"include_usage": true},
	}
	if cr.ExactTokens > 0 {
		body["max_tokens"] = cr.ExactTokens
		body["min_tokens"] = cr.ExactTokens
		body["ignore_eos"] = true
	}
	resp, err := c.post(ctx, "/v1/completions", body)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	var usage *Usage
	sc := bufio.NewScanner(resp.Body)
	sc.Buffer(make([]byte, 1<<20), 1<<20)
	for sc.Scan() {
		line := sc.Text()
		if !strings.HasPrefix(line, "data:") {
			continue
		}
		data := strings.TrimSpace(strings.TrimPrefix(line, "data:"))
		if data == "[DONE]" {
			break
		}
		var chunk struct {
			Choices []struct {
				Text string `json:"text"`
			} `json:"choices"`
			Usage *Usage `json:"usage"`
		}
		if err := json.Unmarshal([]byte(data), &chunk); err != nil {
			return usage, fmt.Errorf("decode stream chunk: %w", err)
		}
		if chunk.Usage != nil {
			usage = chunk.Usage
		}
		for _, ch := range chunk.Choices {
			if ch.Text != "" {
				onText(ch.Text)
			}
		}
	}
	return usage, sc.Err()
}

// --- Pooling (embed / rerank) ----------------------------------------------------

// Embed returns the embedding of text from /v1/embeddings.
func (c *Client) Embed(ctx context.Context, text string) ([]float32, error) {
	var out struct {
		Data []struct {
			Embedding []float32 `json:"embedding"`
		} `json:"data"`
	}
	if err := c.postJSON(ctx, "/v1/embeddings", map[string]any{"model": c.Model, "input": text}, &out); err != nil {
		return nil, err
	}
	if len(out.Data) == 0 {
		return nil, fmt.Errorf("embeddings response has no data")
	}
	return out.Data[0].Embedding, nil
}

// Rerank scores one query-document pair with /rerank.
func (c *Client) Rerank(ctx context.Context, query, doc string) (float64, error) {
	var out struct {
		Results []struct {
			RelevanceScore float64 `json:"relevance_score"`
		} `json:"results"`
	}
	body := map[string]any{"model": c.Model, "query": query, "documents": []string{doc}}
	if err := c.postJSON(ctx, "/rerank", body, &out); err != nil {
		return 0, err
	}
	if len(out.Results) == 0 {
		return 0, fmt.Errorf("rerank response has no results")
	}
	return out.Results[0].RelevanceScore, nil
}
