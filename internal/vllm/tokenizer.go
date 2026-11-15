package vllm

import (
	"context"
	"strings"

	"github.com/DressingPluto84/Model-Scheduler/internal/tenant"
)

// Tokenizer counts tokens for a model's text.
type Tokenizer interface {
	Count(ctx context.Context, model tenant.Endpoint, text string) (int, error)
}

// WordTokenizer counts one token per whitespace-separated word. It is used
// with simulated GPUs.
type WordTokenizer struct{}

func (WordTokenizer) Count(_ context.Context, _ tenant.Endpoint, text string) (int, error) {
	return len(strings.Fields(text)), nil
}

// ServerTokenizer counts tokens with each model's own tokenizer through the
// vLLM servers' /tokenize endpoint.
type ServerTokenizer struct {
	Clients map[tenant.Endpoint]*Client
}

// NewServerTokenizer returns a tokenizer for the servers in cfg.
func NewServerTokenizer(cfg Config) *ServerTokenizer {
	llm, embed, rerank := cfg.Clients()
	return &ServerTokenizer{Clients: map[tenant.Endpoint]*Client{
		tenant.LLM: llm, tenant.Embed: embed, tenant.Rerank: rerank,
	}}
}

func (t *ServerTokenizer) Count(ctx context.Context, model tenant.Endpoint, text string) (int, error) {
	return t.Clients[model].Tokenize(ctx, text)
}
