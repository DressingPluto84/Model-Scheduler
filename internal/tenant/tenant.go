// Package tenant defines tenants, their tiers, and the per-tier limits from
// the design.
package tenant

import (
	"encoding/json"
	"fmt"
	"os"
)

// Tier is a customer status level.
type Tier string

const (
	Free    Tier = "free"
	Premium Tier = "premium"
)

// Endpoint is a public API endpoint.
type Endpoint string

const (
	Embed  Endpoint = "embed"
	Rerank Endpoint = "rerank"
	LLM    Endpoint = "llm"
)

// Endpoints lists all endpoints in a stable order.
var Endpoints = []Endpoint{Embed, Rerank, LLM}

// Tenant is a customer of the platform.
type Tenant struct {
	ID   string `json:"id"`
	Tier Tier   `json:"tier"`
}

// Weight is the EEVDF weight: 3 for premium, 1 for free.
func (t *Tenant) Weight() float64 {
	if t.Tier == Premium {
		return 3
	}
	return 1
}

// RateLimitPerMinute is the gateway's per-(tenant, endpoint) request limit.
func RateLimitPerMinute(tier Tier, ep Endpoint) int {
	free := map[Endpoint]int{Rerank: 10, Embed: 15, LLM: 5}
	premium := map[Endpoint]int{Rerank: 20, Embed: 30, LLM: 10}
	if tier == Premium {
		return premium[ep]
	}
	return free[ep]
}

// BillingCap is the rolling 24 hour token limit (input + 2 × output).
func BillingCap(tier Tier) int64 {
	if tier == Premium {
		return 86_000
	}
	return 43_000
}

const (
	// MaxInputTokens is the largest input allowed in a single request (or a
	// single doc-query pair for rerank).
	MaxInputTokens = 4096
	// MaxOutputTokens is the largest max_tokens allowed for LLM decode.
	MaxOutputTokens = 4096
)

// Registry is an in-memory set of known tenants.
type Registry struct {
	byID map[string]*Tenant
}

// NewRegistry builds a registry from a list of tenants.
func NewRegistry(ts []*Tenant) (*Registry, error) {
	r := &Registry{byID: make(map[string]*Tenant, len(ts))}
	for _, t := range ts {
		if t.ID == "" {
			return nil, fmt.Errorf("tenant with empty id")
		}
		if t.Tier != Free && t.Tier != Premium {
			return nil, fmt.Errorf("tenant %q: unknown tier %q", t.ID, t.Tier)
		}
		if _, dup := r.byID[t.ID]; dup {
			return nil, fmt.Errorf("duplicate tenant %q", t.ID)
		}
		r.byID[t.ID] = t
	}
	return r, nil
}

// LoadRegistry reads a JSON array of tenants from path.
func LoadRegistry(path string) (*Registry, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var ts []*Tenant
	if err := json.Unmarshal(b, &ts); err != nil {
		return nil, fmt.Errorf("parse %s: %w", path, err)
	}
	return NewRegistry(ts)
}

// Get returns the tenant with the given id, or nil.
func (r *Registry) Get(id string) *Tenant { return r.byID[id] }

// All returns every registered tenant.
func (r *Registry) All() []*Tenant {
	out := make([]*Tenant, 0, len(r.byID))
	for _, t := range r.byID {
		out = append(out, t)
	}
	return out
}
