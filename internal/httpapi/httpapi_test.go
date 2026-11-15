package httpapi

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/gorilla/websocket"

	"github.com/DressingPluto84/Model-Scheduler/internal/loop"
	"github.com/DressingPluto84/Model-Scheduler/internal/system"
	"github.com/DressingPluto84/Model-Scheduler/internal/tenant"
	"github.com/DressingPluto84/Model-Scheduler/internal/vllm"
)

func newServer(t *testing.T) *httptest.Server {
	t.Helper()
	lp := loop.NewReal()
	lp.Start()
	t.Cleanup(lp.Stop)
	reg, _ := tenant.NewRegistry([]*tenant.Tenant{{ID: "acme", Tier: tenant.Premium}})
	sys, err := system.New(lp, reg, system.DefaultConfig(), system.GPUs{})
	if err != nil {
		t.Fatal(err)
	}
	srv := httptest.NewServer(New(sys, lp, vllm.WordTokenizer{}).Handler())
	t.Cleanup(srv.Close)
	return srv
}

func post(t *testing.T, url, tenantID, body string) (*http.Response, map[string]any) {
	t.Helper()
	req, _ := http.NewRequest(http.MethodPost, url, strings.NewReader(body))
	req.Header.Set("X-Tenant-ID", tenantID)
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	var out map[string]any
	json.NewDecoder(resp.Body).Decode(&out)
	return resp, out
}

func TestEmbedAndRerank(t *testing.T) {
	srv := newServer(t)
	resp, out := post(t, srv.URL+"/v1/embed", "acme", `{"input": "hello fair scheduler"}`)
	if resp.StatusCode != 200 || len(out["embedding"].([]any)) != 16 {
		t.Fatalf("embed: %d %v", resp.StatusCode, out)
	}
	resp, out = post(t, srv.URL+"/v1/rerank", "acme",
		`{"query": "gpu scheduler", "documents": ["cats", "a gpu scheduler", "gpu"]}`)
	if resp.StatusCode != 200 {
		t.Fatalf("rerank: %d %v", resp.StatusCode, out)
	}
	if first := out["results"].([]any)[0].(map[string]any); first["index"].(float64) != 1 {
		t.Fatalf("rerank order: %v", out["results"])
	}
	resp, _ = post(t, srv.URL+"/v1/embed", "nobody", `{"input": "x"}`)
	if resp.StatusCode != http.StatusUnauthorized {
		t.Fatalf("unknown tenant: %d", resp.StatusCode)
	}
}

func TestLLMWebsocketStream(t *testing.T) {
	srv := newServer(t)
	url := "ws" + strings.TrimPrefix(srv.URL, "http") + "/v1/llm?tenant=acme"
	conn, _, err := websocket.DefaultDialer.Dial(url, nil)
	if err != nil {
		t.Fatal(err)
	}
	defer conn.Close()
	if err := conn.WriteJSON(llmRequest{Prompt: "tell me about fair queueing", MaxTokens: 8}); err != nil {
		t.Fatal(err)
	}
	tokens := 0
	for {
		var ev llmEvent
		if err := conn.ReadJSON(&ev); err != nil {
			t.Fatalf("read: %v", err)
		}
		if ev.Type == "token" {
			if ev.Index != tokens || ev.Token == "" {
				t.Fatalf("bad token event %+v", ev)
			}
			tokens++
			continue
		}
		if ev.Type != "done" || ev.Reason != "completed" || ev.Usage.OutputTokens != tokens || tokens < 1 || tokens > 8 {
			t.Fatalf("bad final event %+v after %d tokens", ev, tokens)
		}
		return
	}
}
