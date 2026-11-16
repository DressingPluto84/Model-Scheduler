package sched

// Payloads carried in Request.Payload for real GPU backends.

// LLMPayload is the content of an LLM generation request.
type LLMPayload struct {
	Prompt string
}

// EmbedPayload is the content of an embed request.
type EmbedPayload struct {
	Text string
}

// RerankPayload is one doc-query pair.
type RerankPayload struct {
	Query string
	Doc   string
}

// RerankAllPayload is a whole rerank request run as one job.
type RerankAllPayload struct {
	Query string
	Docs  []string
}

// Results stored in Request.Result by real GPU backends: an embed request
// gets an EmbedResult, a rerank pair a RerankScore, and a whole rerank job
// RerankScores (one per document, in input order).
type (
	EmbedResult  []float32
	RerankScore  float64
	RerankScores []float64
)
