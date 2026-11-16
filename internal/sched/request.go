// Package sched contains the per-model schedulers: a queueing policy (EEVDF
// or FCFS) feeding a small bounded dispatch queue in front of the model's GPU.
package sched

import (
	"time"

	"github.com/DressingPluto84/Model-Scheduler/internal/rbtree"
	"github.com/DressingPluto84/Model-Scheduler/internal/tenant"
)

// Model identifies which GPU a request runs on.
type Model = tenant.Endpoint

// State is where a request currently is.
type State int

const (
	StateNew        State = iota
	StateQueued           // waiting in the policy's queue
	StateDispatched       // picked into the dispatch queue, not yet on the GPU
	StateRunning          // admitted to the GPU
	StateDone
)

// DoneReason says why a request finished.
type DoneReason int

const (
	Completed DoneReason = iota
	Cancelled
	// Failed means the GPU backend returned an error (see Request.Err).
	Failed
)

func (d DoneReason) String() string {
	switch d {
	case Cancelled:
		return "cancelled"
	case Failed:
		return "failed"
	}
	return "completed"
}

// Request is a single schedulable job. For rerank this is one doc-query pair.
//
// The callbacks are set by the owning service and are always invoked on the
// event loop.
type Request struct {
	ID          uint64
	Tenant      *tenant.Tenant
	Model       Model
	InputTokens int
	// MaxTokens is the LLM decode limit (0 for embed/rerank).
	MaxTokens int
	// OutputLen is the simulated true output length. If 0 for an LLM request,
	// the GPU samples one when the request is admitted.
	OutputLen int
	Arrival   time.Time

	// Group ties together the parts of one client request that are scheduled
	// as a unit (the doc-query pairs of a rerank request). nil for a request
	// that stands alone.
	Group *Group

	// Payload is the request content a real backend needs (prompt text,
	// embedding input, or a query-document pair). Simulated GPUs ignore it.
	Payload any
	// Result is set by the GPU before the request finishes (an embedding
	// vector or a rerank score). Simulated GPUs leave it nil.
	Result any
	// Err is set by the GPU if the backend failed.
	Err error

	// OnUsage reports processed token counts (input on admission, output in
	// batches of k while decoding, flushed on completion or cancellation).
	OnUsage func(input, output int)
	// OnToken is called for each generated LLM token (index from 0). text is
	// empty for simulated GPUs.
	OnToken func(index int, text string)
	// OnDone is called exactly once when the request leaves the system.
	OnDone func(reason DoneReason)

	state    State
	seq      uint64
	deadline float64
	node     *rbtree.Node[*Request]
	lane     int // which queue the Mixed policy put it in

	// GPU-side bookkeeping.
	Generated int // output tokens produced so far
	unflushed int // output tokens not yet reported via OnUsage
	cancelled bool
}

// Group is a client request split into several scheduler requests (the
// doc-query pairs of a rerank). Under EEVDF every part shares one deadline,
// computed from the whole request's input, and once the first part is picked
// the rest follow it, so the request is scheduled like one job of its full
// size but still runs as small pairs.
type Group struct {
	// InputTokens is the input of every part together.
	InputTokens int

	deadline    float64
	deadlineSet bool
	started     bool
	queued      []*Request // parts still in the policy's queue, in arrival order
}

// State returns where the request currently is.
func (r *Request) State() State { return r.state }

// Reservation is how much GPU memory (in tokens) the request holds while it
// runs. LLM requests reserve input + max_tokens upfront so the KV cache never
// needs to evict.
func (r *Request) Reservation() int { return r.InputTokens + r.MaxTokens }

// AddUnflushed records n generated output tokens not yet reported and returns
// the new unreported total. Used by GPUs to batch VTC accounting writes.
func (r *Request) AddUnflushed(n int) int {
	r.unflushed += n
	return r.unflushed
}

// FlushUsage reports any unreported output tokens via OnUsage.
func (r *Request) FlushUsage() {
	if r.unflushed == 0 {
		return
	}
	n := r.unflushed
	r.unflushed = 0
	if r.OnUsage != nil {
		r.OnUsage(0, n)
	}
}
