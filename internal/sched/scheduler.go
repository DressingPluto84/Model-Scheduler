package sched

import (
	"fmt"

	"github.com/DressingPluto84/Model-Scheduler/internal/ledger"
	"github.com/DressingPluto84/Model-Scheduler/internal/loop"
	"github.com/DressingPluto84/Model-Scheduler/internal/tenant"
)

// GPU is the model's (simulated) accelerator as seen by its scheduler.
type GPU interface {
	// CanAdmit reports whether r fits in the GPU's free memory right now.
	CanAdmit(r *Request) bool
	// Admit starts r. The GPU reports input usage, generates output, and
	// calls the scheduler's Finish when r is done.
	Admit(r *Request)
	// Abort stops a running request (client disconnected). The GPU flushes
	// any unreported output usage and frees r's memory, then calls Finish.
	Abort(r *Request)
	// Attach gives the GPU its scheduler, for Finish and capacity callbacks.
	Attach(s *Scheduler)
}

// Scheduler queues requests for one model with a policy, keeps a bounded
// dispatch queue in front of the GPU, and feeds the GPU as space frees up.
//
// If the request at the head of the dispatch queue does not fit in the GPU's
// free memory, the scheduler waits until it does rather than skipping ahead.
type Scheduler struct {
	model  Model
	loop   loop.Loop
	ledger *ledger.Ledger
	policy Policy
	env    *Env
	gpu    GPU

	dispatchQ     []*Request
	dispatchDepth int
	seq           uint64

	// served is the weighted service (input + 2 × output) each tenant has
	// received on this model, used for fairness metrics.
	served map[*tenant.Tenant]int64

	// ratio tracks premium vs free service for the vruntime reset; see
	// reset.go.
	ratio ratioTracker

	// waiting counts each tenant's requests in the policy's queue, so we can
	// tell how often a pick was a real choice between tenants.
	waiting          map[*tenant.Tenant]int
	picks, contested int
}

// New builds a scheduler for model and attaches it to gpu.
func New(model Model, lp loop.Loop, l *ledger.Ledger, policy Policy, gpu GPU, dispatchDepth int) *Scheduler {
	if dispatchDepth < 1 {
		dispatchDepth = 1
	}
	s := &Scheduler{
		model:  model,
		loop:   lp,
		ledger: l,
		policy: policy,
		env: &Env{
			Ledger:  l,
			Model:   model,
			Active:  make(map[*tenant.Tenant]int),
			Pending: make(map[*tenant.Tenant]int64),
		},
		gpu:           gpu,
		dispatchDepth: dispatchDepth,
		served:        make(map[*tenant.Tenant]int64),
		waiting:       make(map[*tenant.Tenant]int),
	}
	s.ratio.reset(s.env.Active)
	gpu.Attach(s)
	return s
}

// Model returns the model this scheduler serves.
func (s *Scheduler) Model() Model { return s.model }

// Policy returns the queueing policy.
func (s *Scheduler) Policy() Policy { return s.policy }

// Loop returns the event loop the scheduler runs on.
func (s *Scheduler) Loop() loop.Loop { return s.loop }

// Submit enqueues r and dispatches whatever can run.
func (s *Scheduler) Submit(r *Request) {
	if r.state != StateNew {
		panic(fmt.Sprintf("request %d submitted twice", r.ID))
	}
	now := s.loop.Now()
	t := r.Tenant
	waking := s.ledger.IsAway(t, s.model, now)
	s.ledger.Arrive(t, s.model, now)

	s.seq++
	r.seq = s.seq
	if r.Arrival.IsZero() {
		r.Arrival = now
	}
	r.state = StateQueued
	s.ratio.see(t)
	usage := r.OnUsage
	r.OnUsage = func(in, out int) {
		amount := int64(in) + ledger.OutputWeight*int64(out)
		s.served[t] += amount
		s.ratio.add(t, amount)
		if usage != nil {
			usage(in, out)
		}
	}
	s.policy.Add(r, s.env, waking)
	s.waiting[t]++
	s.env.Active[t]++
	s.pump()
}

// Cancel removes the given requests wherever they are. Tokens already
// processed still count. All requests are removed before anything new is
// dispatched, so cancelling a multi-part request (rerank pairs) never admits
// one of its own remaining parts in between.
func (s *Scheduler) Cancel(rs ...*Request) {
	for _, r := range rs {
		s.cancel(r)
	}
	s.pump()
}

func (s *Scheduler) cancel(r *Request) {
	switch r.state {
	case StateQueued:
		s.policy.Remove(r, s.env)
		s.unwait(r.Tenant)
		s.finish(r, Cancelled)
	case StateDispatched:
		for i, q := range s.dispatchQ {
			if q == r {
				s.dispatchQ = append(s.dispatchQ[:i], s.dispatchQ[i+1:]...)
				break
			}
		}
		s.removePending(r)
		s.finish(r, Cancelled)
	case StateRunning:
		r.cancelled = true
		s.gpu.Abort(r) // calls Finish
	}
}

// Finish is called by the GPU when a running request completes or is aborted.
func (s *Scheduler) Finish(r *Request) {
	reason := Completed
	switch {
	case r.cancelled:
		reason = Cancelled
	case r.Err != nil:
		reason = Failed
	}
	s.finish(r, reason)
}

func (s *Scheduler) unwait(t *tenant.Tenant) {
	if s.waiting[t]--; s.waiting[t] <= 0 {
		delete(s.waiting, t)
	}
}

// CapacityFreed is called by the GPU whenever memory frees up.
func (s *Scheduler) CapacityFreed() { s.pump() }

func (s *Scheduler) finish(r *Request, reason DoneReason) {
	if r.state == StateDone {
		return
	}
	r.state = StateDone
	t := r.Tenant
	if s.env.Active[t]--; s.env.Active[t] <= 0 {
		delete(s.env.Active, t)
	}
	now := s.loop.Now()
	s.ledger.Depart(t, s.model, now)
	if reason == Cancelled {
		s.ledger.Disconnect(t, s.model)
	}
	if r.OnDone != nil {
		r.OnDone(reason)
	}
}

func (s *Scheduler) removePending(r *Request) {
	t := r.Tenant
	if s.env.Pending[t] -= int64(r.InputTokens); s.env.Pending[t] <= 0 {
		delete(s.env.Pending, t)
	}
}

// pump moves requests from the policy into the dispatch queue and from the
// head of the dispatch queue onto the GPU until neither can make progress.
func (s *Scheduler) pump() {
	s.maybeResetVRuntimes()
	for {
		progressed := false
		for len(s.dispatchQ) < s.dispatchDepth && s.policy.Len() > 0 {
			s.picks++
			if len(s.waiting) >= 2 {
				s.contested++
			}
			r := s.policy.Pick(s.env)
			s.unwait(r.Tenant)
			r.state = StateDispatched
			s.env.Pending[r.Tenant] += int64(r.InputTokens)
			s.dispatchQ = append(s.dispatchQ, r)
			progressed = true
		}
		for len(s.dispatchQ) > 0 && s.gpu.CanAdmit(s.dispatchQ[0]) {
			r := s.dispatchQ[0]
			s.dispatchQ[0] = nil
			s.dispatchQ = s.dispatchQ[1:]
			s.removePending(r)
			r.state = StateRunning
			s.gpu.Admit(r)
			progressed = true
		}
		if !progressed {
			return
		}
	}
}

// Stats is a snapshot of scheduler state.
type Stats struct {
	// Resets is how many times the ratio check reset this model's vruntimes.
	Resets int
	// Picks is how many requests the policy has picked. Contested is how
	// many of those picks had requests from two or more tenants waiting,
	// i.e. when the policy's choice could matter.
	Picks, Contested int
	Queued           int
	Dispatched       int
	Active           map[*tenant.Tenant]int
	Served           map[*tenant.Tenant]int64
}

// Stats returns a copy of the scheduler's current state.
func (s *Scheduler) Stats() Stats {
	st := Stats{
		Resets:     s.ratio.resets,
		Picks:      s.picks,
		Contested:  s.contested,
		Queued:     s.policy.Len(),
		Dispatched: len(s.dispatchQ),
		Active:     make(map[*tenant.Tenant]int, len(s.env.Active)),
		Served:     make(map[*tenant.Tenant]int64, len(s.served)),
	}
	for t, n := range s.env.Active {
		st.Active[t] = n
	}
	for t, n := range s.served {
		st.Served[t] = n
	}
	return st
}
