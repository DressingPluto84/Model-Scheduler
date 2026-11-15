package loop

import (
	"container/heap"
	"time"
)

// Sim is a discrete-event loop with a simulated clock. It is not safe for use
// from multiple goroutines: everything, including the code that seeds the
// initial events, must run on the goroutine that calls Run.
type Sim struct {
	now    time.Time
	events eventHeap
	seq    uint64
}

// NewSim returns a simulated loop whose clock starts at start.
func NewSim(start time.Time) *Sim {
	return &Sim{now: start}
}

func (s *Sim) Now() time.Time { return s.now }

func (s *Sim) After(d time.Duration, fn func()) Timer {
	if d < 0 {
		d = 0
	}
	s.seq++
	ev := &event{at: s.now.Add(d), seq: s.seq, fn: fn}
	heap.Push(&s.events, ev)
	return ev
}

func (s *Sim) Post(fn func()) { s.After(0, fn) }

// Run processes events in time order until there are none left or the next
// event is after until. The clock finishes at until (or at the last event if
// until is zero).
func (s *Sim) Run(until time.Time) {
	for s.events.Len() > 0 {
		ev := s.events[0]
		if !until.IsZero() && ev.at.After(until) {
			break
		}
		heap.Pop(&s.events)
		if ev.stopped {
			continue
		}
		ev.fired = true
		s.now = ev.at
		ev.fn()
	}
	if !until.IsZero() && s.now.Before(until) {
		s.now = until
	}
}

// Pending reports the number of scheduled events (including stopped ones not
// yet discarded).
func (s *Sim) Pending() int { return s.events.Len() }

type event struct {
	at      time.Time
	seq     uint64
	fn      func()
	index   int
	stopped bool
	fired   bool
}

func (e *event) Stop() bool {
	if e.stopped || e.fired {
		return false
	}
	e.stopped = true
	return true
}

type eventHeap []*event

func (h eventHeap) Len() int { return len(h) }
func (h eventHeap) Less(i, j int) bool {
	if h[i].at.Equal(h[j].at) {
		return h[i].seq < h[j].seq
	}
	return h[i].at.Before(h[j].at)
}
func (h eventHeap) Swap(i, j int) {
	h[i], h[j] = h[j], h[i]
	h[i].index = i
	h[j].index = j
}
func (h *eventHeap) Push(x any) {
	e := x.(*event)
	e.index = len(*h)
	*h = append(*h, e)
}
func (h *eventHeap) Pop() any {
	old := *h
	n := len(old)
	e := old[n-1]
	old[n-1] = nil
	*h = old[:n-1]
	return e
}
