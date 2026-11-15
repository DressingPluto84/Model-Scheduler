package loop

import (
	"sync"
	"time"
)

// Real is a wall-clock event loop. Callbacks run one at a time on a single
// goroutine started by Start. Its queue is unbounded, so Post never blocks:
// callbacks running on the loop can safely Post more work, however many
// backend goroutines are posting at the same time.
type Real struct {
	mu     sync.Mutex
	queue  []func()
	wake   chan struct{}
	quit   chan struct{}
	once   sync.Once
	closed bool
}

// NewReal returns a wall-clock loop. Call Start before posting work.
func NewReal() *Real {
	return &Real{wake: make(chan struct{}, 1), quit: make(chan struct{})}
}

// Start runs the loop on a new goroutine.
func (r *Real) Start() {
	go func() {
		for {
			select {
			case <-r.wake:
			case <-r.quit:
				return
			}
			for {
				r.mu.Lock()
				batch := r.queue
				r.queue = nil
				r.mu.Unlock()
				if len(batch) == 0 {
					break
				}
				for _, fn := range batch {
					fn()
				}
			}
		}
	}()
}

// Stop shuts the loop down. Pending work is dropped.
func (r *Real) Stop() {
	r.once.Do(func() {
		r.mu.Lock()
		r.closed = true
		r.mu.Unlock()
		close(r.quit)
	})
}

func (r *Real) Now() time.Time { return time.Now() }

func (r *Real) Post(fn func()) {
	r.mu.Lock()
	if r.closed {
		r.mu.Unlock()
		return
	}
	r.queue = append(r.queue, fn)
	r.mu.Unlock()
	select {
	case r.wake <- struct{}{}:
	default:
	}
}

func (r *Real) After(d time.Duration, fn func()) Timer {
	return &realTimer{t: time.AfterFunc(d, func() { r.Post(fn) })}
}

// Do runs fn on the loop and waits for it to return. It must not be called
// from inside the loop.
func (r *Real) Do(fn func()) {
	done := make(chan struct{})
	r.Post(func() {
		defer close(done)
		fn()
	})
	select {
	case <-done:
	case <-r.quit:
	}
}

type realTimer struct{ t *time.Timer }

func (t *realTimer) Stop() bool { return t.t.Stop() }
