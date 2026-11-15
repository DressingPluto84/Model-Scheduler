// Package loop provides the single-threaded event loop that every core
// component (schedulers, GPUs, services, accounting) runs on.
//
// All scheduling state is only ever touched from inside the loop, so the core
// needs no locks. Two implementations exist:
//
//   - Real: wall-clock time, used by the live server. Other goroutines (HTTP
//     handlers) hand work to it with Post or Do.
//   - Sim: a discrete-event simulated clock used by benchmarks and tests. Time
//     jumps straight to the next scheduled event, so runs are fast and
//     deterministic.
package loop

import "time"

// Loop is the clock and executor shared by all core components.
type Loop interface {
	// Now returns the current (real or simulated) time.
	Now() time.Time
	// After runs fn on the loop once d has elapsed.
	After(d time.Duration, fn func()) Timer
	// Post runs fn on the loop as soon as possible, after the currently
	// running callback returns.
	Post(fn func())
}

// Timer is a pending After callback.
type Timer interface {
	// Stop prevents the callback from running. It reports whether the timer
	// was still pending.
	Stop() bool
}
