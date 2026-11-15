// Package ratelimit implements a sliding-window-log rate limiter keyed by
// (tenant, endpoint).
package ratelimit

import "time"

// Key identifies a rate-limit bucket.
type Key struct {
	Tenant   string
	Endpoint string
}

// Limiter allows at most a given number of requests per key within any
// sliding window. It is only accessed from the event loop.
type Limiter struct {
	window time.Duration
	logs   map[Key][]time.Time
}

// New returns a limiter with the given window (e.g. one minute).
func New(window time.Duration) *Limiter {
	return &Limiter{window: window, logs: make(map[Key][]time.Time)}
}

// Allow records a request at now if fewer than limit requests for key fall in
// the window ending at now. If not allowed, it returns how long until a slot
// frees up.
func (l *Limiter) Allow(key Key, limit int, now time.Time) (bool, time.Duration) {
	log := l.logs[key]
	cutoff := now.Add(-l.window)
	i := 0
	for i < len(log) && !log[i].After(cutoff) {
		i++
	}
	log = log[i:]
	if len(log) >= limit {
		l.logs[key] = log
		if limit <= 0 || len(log) == 0 {
			return false, l.window
		}
		return false, log[len(log)-limit].Add(l.window).Sub(now)
	}
	l.logs[key] = append(log, now)
	return true, 0
}
