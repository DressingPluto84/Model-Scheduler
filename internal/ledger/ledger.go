// Package ledger holds the global per-tenant token accounting: vruntime for
// scheduling, the rolling 24 hour billing usage, and the activity state used
// to decide when a tenant has left and comes back.
//
// Both vruntime and billing count input + 2 × output tokens. vruntime may be
// lifted when a tenant comes back after being away; billing never is.
//
// The ledger is only accessed from the event loop.
package ledger

import (
	"time"

	"github.com/DressingPluto84/Model-Scheduler/internal/tenant"
)

// OutputWeight is how much one output token counts relative to an input token.
const OutputWeight = 2

// Ledger tracks accounting for all tenants.
type Ledger struct {
	idleTimeout   time.Duration
	billingWindow time.Duration
	accts         map[*tenant.Tenant]*account
}

type account struct {
	vruntime float64

	usage      []usageEntry // billing entries inside the window, oldest first
	usageSum   int64
	usageTotal int64 // all-time, for reporting

	inflight     int
	lastActivity time.Time
	seen         bool
	disconnected bool
}

type usageEntry struct {
	at     time.Time
	amount int64
}

// New returns a ledger. idleTimeout is how long a tenant with nothing queued
// or running must stay quiet before it counts as having left.
func New(idleTimeout, billingWindow time.Duration) *Ledger {
	return &Ledger{
		idleTimeout:   idleTimeout,
		billingWindow: billingWindow,
		accts:         make(map[*tenant.Tenant]*account),
	}
}

func (l *Ledger) acct(t *tenant.Tenant) *account {
	a := l.accts[t]
	if a == nil {
		a = &account{}
		l.accts[t] = a
	}
	return a
}

// VRuntime returns the tenant's vruntime (input + 2 × output for all tokens
// processed, plus any lift applied on wake-up).
func (l *Ledger) VRuntime(t *tenant.Tenant) float64 { return l.acct(t).vruntime }

// LiftVRuntime raises the tenant's vruntime to at least v.
func (l *Ledger) LiftVRuntime(t *tenant.Tenant, v float64) {
	a := l.acct(t)
	if v > a.vruntime {
		a.vruntime = v
	}
}

// Charge records processed tokens against both vruntime and billing.
func (l *Ledger) Charge(t *tenant.Tenant, input, output int, now time.Time) {
	amount := int64(input) + OutputWeight*int64(output)
	if amount <= 0 {
		return
	}
	a := l.acct(t)
	a.vruntime += float64(amount)
	a.usage = append(a.usage, usageEntry{at: now, amount: amount})
	a.usageSum += amount
	a.usageTotal += amount
}

// BillingUsage returns the tenant's usage over the rolling billing window.
func (l *Ledger) BillingUsage(t *tenant.Tenant, now time.Time) int64 {
	a := l.acct(t)
	cutoff := now.Add(-l.billingWindow)
	i := 0
	for i < len(a.usage) && !a.usage[i].at.After(cutoff) {
		a.usageSum -= a.usage[i].amount
		i++
	}
	if i > 0 {
		a.usage = append(a.usage[:0], a.usage[i:]...)
	}
	return a.usageSum
}

// BillingResetIn returns how long until usage drops below limit, assuming no
// new usage. It returns 0 if usage is already below limit.
func (l *Ledger) BillingResetIn(t *tenant.Tenant, limit int64, now time.Time) time.Duration {
	sum := l.BillingUsage(t, now)
	a := l.acct(t)
	for _, e := range a.usage {
		if sum < limit {
			break
		}
		sum -= e.amount
		if sum < limit {
			return e.at.Add(l.billingWindow).Sub(now)
		}
	}
	return 0
}

// TotalUsage returns all-time billed tokens for the tenant.
func (l *Ledger) TotalUsage(t *tenant.Tenant) int64 { return l.acct(t).usageTotal }

// IsAway reports whether the tenant has left: it has nothing queued or
// running and has either never been seen, disconnected, or been idle for at
// least the idle timeout.
func (l *Ledger) IsAway(t *tenant.Tenant, now time.Time) bool {
	a := l.acct(t)
	if a.inflight > 0 {
		return false
	}
	return !a.seen || a.disconnected || now.Sub(a.lastActivity) >= l.idleTimeout
}

// Arrive records a new request from the tenant entering a scheduler.
func (l *Ledger) Arrive(t *tenant.Tenant, now time.Time) {
	a := l.acct(t)
	a.inflight++
	a.seen = true
	a.disconnected = false
	a.lastActivity = now
}

// Depart records one of the tenant's requests finishing (for any reason).
func (l *Ledger) Depart(t *tenant.Tenant, now time.Time) {
	a := l.acct(t)
	if a.inflight > 0 {
		a.inflight--
	}
	a.lastActivity = now
}

// Disconnect records that the tenant's client went away mid-request.
func (l *Ledger) Disconnect(t *tenant.Tenant) { l.acct(t).disconnected = true }

// Inflight returns the number of the tenant's requests queued or running.
func (l *Ledger) Inflight(t *tenant.Tenant) int { return l.acct(t).inflight }
