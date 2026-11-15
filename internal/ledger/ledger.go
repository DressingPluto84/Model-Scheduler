// Package ledger holds the per-tenant token accounting: vruntime for
// scheduling (kept separately for each model), the rolling 24 hour billing
// usage (summed over all models), and the activity state used to decide when
// a tenant has left a model's scheduler and comes back.
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

// Model identifies a model's scheduler.
type Model = tenant.Endpoint

// Ledger tracks accounting for all tenants.
type Ledger struct {
	idleTimeout   time.Duration
	billingWindow time.Duration
	billing       map[*tenant.Tenant]*billingAcct
	models        map[modelKey]*modelAcct
}

type modelKey struct {
	t *tenant.Tenant
	m Model
}

// modelAcct is a tenant's state on one model's scheduler.
type modelAcct struct {
	vruntime float64

	inflight     int
	lastActivity time.Time
	seen         bool
	disconnected bool
}

// billingAcct is a tenant's billing usage over all models.
type billingAcct struct {
	usage      []usageEntry // entries inside the window, oldest first
	usageSum   int64
	usageTotal int64 // all-time, for reporting
}

type usageEntry struct {
	at     time.Time
	amount int64
}

// New returns a ledger. idleTimeout is how long a tenant with nothing queued
// or running on a model must stay quiet before it counts as having left that
// model's scheduler.
func New(idleTimeout, billingWindow time.Duration) *Ledger {
	return &Ledger{
		idleTimeout:   idleTimeout,
		billingWindow: billingWindow,
		billing:       make(map[*tenant.Tenant]*billingAcct),
		models:        make(map[modelKey]*modelAcct),
	}
}

func (l *Ledger) model(t *tenant.Tenant, m Model) *modelAcct {
	k := modelKey{t, m}
	a := l.models[k]
	if a == nil {
		a = &modelAcct{}
		l.models[k] = a
	}
	return a
}

func (l *Ledger) bill(t *tenant.Tenant) *billingAcct {
	a := l.billing[t]
	if a == nil {
		a = &billingAcct{}
		l.billing[t] = a
	}
	return a
}

// VRuntime returns the tenant's vruntime on model m (input + 2 × output for
// all tokens processed on m, plus any lift applied on wake-up).
func (l *Ledger) VRuntime(t *tenant.Tenant, m Model) float64 { return l.model(t, m).vruntime }

// TotalVRuntime returns the tenant's vruntime summed over all models.
func (l *Ledger) TotalVRuntime(t *tenant.Tenant) float64 {
	var sum float64
	for _, m := range tenant.Endpoints {
		sum += l.VRuntime(t, m)
	}
	return sum
}

// LiftVRuntime raises the tenant's vruntime on model m to at least v.
func (l *Ledger) LiftVRuntime(t *tenant.Tenant, m Model, v float64) {
	a := l.model(t, m)
	if v > a.vruntime {
		a.vruntime = v
	}
}

// Charge records tokens processed on model m against that model's vruntime
// and against the tenant's billing usage.
func (l *Ledger) Charge(t *tenant.Tenant, m Model, input, output int, now time.Time) {
	amount := int64(input) + OutputWeight*int64(output)
	if amount <= 0 {
		return
	}
	l.model(t, m).vruntime += float64(amount)
	b := l.bill(t)
	b.usage = append(b.usage, usageEntry{at: now, amount: amount})
	b.usageSum += amount
	b.usageTotal += amount
}

// BillingUsage returns the tenant's usage over the rolling billing window,
// summed over all models.
func (l *Ledger) BillingUsage(t *tenant.Tenant, now time.Time) int64 {
	b := l.bill(t)
	cutoff := now.Add(-l.billingWindow)
	i := 0
	for i < len(b.usage) && !b.usage[i].at.After(cutoff) {
		b.usageSum -= b.usage[i].amount
		i++
	}
	if i > 0 {
		b.usage = append(b.usage[:0], b.usage[i:]...)
	}
	return b.usageSum
}

// BillingResetIn returns how long until usage drops below limit, assuming no
// new usage. It returns 0 if usage is already below limit.
func (l *Ledger) BillingResetIn(t *tenant.Tenant, limit int64, now time.Time) time.Duration {
	sum := l.BillingUsage(t, now)
	for _, e := range l.bill(t).usage {
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
func (l *Ledger) TotalUsage(t *tenant.Tenant) int64 { return l.bill(t).usageTotal }

// IsAway reports whether the tenant has left model m's scheduler: it has
// nothing queued or running there and has either never been seen there,
// disconnected, or been idle there for at least the idle timeout.
func (l *Ledger) IsAway(t *tenant.Tenant, m Model, now time.Time) bool {
	a := l.model(t, m)
	if a.inflight > 0 {
		return false
	}
	return !a.seen || a.disconnected || now.Sub(a.lastActivity) >= l.idleTimeout
}

// Arrive records a new request from the tenant entering model m's scheduler.
func (l *Ledger) Arrive(t *tenant.Tenant, m Model, now time.Time) {
	a := l.model(t, m)
	a.inflight++
	a.seen = true
	a.disconnected = false
	a.lastActivity = now
}

// Depart records one of the tenant's requests on model m finishing (for any
// reason).
func (l *Ledger) Depart(t *tenant.Tenant, m Model, now time.Time) {
	a := l.model(t, m)
	if a.inflight > 0 {
		a.inflight--
	}
	a.lastActivity = now
}

// Disconnect records that the tenant's client went away mid-request on m.
func (l *Ledger) Disconnect(t *tenant.Tenant, m Model) { l.model(t, m).disconnected = true }

// Inflight returns the number of the tenant's requests queued or running on
// any model.
func (l *Ledger) Inflight(t *tenant.Tenant) int {
	n := 0
	for _, m := range tenant.Endpoints {
		n += l.model(t, m).inflight
	}
	return n
}

// ResetVRuntimes sets every tenant's vruntime on model m to 0. Billing is
// unaffected.
func (l *Ledger) ResetVRuntimes(m Model) {
	for k, a := range l.models {
		if k.m == m {
			a.vruntime = 0
		}
	}
}
