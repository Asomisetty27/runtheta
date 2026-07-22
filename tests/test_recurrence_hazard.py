"""Recurrence-hazard watch (F23).

Incident recurrence concentrates on repeat units on two fleets across both
cooling regimes (GWDG air: 78% of GPU incidents on repeat nodes, median 26-day
gap; Summit liquid: 25.7% per-unit detachment recurrence). A first hardware
incident therefore places a unit in an elevated re-fault hazard state for a
weeks-scale horizon.

These pin the product behavior that the paper's Validation C claims:
- the watch condition raises on a hardware incident and is INFO-level (the GPU
  stays schedulable — it works now; this is maintenance planning, not a drain);
- it persists for the bounded window and then clears on its own;
- a fresh DBE auto-opens it; XID detachment opens it via the daemon path;
- repeat incidents refresh the horizon and increment the count.
"""

from theta.agent.health import (
    RECURRENCE_WATCH_WINDOW_S,
    HealthConditionTracker,
    HealthStatus,
)
from theta.agent.metrics import GPUState

T0 = 1_700_000_000.0


def _healthy_observe(tracker, gpu, ts):
    """A clean, warmed, no-fault observation."""
    tracker.observe(gpu, ts=ts, warming=False, state=GPUState.UNDER_LOAD)


class TestRecurrenceWatch:
    def test_incident_raises_info_watch_but_stays_schedulable(self):
        t = HealthConditionTracker()
        t.record_incident(0, T0, kind="fallen_off_bus")
        _healthy_observe(t, 0, T0 + 60)
        h = t.health(0)
        names = {c.name for c in h.conditions}
        assert "RecurrentFaultWatch" in names
        # elevated hazard, but the GPU works now:
        assert h.schedulable is True
        assert h.status == HealthStatus.HEALTHY
        assert "hazard" in h.message.lower()

    def test_window_closes_after_horizon(self):
        t = HealthConditionTracker()
        t.record_incident(0, T0, kind="gpu_reset_required")
        # one second before the horizon: still watched
        _healthy_observe(t, 0, T0 + RECURRENCE_WATCH_WINDOW_S - 1)
        assert any(c.name == "RecurrentFaultWatch" for c in t.health(0).conditions)
        # past the horizon with no new incident: cleared
        _healthy_observe(t, 0, T0 + RECURRENCE_WATCH_WINDOW_S + 1)
        assert not any(c.name == "RecurrentFaultWatch" for c in t.health(0).conditions)

    def test_dbe_auto_opens_window(self):
        t = HealthConditionTracker()
        # a fresh double-bit ECC error is itself a hardware incident
        t.observe(0, ts=T0, warming=False, state=GPUState.UNDER_LOAD, ecc_dbit=1)
        watch = next(c for c in t.health(0).conditions if c.name == "RecurrentFaultWatch")
        assert watch.since == T0

    def test_dbe_records_once_not_per_tick(self):
        t = HealthConditionTracker()
        for k in range(5):                                   # same DBE persists
            t.observe(0, ts=T0 + k, warming=False,
                      state=GPUState.UNDER_LOAD, ecc_dbit=1)
        assert t._g(0).incident_count == 1                   # edge-triggered, not level

    def test_repeat_incident_refreshes_horizon_and_counts(self):
        t = HealthConditionTracker()
        t.record_incident(0, T0, kind="fallen_off_bus")
        later = T0 + RECURRENCE_WATCH_WINDOW_S - 5 * 86400   # 5 days before expiry
        t.record_incident(0, later, kind="fallen_off_bus")
        _healthy_observe(t, 0, later + 60)
        g = t._g(0)
        assert g.incident_count == 2
        # horizon now measured from the second incident, so at old-expiry it lives on
        _healthy_observe(t, 0, T0 + RECURRENCE_WATCH_WINDOW_S + 60)
        assert any(c.name == "RecurrentFaultWatch" for c in t.health(0).conditions)

    def test_no_incident_no_watch(self):
        t = HealthConditionTracker()
        _healthy_observe(t, 0, T0)
        assert not any(c.name == "RecurrentFaultWatch" for c in t.health(0).conditions)
        assert t.health(0).message == "healthy — no active problem conditions"
