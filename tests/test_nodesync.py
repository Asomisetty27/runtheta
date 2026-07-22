"""Node-synchronous event discrimination (one node event != N dead GPUs).

Pins the cross-dataset finding productized: 10/14 GWDG detachment incidents
were node-synchronous (all 4 GPUs within ~30 min). A burst must be classified
node-scope, summarize exactly once, and void the device-level recurrence
records made before the burst became visible; a lone device event must keep
its F23 hazard window.
"""
from theta.agent.health import HealthConditionTracker
from theta.agent.nodesync import NodeSyncDiscriminator


class TestDiscriminator:
    def test_single_device_event_is_device_scoped(self):
        d = NodeSyncDiscriminator(fleet_size=4)
        v = d.record(0, 1000.0, "fallen_off_bus")
        assert not v.node_synchronous

    def test_full_node_burst_is_node_synchronous(self):
        # GWDG shape: all 4 GPUs of a node within 30 min
        d = NodeSyncDiscriminator(fleet_size=4)
        assert not d.record(0, 0.0, "fallen_off_bus").node_synchronous
        assert not d.record(1, 300.0, "fallen_off_bus").node_synchronous
        v3 = d.record(2, 600.0, "fallen_off_bus")   # 3/4 = 75% AND >= 3 GPUs
        assert v3.node_synchronous and v3.first_of_burst
        assert v3.burst_gpus == [0, 1, 2]

    def test_burst_summarizes_exactly_once(self):
        d = NodeSyncDiscriminator(fleet_size=4)
        d.record(0, 0.0, "fallen_off_bus")
        d.record(1, 60.0, "fallen_off_bus")
        v3 = d.record(2, 120.0, "fallen_off_bus")
        v4 = d.record(3, 180.0, "fallen_off_bus")
        assert v3.first_of_burst and not v4.first_of_burst
        assert v4.node_synchronous          # still classified, just not re-announced

    def test_two_of_eight_stays_device_scoped(self):
        # coincidence on a big node must not trip the classifier (min_frac gate)
        d = NodeSyncDiscriminator(fleet_size=8)
        d.record(0, 0.0, "fallen_off_bus")
        v = d.record(5, 100.0, "fallen_off_bus")
        assert not v.node_synchronous

    def test_events_outside_window_do_not_pool(self):
        d = NodeSyncDiscriminator(fleet_size=4, window_s=1800.0)
        d.record(0, 0.0, "fallen_off_bus")
        d.record(1, 100.0, "fallen_off_bus")
        v = d.record(2, 5000.0, "fallen_off_bus")   # first two aged out
        assert not v.node_synchronous

    def test_categories_do_not_cross_pool(self):
        # a memory fault and two detachments are not one burst
        d = NodeSyncDiscriminator(fleet_size=4)
        d.record(0, 0.0, "fallen_off_bus")
        d.record(1, 60.0, "memory_error")
        v = d.record(2, 120.0, "fallen_off_bus")
        assert not v.node_synchronous

    def test_disjoint_later_burst_announces_again(self):
        d = NodeSyncDiscriminator(fleet_size=4, window_s=600.0)
        for i, t in ((0, 0.0), (1, 60.0), (2, 120.0)):
            v = d.record(i, t, "fallen_off_bus")
        assert v.first_of_burst
        # a fresh burst two windows later must announce again
        for i, t in ((0, 2000.0), (1, 2060.0), (2, 2120.0)):
            v = d.record(i, t, "fallen_off_bus")
        assert v.first_of_burst


class TestVoidIncident:
    def test_burst_members_recorded_early_are_voided(self):
        # the daemon flow: first two events record (below threshold), the third
        # reveals the burst -> void all three
        h = HealthConditionTracker()
        d = NodeSyncDiscriminator(fleet_size=4)
        for gpu, ts in ((0, 0.0), (1, 300.0)):
            v = d.record(gpu, ts, "fallen_off_bus")
            assert not v.node_synchronous
            h.record_incident(gpu, ts, kind="fallen_off_bus")
        v = d.record(2, 600.0, "fallen_off_bus")
        assert v.node_synchronous
        since = 600.0 - v.window_s
        voided = [g for g in v.burst_gpus if h.void_incident(g, "fallen_off_bus", since)]
        assert voided == [0, 1]            # gpu2 was never recorded
        for g in (0, 1):
            assert h._g(g).last_incident_ts is None
            assert h._g(g).incident_count == 0

    def test_void_does_not_erase_older_real_incident(self):
        # a unit with a genuine incident LAST WEEK keeps it after a burst void
        h = HealthConditionTracker()
        h.record_incident(0, 1000.0, kind="fallen_off_bus")       # real, old
        h.record_incident(0, 900000.0, kind="fallen_off_bus")     # burst member
        assert h.void_incident(0, "fallen_off_bus", since_ts=899000.0)
        g = h._g(0)
        assert g.last_incident_ts == 1000.0                       # restored
        assert g.incident_count == 1

    def test_void_refuses_wrong_kind_or_old_event(self):
        h = HealthConditionTracker()
        h.record_incident(0, 1000.0, kind="memory_error")
        assert not h.void_incident(0, "fallen_off_bus", since_ts=0.0)   # kind mismatch
        assert not h.void_incident(0, "memory_error", since_ts=2000.0)  # too old
        assert h._g(0).incident_count == 1
