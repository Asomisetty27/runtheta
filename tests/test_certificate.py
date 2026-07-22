"""theta certify — the verified health certificate (schema v1).

Pins the product decisions (2026-07-15, Amogh + Sam):
- a health SCORE that feeds valuation, never a price;
- history is the product: no record => explicit point-in-time downgrade;
- component attribution carries a validation grade (TIM ground-truthed,
  others heuristic) instead of implying uniform confidence;
- risk is actuarial (F23 56%/30d, window arithmetic), never a failure date;
- the refusals are printed on the document;
- integrity hash detects tampering.
"""
import time

from theta.agent.certificate import (
    REFUSALS,
    build_certificate,
    verify_certificate,
)

IDENT = {"gpu_index": 0, "name": "H100", "uuid": "GPU-x", "serial": "123",
         "vbios": "96.00", "driver": "580"}
NOW = 1_800_000_000.0


def _full_cert(**over):
    kw = dict(
        now=NOW, agent_version="0.1.13", identity=IDENT,
        record_span=(NOW - 90 * 86400, NOW), observed_hours=90 * 20.0,
        health={"status": "healthy", "schedulable": True, "conditions": []},
        econ={"observed_hours": 1800.0, "thermal_throttle_hours": 3.6,
              "min_thermal_margin_c": 14.0, "degraded_cooling": False},
        diagnosis={"cause": "tim_degradation", "confidence": 0.8,
                   "curve_slope": 0.0002, "intercept": 0.11},
        incident=None,
    )
    kw.update(over)
    return build_certificate(**kw)


class TestGrades:
    def test_record_backed_certificate(self):
        c = _full_cert()
        assert c["record"]["grade"] == "continuous-record"
        assert c["record"]["span_days"] == 90.0
        # coverage: 1800h observed / (24h * 90d) = 0.833
        assert abs(c["record"]["coverage_frac"] - 0.833) < 1e-3

    def test_no_history_is_point_in_time_and_says_so(self):
        c = _full_cert(record_span=None, observed_hours=None)
        assert c["record"]["grade"] == "point-in-time"
        assert "cannot reproduce operating history" in c["record"]["note"]

    def test_gaps_are_explicit(self):
        gaps = [{"start": NOW - 40 * 86400, "end": NOW - 38 * 86400,
                 "reason": "agent offline"}]
        c = _full_cert(gaps=gaps)
        assert c["record"]["gaps"] == gaps


class TestConditionAndAttribution:
    def test_tim_call_is_ground_truthed_others_heuristic(self):
        c = _full_cert()
        assert c["condition"]["fault_attribution"]["validation"] == "ground-truthed"
        c2 = _full_cert(diagnosis={"cause": "fan_bearing_wear", "confidence": 0.6})
        assert c2["condition"]["fault_attribution"]["validation"] == "heuristic"

    def test_throttle_residency_is_observed_fraction(self):
        c = _full_cert()   # 3.6h / 1800h
        assert abs(c["condition"]["throttle_residency_frac"] - 0.002) < 1e-6

    def test_absent_evidence_is_null_not_missing(self):
        c = _full_cert(health=None, econ=None, diagnosis=None)
        assert c["condition"]["health_status"] is None            # present, null
        assert c["condition"]["fault_attribution"]["cause"] == "insufficient_data"


class TestActuarialRisk:
    def test_open_watch_window_quotes_f23_and_remaining_days(self):
        c = _full_cert(incident={"last_incident_ts": NOW - 10 * 86400,
                                 "incident_count": 1, "kind": "fallen_off_bus"})
        r = c["risk"]
        assert r["recurrence_watch"] is True
        assert abs(r["window_remaining_days"] - 20.0) < 0.1
        assert r["p_refault_within_30d_given_incident"] == 0.56

    def test_expired_window_claims_no_elevated_hazard(self):
        c = _full_cert(incident={"last_incident_ts": NOW - 45 * 86400,
                                 "incident_count": 2, "kind": "memory_error"})
        r = c["risk"]
        assert r["recurrence_watch"] is False
        assert r["p_refault_within_30d_given_incident"] is None   # never prophecy

    def test_no_incident_no_claim(self):
        assert _full_cert()["risk"]["recurrence_watch"] is False


class TestDocumentIntegrity:
    def test_refusals_are_on_the_document(self):
        c = _full_cert()
        assert c["scope_refusals"] == REFUSALS
        assert any("No failure-date prediction" in r for r in REFUSALS)
        assert any("No performance-loss projection" in r for r in REFUSALS)

    def test_hash_verifies_and_detects_tampering(self):
        c = _full_cert()
        assert verify_certificate(c)
        c["condition"]["degraded_cooling"] = True   # forge a better/worse card
        assert not verify_certificate(c)

    def test_hash_is_deterministic(self):
        assert _full_cert()["integrity"] == _full_cert()["integrity"]
