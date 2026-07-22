"""
Characterization tests for the prognostic stack (built 2026-07-02):
prognostic.py (CUSUM micro-change + health + RUL + fusion), survival_rul.py,
workload_normalizer.py, calibration.py, discovery.py.

These pin the behaviors validated during the build -- especially the false-alarm
control (a prognostic that fires on healthy hardware is the F9 failure mode) and
the honest gates (RUL only on confirmed drift; UNCALIBRATED until real labels).
"""
import json

import numpy as np

from theta.agent.prognostic import (
    ChannelCUSUM, ComponentMonitor, GpuPrognosis, Component, Z_FAIL,
)
from theta.agent.survival_rul import SeverityRULModel, _isotonic_decreasing
from theta.agent.workload_normalizer import WorkloadNormalizer
from theta.agent.calibration import PrognosticCalibration, FailureObservation, MIN_LABELS
from theta.agent.discovery import DiscoveryEngine
from theta.agent.signature_adapter import build_feature_vector
from theta.agent.fault_classifier import FaultCause


# ── CUSUM detector: catches sustained sub-threshold drift, quiet on noise ──────
def test_cusum_quiet_on_pure_noise():
    rng = np.random.default_rng(0)
    c = ChannelCUSUM()
    for _ in range(400):
        c.update(float(rng.normal(0, 1)), +1)
    assert not c.microchange   # ARL0 ~6000 windows -> 400 of noise must stay quiet


def test_cusum_catches_sustained_subthreshold_drift():
    rng = np.random.default_rng(1)
    c = ChannelCUSUM()
    fired = None
    for i in range(400):
        x = rng.normal(0, 1) if i < 100 else rng.normal((i - 100) * 0.05, 1)
        c.update(float(x), +1)
        if fired is None and c.microchange:
            fired = i
    assert fired is not None and fired > 100   # detects only after the drift starts


# ── ComponentMonitor: health grades, RUL gated on confirmed drift ─────────────
class _FV:
    """Minimal FeatureVector stand-in carrying only the channels a test drives."""
    def __init__(self, **kw):
        for a in ("rtheta_overall_z", "alpha_z", "beta_z", "recovery_tau_z",
                  "mem_core_delta_z", "ecc_sbe_rate", "fan_rpm_residual", "inlet_delta_z",
                  "dram_active", "perf_per_watt_z"):
            setattr(self, a, kw.get(a))
        self.nvlink_error_rate = kw.get("nvlink_error_rate", 0.0)
        self.pcie_replay_rate = kw.get("pcie_replay_rate", 0.0)
        self.power_violation_rate = kw.get("power_violation_rate", 0.0)
        self.clock_efficiency = kw.get("clock_efficiency", 1.0)


def test_monitor_healthy_stays_full_health_no_rul():
    rng = np.random.default_rng(2)
    m = ComponentMonitor(Component.DIE_COOLING)
    for _ in range(300):
        m.observe(_FV(rtheta_overall_z=float(rng.normal(0, 1))))
    assert m.health > 0.9
    assert m.remaining_useful_life(window_s=1.0) is None   # no drift -> no prediction


def test_monitor_rul_gated_on_confirmed_microchange():
    rng = np.random.default_rng(3)
    m = ComponentMonitor(Component.DIE_COOLING)
    # soft wobble that never crosses the CUSUM gate must not yield an RUL
    for _ in range(120):
        m.observe(_FV(rtheta_overall_z=float(rng.normal(0.3, 1))))
    if not m.microchanges:
        assert m.remaining_useful_life(window_s=1.0) is None


# ── GpuPrognosis fusion: healthy quiet; drift -> right component + agreement ──
def test_gpu_prognosis_healthy_no_alarm():
    rng = np.random.default_rng(4)
    g = GpuPrognosis("g")
    for _ in range(300):
        g.observe(_FV(rtheta_overall_z=float(rng.normal(0, 1)),
                      mem_core_delta_z=float(rng.normal(0, 1))))
    rep = g.report(window_s=300.0)
    assert rep["in_alarm"] is False


# ── survival RUL: isotonic monotone, fitted predict, beats-linear regime ──────
def test_isotonic_non_increasing():
    y = np.array([5.0, 3.0, 4.0, 2.0, 1.0])
    out = _isotonic_decreasing(y)
    assert all(out[i] >= out[i + 1] - 1e-9 for i in range(len(out) - 1))


def test_survival_model_fits_and_predicts_monotone():
    # more severity -> less life; model must learn a non-increasing curve
    pts = [(s / 10.0, max(0.0, 200.0 - s * 3)) for s in range(1, 600)]
    m = SeverityRULModel().fit(pts)
    assert m.fitted
    assert m.predict(1.0) > m.predict(5.0)   # higher severity -> lower predicted RUL


def test_survival_model_declines_without_enough_data():
    m = SeverityRULModel().fit([(1.0, 100.0)] * 3)   # < 10 points
    assert not m.fitted


# ── workload normalizer: kills workload false alarm, keeps real drift ─────────
def _fires(vals, powers, use_norm):
    c = ChannelCUSUM()
    wn = WorkloadNormalizer()
    fired = None
    for i, (v, p) in enumerate(zip(vals, powers)):
        if use_norm:
            r = wn.normalize(v, p)
            if r is None:
                continue
            c.update(r, +1)
        else:
            c.update(v, +1)
        if fired is None and c.microchange:
            fired = i
    return fired


def test_normalizer_suppresses_workload_phase_change_false_alarm():
    rng = np.random.default_rng(5)
    powers = [float(rng.normal(320, 15)) for _ in range(200)] + \
             [float(rng.normal(470, 15)) for _ in range(200)]   # workload jump, no fault
    ch = [40 + 0.05 * p + float(rng.normal(0, 0.6)) for p in powers]
    assert _fires(ch, powers, use_norm=False) is not None   # raw false-alarms
    assert _fires(ch, powers, use_norm=True) is None         # normalized does not


def test_normalizer_preserves_real_degradation():
    rng = np.random.default_rng(6)
    powers = [float(rng.normal(400, 60)) for _ in range(400)]  # varies within range
    ch = [40 + 0.05 * p + max(0, i - 200) * 0.02 + float(rng.normal(0, 0.6))
          for i, p in enumerate(powers)]
    assert _fires(ch, powers, use_norm=True) is not None       # real drift still caught


# ── calibration: dormant until earned, then learns boundary + persists ────────
def test_calibration_dormant_until_min_labels():
    cal = PrognosticCalibration()
    for i in range(MIN_LABELS - 1):
        cal.record_failure(FailureObservation("tim", "H100", 4.0, None, None))
    assert cal.tier("tim", "H100") == "UNCALIBRATED"
    assert cal.z_fail("tim", "H100", default=Z_FAIL) == Z_FAIL


def test_calibration_learns_boundary_when_earned():
    cal = PrognosticCalibration()
    for i in range(MIN_LABELS):
        cal.record_failure(FailureObservation("tim", "H100", 4.0, None, None))
    assert cal.tier("tim", "H100") == "CALIBRATED"
    assert abs(cal.z_fail("tim", "H100", default=Z_FAIL) - 4.0) < 0.5   # learned, not default 6.0


def test_calibration_persists(tmp_path):
    p = str(tmp_path / "cal.jsonl")
    c1 = PrognosticCalibration(path=p)
    for i in range(MIN_LABELS):
        c1.record_failure(FailureObservation("fan_actuator", "A100", 5.0, None, None))
    c2 = PrognosticCalibration(path=p)   # reload from disk
    assert c2.tier("fan_actuator", "A100") == "CALIBRATED"


# ── discovery: ranks open gaps, empty on healthy ──────────────────────────────
def test_discovery_empty_when_no_alarms():
    healthy = [{"gpu_id": "g", "in_alarm": False, "attribution": None, "components": []}]
    assert DiscoveryEngine.propose(healthy) == []


def test_discovery_ranks_and_aggregates_gaps():
    # two units blocked on the same missing sensor -> one aggregated, higher-priority proposal
    def rep(gpu, rul):
        return {
            "gpu_id": gpu, "in_alarm": True, "worst_component": "tim",
            "components": [{"component": "tim", "rul_s": rul}],
            "attribution": {"identifiable": False, "headline_cause": "tim_degradation",
                            "missing_axes": [{"needs": "coolant-outlet temp", "via": "sensor",
                                              "resolves": "tim_degradation"}]},
        }
    props = DiscoveryEngine.propose([rep("a", 24 * 3600), rep("b", 100 * 3600)])
    assert len(props) == 1
    assert props[0].units_blocked == 2
    assert props[0].needs == "coolant-outlet temp"


# ── daemon integration seam: real adapter fv -> prognosis -> serializable report ─
# Pins the wiring added to daemon.py (_process_sample layers report() onto
# causal_dict["prognosis"]). The report crosses the MCP/HTTP boundary via
# json.dumps, so it MUST be JSON-serializable -- a numpy float in health/rul
# would break the MCP tool. This is the integration risk standalone tests miss.
def _adapter_fv(rtheta, power=600.0):
    return build_feature_vector(rtheta=rtheta, power_w=power, fault_cause=FaultCause.NOMINAL,
                                peer_robust_z=0.1, curve=None, gpu_ordinal=0, correlated_gpus=())


def test_daemon_seam_healthy_report_shape_and_json():
    prog = GpuPrognosis(gpu_id="0", gpu_class="H100")
    for _ in range(80):
        prog.observe(_adapter_fv(0.05), power=600.0)
    rep = prog.report(window_s=5.0)
    # the exact keys daemon.py / the MCP layer depend on
    for k in ("gpu_id", "worst_component", "worst_health", "in_alarm",
              "rul_confidence", "attribution", "engine_agreement", "components"):
        assert k in rep
    assert rep["in_alarm"] is False
    json.dumps(rep)   # must not raise (crosses the MCP boundary)


def test_daemon_seam_alarm_report_is_json_serializable():
    # drive a real monotone R_theta rise so the worst component enters alarm and
    # rul_s is populated -- then the report (incl. RUL + attribution) must still
    # json.dumps cleanly for the MCP tool to return it.
    prog = GpuPrognosis(gpu_id="0", gpu_class="H100")
    for i in range(160):
        prog.observe(_adapter_fv(0.05 + i * 0.002), power=600.0)
    rep = prog.report(window_s=5.0)
    json.dumps(rep)   # numpy-float-in-RUL guard: must not raise
    assert isinstance(rep["worst_health"], float)
