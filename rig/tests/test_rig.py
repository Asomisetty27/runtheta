"""
Tests for the E-MR rig pipeline, on synthetic run-to-throttle traces (no hardware).

Pins the behavior that matters when real data arrives: the shipped detector fires
BEFORE the throttle (positive lead time), the throttle detector ignores power-cap and
transient blips, and a set of labeled runs flips calibration to CALIBRATED + fits a
monotone survival RUL curve. If these hold on synthetic degradation, the only unknown
left at rig time is the real degradation itself.
"""
import numpy as np

from rig.throttle import (
    is_thermal_throttle, thermal_flags_from_bitmasks, first_sustained_true,
    THERMAL_HW_SLOWDOWN, SW_POWER_CAP,
)
from rig.analyze import analyze_run, synthetic_run, RunTrace
from rig.calibrate_from_runs import calibrate_from_runs, fit_survival, survival_training_points
from theta.agent.calibration import MIN_LABELS


# ── throttle detection ────────────────────────────────────────────────────────
def test_thermal_bit_vs_power_cap():
    assert is_thermal_throttle(THERMAL_HW_SLOWDOWN) is True
    assert is_thermal_throttle(SW_POWER_CAP) is False           # power cap is normal, not failure
    assert is_thermal_throttle(THERMAL_HW_SLOWDOWN | SW_POWER_CAP) is True


def test_thermal_flags_vectorized():
    flags = thermal_flags_from_bitmasks([0, SW_POWER_CAP, THERMAL_HW_SLOWDOWN, 0])
    assert list(flags) == [False, False, True, False]


def test_first_sustained_ignores_transient_blip():
    t = list(range(10))
    flags = [False, False, True, False, False, True, True, True, False, False]
    # the lone True at t=2 is a blip; the sustained run starts at t=5
    assert first_sustained_true(flags, t, min_run=3) == 5.0


def test_first_sustained_none_when_never_throttles():
    assert first_sustained_true([False] * 10, list(range(10)), min_run=3) is None


# ── lead-time analysis ────────────────────────────────────────────────────────
def test_slow_arm_shipped_cusum_fires_before_throttle():
    # slow TIM/fan-duty arm (default): the conservative shipped CUSUM has runway and
    # catches the drift before throttle -> positive lead time.
    res = analyze_run(synthetic_run(seed=1))
    assert res.t_throttle is not None
    assert res.detected is True
    assert res.lead_time_s is not None and res.lead_time_s > 0


def test_fast_fault_cusum_late_but_ksigma_catches_it():
    # acute fault (fan yank): degradation is so fast the conservative CUSUM (tuned for
    # ARL0~6000, no false alarms) fires late -- the honest E-LT fan-mode property. The
    # sensitive k-sigma sweep still catches it early. The rig measures which regime a real
    # fault falls in; the tooling must surface BOTH, not hide the conservative miss.
    fast = synthetic_run(seed=3, n=400, degrade_start=120, degrade_rate=0.010)
    res = analyze_run(fast)
    assert res.t_throttle is not None
    # the k-sigma sweep catches the fast fault with lead time even if the CUSUM did not
    ksig_leads = [v for v in res.ksigma_lead_s.values() if v is not None]
    assert ksig_leads and max(ksig_leads) > 0


def test_healthy_run_never_throttles_no_false_throttle():
    # constant cooling, no degradation -> T_j stays low -> no throttle, and the detector
    # must not manufacture one
    n = 400
    t = np.arange(n, dtype=float)
    trace = RunTrace(
        t=t, t_junction=np.full(n, 60.0) + np.random.default_rng(0).normal(0, 0.5, n),
        t_ambient=np.full(n, 25.0), power_w=np.full(n, 150.0),
        thermal_throttle=np.zeros(n, dtype=bool),
    )
    res = analyze_run(trace)
    assert res.t_throttle is None
    assert res.detected is False
    assert res.lead_time_s is None


def test_ksigma_sweep_orders_by_threshold():
    res = analyze_run(synthetic_run(seed=2))
    leads = res.ksigma_lead_s
    assert set(leads) == {2.0, 3.0, 4.0}
    got = {k: v for k, v in leads.items() if v is not None}
    # a more sensitive threshold (lower k) should fire no later -> >= lead time
    if 2.0 in got and 4.0 in got:
        assert got[2.0] >= got[4.0] - 1e-9


# ── calibration + survival ingestion (lights up the dormant stack) ────────────
def test_runs_flip_calibration_to_calibrated():
    traces = [synthetic_run(seed=s) for s in range(MIN_LABELS)]
    results = [analyze_run(tr) for tr in traces]
    cal = calibrate_from_runs(results, component="tim", gpu_class="RTX3060")
    assert cal.tier("tim", "RTX3060") == "CALIBRATED"
    # learned boundary is the observed severity at failure, not the assumed 6.0
    z = cal.z_fail("tim", "RTX3060", default=6.0)
    assert z != 6.0 and z > 0.0


def test_below_min_labels_stays_uncalibrated():
    traces = [synthetic_run(seed=s) for s in range(MIN_LABELS - 1)]
    results = [analyze_run(tr) for tr in traces]
    cal = calibrate_from_runs(results, component="tim", gpu_class="RTX3060")
    assert cal.tier("tim", "RTX3060") == "UNCALIBRATED"
    assert cal.z_fail("tim", "RTX3060", default=6.0) == 6.0


def test_survival_fits_monotone_curve_from_runs():
    traces = [synthetic_run(seed=s) for s in range(14)]   # >=10 runs -> enough points
    model = fit_survival(traces)
    assert model.fitted
    # more severity -> less remaining life
    assert model.predict(0.5) > model.predict(3.0)


def test_survival_points_are_positive_rul():
    pts = survival_training_points([synthetic_run(seed=0)])
    assert pts
    assert all(rul >= 0 for _, rul in pts)      # RUL is time-to-throttle, never negative


# ── serial logger: Pico parse + ambient join (closes the loop to a RunTrace) ──
def test_parse_pico_line_and_bad_lines():
    from rig.serial_logger import parse_line
    ok = parse_line("1500,24.5,31.2,40.0,100,1800")
    assert ok["elapsed_ms"] == 1500 and ok["inlet_c"] == 24.5 and ok["fan_rpm"] == 1800
    assert parse_line("1500,,,,,")["inlet_c"] is None   # valid elapsed, empty temp -> None
    assert parse_line("garbage") is None                # malformed -> None (serial noise)
    assert parse_line("1,2,3") is None                  # wrong arity -> None


def test_merge_ambient_nearest_host_time():
    from rig.serial_logger import merge_ambient
    # NVML capture started at epoch 1000; samples at t=0,1,2
    nvml = [{"t": 0.0, "gpu_temp_c": 60}, {"t": 1.0, "gpu_temp_c": 61}, {"t": 2.0, "gpu_temp_c": 62}]
    pico = [{"host_epoch": 1000.1, "inlet_c": 25.0}, {"host_epoch": 1001.0, "inlet_c": 26.0},
            {"host_epoch": 1002.2, "inlet_c": 27.0}]
    merged = merge_ambient(nvml, pico, nvml_start_epoch=1000.0)
    assert [round(m["ambient_c"]) for m in merged] == [25, 26, 27]   # nearest-time join


def test_build_run_trace_drops_samples_without_ambient():
    from rig.serial_logger import build_run_trace
    from rig.analyze import analyze_run
    nvml = [{"t": float(i), "gpu_temp_c": 60 + i * 0.1, "power_w": 100,
             "thermal_throttle": "False", "hotspot_temp_c": ""} for i in range(120)]
    pico = [{"host_epoch": 1000.0 + i, "inlet_c": 25.0} for i in range(120)]
    trace = build_run_trace(nvml, pico, nvml_start_epoch=1000.0)
    assert len(trace.t) == 120
    assert float(trace.t_ambient[0]) == 25.0
    # the joined trace flows straight into the analyzer without error
    res = analyze_run(trace)
    assert res.t_throttle is None   # this synthetic NVML never throttles
