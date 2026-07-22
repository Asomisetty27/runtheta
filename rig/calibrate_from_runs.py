"""
Turn E-MR run-to-throttle runs into the ground truth that lights up the dormant stack.

Each real degradation run that reaches a thermal throttle is a labeled failure:
  - it feeds theta.agent.calibration -> once >= MIN_LABELS runs exist for a
    (component, gpu_class), the failure boundary flips UNCALIBRATED -> CALIBRATED and
    the assumed z_fail (6.0) is replaced by the observed median severity at failure;
  - its per-sample severity trajectory feeds theta.agent.survival_rul -> a real
    severity->RUL curve (the same survival model validated cross-domain on C-MAPSS,
    now on real GPU run-to-failure data).

This is the plumbing from "a labeled run CSV" to "the agent's predictions are
calibrated." Built + tested on synthetic runs now so real data flows straight through.
"""
from __future__ import annotations

from typing import Iterable, Optional

from theta.agent.calibration import FailureObservation, PrognosticCalibration
from theta.agent.survival_rul import SeverityRULModel

from .analyze import RunResult, RunTrace, severity_trajectory


def failure_observation(result: RunResult, component: str, gpu_class: str,
                        incident_id: Optional[str] = None) -> FailureObservation:
    """One run-to-throttle -> one FailureObservation. severity_at_failure is the shipped
    CUSUM severity at the throttle; actual_ttf_s is the observed lead time (detection ->
    throttle), so calibration can score the engine's RUL against ground truth."""
    return FailureObservation(
        component=component,
        gpu_class=gpu_class,
        severity_at_failure=result.severity_at_throttle,
        predicted_rul_s=None,               # filled once a survival model predicts at the lead point
        actual_ttf_s=result.lead_time_s,
        incident_id=incident_id,
    )


def calibrate_from_runs(results: Iterable[RunResult], component: str, gpu_class: str,
                        path: Optional[str] = None) -> PrognosticCalibration:
    """Record every run that reached a throttle as a confirmed failure. Returns the
    calibration store (CALIBRATED once >= MIN_LABELS runs for the component+class)."""
    cal = PrognosticCalibration(path)
    for res in results:
        if res.t_throttle is not None:
            cal.record_failure(failure_observation(res, component, gpu_class))
    return cal


def survival_training_points(traces: Iterable[RunTrace]) -> list[tuple[float, float]]:
    """Across runs, emit (severity, actual_RUL) pairs: at each post-warmup sample the
    remaining life is t_throttle - t. These train the survival RUL model."""
    pts: list[tuple[float, float]] = []
    for tr in traces:
        ts, sev, t_thr = severity_trajectory(tr)
        if t_thr is None:
            continue
        for ti, si in zip(ts, sev):
            if ti <= t_thr and si > 0.0:
                pts.append((float(si), float(t_thr - ti)))
    return pts


def fit_survival(traces: Iterable[RunTrace]) -> SeverityRULModel:
    """Fit the shipped survival RUL model on the runs' severity->RUL points."""
    return SeverityRULModel().fit(survival_training_points(list(traces)))
