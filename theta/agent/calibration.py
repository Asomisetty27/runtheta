"""
Prognostic calibration — the learning loop that moves a per-component failure
prediction from UNCALIBRATED to CALIBRATED as real, operator-confirmed failures
accumulate. Step 4 of the prognostic architecture.

The prognostic layer (prognostic.py) assumes a failure boundary (Z_FAIL = severity at
which a component is treated as failed) and derives RUL from it. That assumption is the
single most important thing to replace with ground truth. This module does exactly that:
each time a component actually fails and an operator confirms it, we record the severity
the prognostic engine had measured at failure and how far off its RUL prediction was.
Once enough real failures accumulate FOR THAT COMPONENT + GPU CLASS, the assumed boundary
is replaced by the observed one and the prediction is marked CALIBRATED.

Discipline (mirrors incident_store's "accuracy only reported when earned"):
  - Below MIN_LABELS confirmed failures for a (component, gpu_class), everything stays
    UNCALIBRATED, the default assumed boundary is used, and NO accuracy number is emitted.
  - This module is DORMANT by design until real failure labels exist -- which, as of
    2026-07, they do not (no real GPU cooling-degradation-to-failure dataset exists;
    see thermalos-vault Q_lead_time). It is built and wired so that the moment E-LT /
    the mini-rig / a fleet install produces labeled failures, calibration begins with
    zero further code changes. Ready, not pretending to be trained.

Reads confirmed labels from the labeling substrate; does not duplicate labeling.
Persisted as JSONL. scipy-free.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import mean, median
from typing import Optional

MIN_LABELS = 5   # confirmed real failures per (component, gpu_class) before we trust calibration


@dataclass
class FailureObservation:
    """One real, operator-confirmed component failure and how the prognostic engine did
    on it. This is the ground-truth record calibration learns from."""
    component:            str
    gpu_class:            str            # e.g. "H100-SXM5", "A100-SXM4-80GB"
    severity_at_failure:  float          # the prognostic severity (sigmas) measured when it actually failed
    predicted_rul_s:      Optional[float]  # RUL the engine predicted at the reference lead point (None if it never fired)
    actual_ttf_s:         Optional[float]  # actual time from that lead point to failure
    incident_id:          Optional[str] = None  # link back to incident_store, for traceability

    @property
    def rul_error_s(self) -> Optional[float]:
        if self.predicted_rul_s is None or self.actual_ttf_s is None:
            return None
        return self.predicted_rul_s - self.actual_ttf_s   # +ve = predicted too late (optimistic)


@dataclass
class ComponentCalibration:
    """Calibrated boundary + earned accuracy for one (component, gpu_class)."""
    component: str
    gpu_class: str
    observations: list[FailureObservation] = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.observations)

    @property
    def calibrated(self) -> bool:
        return self.n >= MIN_LABELS

    @property
    def z_fail(self) -> Optional[float]:
        """Observed failure boundary: robust central severity at real failures. None
        (caller falls back to the assumed default) until enough labels are earned."""
        if not self.calibrated:
            return None
        return float(median(o.severity_at_failure for o in self.observations))

    def rul_mae_s(self) -> Optional[float]:
        """Mean absolute RUL error over labeled failures -- ONLY once earned."""
        if not self.calibrated:
            return None
        errs = [abs(o.rul_error_s) for o in self.observations if o.rul_error_s is not None]
        return float(mean(errs)) if errs else None

    def rul_bias_s(self) -> Optional[float]:
        """Signed mean RUL error: +ve = engine predicts failure too late (dangerous)."""
        if not self.calibrated:
            return None
        errs = [o.rul_error_s for o in self.observations if o.rul_error_s is not None]
        return float(mean(errs)) if errs else None


class PrognosticCalibration:
    """Store + query layer for per-(component, gpu_class) failure calibration."""

    def __init__(self, path: Optional[str] = None):
        self.path = Path(path) if path else None
        self._cals: dict[tuple[str, str], ComponentCalibration] = {}
        if self.path and self.path.exists():
            self._load()

    def _key(self, component: str, gpu_class: str) -> tuple[str, str]:
        return (component, gpu_class)

    def record_failure(self, obs: FailureObservation) -> None:
        """Ingest one confirmed real failure. This is the ONLY thing that moves a
        component toward CALIBRATED -- passive predictions never do (same gate as
        incident_store's confirmed-label rule)."""
        k = self._key(obs.component, obs.gpu_class)
        cal = self._cals.setdefault(k, ComponentCalibration(obs.component, obs.gpu_class))
        cal.observations.append(obs)
        self._flush()

    def get(self, component: str, gpu_class: str) -> Optional[ComponentCalibration]:
        return self._cals.get(self._key(component, gpu_class))

    def z_fail(self, component: str, gpu_class: str, default: float) -> float:
        """The failure boundary to actually use: observed if earned, else the assumed
        default. This is the one call the prognostic layer makes."""
        cal = self.get(component, gpu_class)
        z = cal.z_fail if cal else None
        return z if z is not None else default

    def tier(self, component: str, gpu_class: str) -> str:
        cal = self.get(component, gpu_class)
        return "CALIBRATED" if (cal and cal.calibrated) else "UNCALIBRATED"

    def report(self, component: str, gpu_class: str) -> dict:
        cal = self.get(component, gpu_class)
        if cal is None:
            return {"tier": "UNCALIBRATED", "n_failures": 0, "z_fail": None,
                    "rul_mae_h": None, "rul_bias_h": None}
        mae = cal.rul_mae_s()
        bias = cal.rul_bias_s()
        return {
            "tier": "CALIBRATED" if cal.calibrated else "UNCALIBRATED",
            "n_failures": cal.n,
            "n_to_calibrate": max(0, MIN_LABELS - cal.n),
            "z_fail": cal.z_fail,
            "rul_mae_h": None if mae is None else round(mae / 3600, 2),
            "rul_bias_h": None if bias is None else round(bias / 3600, 2),
        }

    def _flush(self) -> None:
        if not self.path:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w") as f:
            for cal in self._cals.values():
                for obs in cal.observations:
                    f.write(json.dumps(asdict(obs)) + "\n")

    def _load(self) -> None:
        with self.path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obs = FailureObservation(**json.loads(line))
                k = self._key(obs.component, obs.gpu_class)
                self._cals.setdefault(k, ComponentCalibration(obs.component, obs.gpu_class)).observations.append(obs)
