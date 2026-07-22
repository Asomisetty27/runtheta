"""
Prognostic engine — continuous per-component health tracking and failure prediction.

This is the layer that turns Theta from a reactive R_θ detector into a multi-component
prognostic system (see thermalos-vault: theta_prognostic_architecture_2026_07_02).

The signature classifier (signature.py) answers "given an incident, which fault?" —
REACTIVELY, per window. This module answers the forward question: "which component is
drifting RIGHT NOW, by how much, and how long until it fails?" — CONTINUOUSLY, per GPU,
per subsystem, by tracking micro-changes in every channel before any one of them crosses
an alarm threshold.

Method: CUSUM (cumulative-sum) change detection per channel. CUSUM accumulates small,
persistent deviations from a learned baseline, so it flags a sustained drift of a
fraction of a sigma long before a k·σ threshold fires — this is the "identify
micro-changes in every component" requirement, done with standard statistical process
control (auditable, no black box). Per-component health = fusion of its channels' CUSUM
state → [0,1] index → trajectory → extrapolated remaining useful life (RUL).

HONESTY DISCIPLINE (mirrors signature.py's tiers): a component's failure threshold, and
therefore its RUL, is CALIBRATED only once validated against real labeled failures for
that component + GPU class. Until then predictions are emitted as UNCALIBRATED —
physically plausible, threshold assumed, explicitly not yet trustworthy. Ship the
capability; label the claim.

scipy-free (numpy only).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np

# Assumed failure boundary: severity (smoothed sigmas of drift above healthy baseline)
# at which a component is treated as failed for health/RUL purposes. UNCALIBRATED --
# this is the single most important number to pin against real labeled failures per
# component + GPU class; until then every RUL derived from it is a falsifiable
# prediction, not a guarantee (see module docstring).
Z_FAIL: float = 6.0


class Component(Enum):
    DIE_COOLING   = "die_cooling"        # core-to-coolant conduction path
    TIM           = "tim"                # thermal interface material specifically
    HBM_MEMORY    = "hbm_memory"
    POWER_DELIVERY = "power_delivery"    # VRM / rails
    FAN_ACTUATOR  = "fan_actuator"
    FABRIC        = "fabric"             # NVLink / PCIe interconnect
    SILICON_AGING = "silicon_aging"


# Which FeatureVector channels inform each component's health. Each entry:
# (attribute_name, direction) where direction=+1 means "larger = worse" and -1 the reverse.
# clock_efficiency=1.0 healthy -> lower is worse -> direction -1, etc.
COMPONENT_CHANNELS: dict[Component, list[tuple[str, int]]] = {
    Component.DIE_COOLING:    [("rtheta_overall_z", +1), ("alpha_z", +1), ("recovery_tau_z", +1)],
    Component.TIM:            [("beta_z", +1), ("recovery_tau_z", +1), ("mem_core_delta_z", +1)],
    Component.HBM_MEMORY:     [("mem_core_delta_z", +1), ("ecc_sbe_rate", +1)],
    Component.POWER_DELIVERY: [("power_violation_rate", +1), ("clock_efficiency", -1), ("perf_per_watt_z", +1)],
    Component.FAN_ACTUATOR:   [("fan_rpm_residual", -1), ("recovery_tau_z", +1), ("inlet_delta_z", +1)],
    Component.FABRIC:         [("nvlink_error_rate", +1), ("pcie_replay_rate", +1)],
    Component.SILICON_AGING:  [("perf_per_watt_z", +1), ("clock_efficiency", -1)],
}


# Channels whose RAW value depends on the operating point (power/workload) and are NOT
# already ratio-normalized, so they must be power-conditioned before a change detector
# sees them (generalized R_theta trick):
#   mem_core -- memory-to-core delta rises under memory-heavy load;
#   clock    -- SM clock tracks power cap / thermal state.
# Deliberately NOT including rtheta_overall_z: it is ALREADY the R_theta power normalization
# (dT/P). Re-normalizing it against power risks absorbing real thermal-degradation signal
# (TIM dry-out legitimately raises R_theta), and empirically it introduced weak spurious
# alarms on the GWDG node -- so the canonical thermal channel keeps its existing
# baseline-deviation treatment. Counter-rates (ecc, pcie, nvlink) and residual-by-
# construction channels (fan_rpm_residual, *_z scores) are left as-is. Conservative on purpose.
WORKLOAD_DEPENDENT_CHANNELS = ("mem_core_delta_z", "clock_efficiency")

# All FeatureVector attributes the monitors may read (for building a normalized copy).
_FV_ATTRS = (
    "rtheta_overall_z", "power_range_observed", "alpha_z", "beta_z", "drift_rate_z",
    "step_detected", "near_service_event", "locality", "fan_rpm_residual", "inlet_delta_z",
    "mem_core_delta_z", "dram_active", "ecc_sbe_rate", "nvlink_error_rate", "pcie_replay_rate",
    "power_violation_rate", "clock_efficiency", "recovery_tau_z", "perf_per_watt_z",
)


@dataclass
class ChannelCUSUM:
    """One channel's CUSUM change detector with an online baseline.

    Tracks a running mean/std (EWMA) as the healthy baseline, then accumulates the
    one-sided standardized deviation. `k` is the slack (in sigma) below which drift is
    treated as noise; `s_hi` accumulates upward excursions. A micro-change is flagged
    when s_hi exceeds `h` (the decision interval) even though no single sample was
    individually anomalous."""
    # k/h are set for a LONG false-alarm run length (ARL0). k=0.5 keeps sensitivity to
    # ~1 sigma sustained shifts; h=8 gives ARL0 ~6000 windows (Siegmund approx) so a
    # healthy component false-alarms on the order of once per ~6000 windows, not every
    # ~280 (which h=5 gave -- a healthy GPU firing is the F9 failure mode, not acceptable).
    # Two separated concerns:
    #   CUSUM (s_hi, k, h)  -> DETECTION: is it drifting? tuned for long false-alarm run
    #                          length (ARL0 ~6000 windows) so healthy hardware stays quiet.
    #   smooth_z            -> PROGNOSIS: HOW FAR into degradation, as a smoothed count of
    #                          sigmas above baseline. This grows gradually with the real
    #                          drift (unlike the CUSUM accumulator, which saturates fast),
    #                          so it's the signal health and RUL extrapolate against.
    k: float = 0.5          # slack: half a sigma of drift is absorbed as noise
    h: float = 8.0          # decision interval: tuned for ARL0 ~6000 windows on N(0,1)
    ewma_alpha: float = 0.02  # baseline adaptation rate (slow: baseline is "healthy normal")
    sev_alpha: float = 0.1    # severity smoothing (faster: this is the trended signal)
    mean: Optional[float] = None
    var: float = 1.0
    s_hi: float = 0.0
    smooth_z: float = 0.0   # EWMA of the oriented standardized deviation (>=0 side)
    n: int = 0
    warm_n: int = 50        # samples before baseline variance is trusted / CUSUM runs

    def update(self, x: float, direction: int) -> None:
        """Feed one observation. `direction` orients the channel so +excursion = worse."""
        xd = x * direction
        self.n += 1
        if self.mean is None:
            self.mean = xd
            return
        # freeze baseline adaptation once drifting, so a real drift doesn't get
        # absorbed into "normal" (classic CUSUM pitfall)
        if self.s_hi < self.h * 0.5:
            d = xd - self.mean
            self.mean += self.ewma_alpha * d
            self.var = (1 - self.ewma_alpha) * (self.var + self.ewma_alpha * d * d)
        if self.n < self.warm_n:
            return
        sigma = max(np.sqrt(self.var), 1e-6)
        z = (xd - self.mean) / sigma
        self.s_hi = max(0.0, self.s_hi + z - self.k)          # detector: one-sided CUSUM
        self.smooth_z += self.sev_alpha * (max(z, 0.0) - self.smooth_z)  # prognosis: EWMA of drift magnitude

    @property
    def microchange(self) -> bool:
        """DETECTION: a sustained sub-threshold drift has crossed the decision interval."""
        return self.n >= self.warm_n and self.s_hi >= self.h

    @property
    def severity(self) -> float:
        """PROGNOSIS: smoothed drift magnitude in sigmas above healthy baseline."""
        return self.smooth_z if self.n >= self.warm_n else 0.0


@dataclass
class ComponentMonitor:
    """Continuous health tracker for one subsystem of one GPU."""
    component: Component
    channels: dict[str, ChannelCUSUM] = field(default_factory=dict)
    health_history: list[float] = field(default_factory=list)  # health index over time
    gpu_class: str = "unknown"          # for calibration lookup (e.g. "H100-SXM5")
    calibration: object = None          # optional PrognosticCalibration
    rul_model: object = None            # optional fitted survival_rul.SeverityRULModel

    def _worst_severity(self) -> float:
        active = [c for c in self.channels.values() if c.n >= c.warm_n]
        return max((c.severity for c in active), default=0.0)

    def _z_fail(self) -> float:
        """The failure boundary in effect: calibrated (observed) if earned for this
        component + GPU class, else the assumed default Z_FAIL."""
        if self.calibration is not None:
            return self.calibration.z_fail(self.component.value, self.gpu_class, Z_FAIL)
        return Z_FAIL

    def confidence_tier(self) -> str:
        if self.calibration is not None:
            return self.calibration.tier(self.component.value, self.gpu_class)
        return "UNCALIBRATED"

    def observe(self, fv) -> None:
        """Update from one FeatureVector window. Unobserved channels (None) are skipped —
        an absent sensor is not evidence of health (same honesty as signature.py)."""
        for attr, direction in COMPONENT_CHANNELS[self.component]:
            val = getattr(fv, attr, None)
            if val is None:
                continue
            self.channels.setdefault(attr, ChannelCUSUM()).update(float(val), direction)
        self.health_history.append(self.health)

    @property
    def health(self) -> float:
        """Component health in [0,1] (1 = healthy). Driven by the worst-drifting channel's
        SEVERITY (smoothed sigmas of drift), mapped against an assumed failure boundary
        Z_FAIL. Gradual by construction, so it yields a trajectory RUL can extrapolate.
        A single failing subsystem is not averaged away by healthy siblings (max, not mean).

        Z_FAIL (the deviation treated as 'failed') is UNCALIBRATED -- an assumed boundary,
        not one validated against a real failure for this component + GPU class."""
        active = [c for c in self.channels.values() if c.n >= c.warm_n]
        if not active:
            return 1.0
        worst_sev = max(c.severity for c in active)
        return float(1.0 - min(1.0, worst_sev / self._z_fail()))

    @property
    def microchanges(self) -> list[str]:
        """Channels showing an accumulated sub-threshold drift right now."""
        return [name for name, c in self.channels.items() if c.microchange]

    def remaining_useful_life(self, window_s: float) -> Optional[float]:
        """Time-to-failure (seconds), gated on a CONFIRMED CUSUM drift so soft noise on
        healthy hardware never yields a phantom RUL.

        Uses the SURVIVAL model when one is fitted for this component (learned from real
        run-to-failure trajectories) -- the trustworthy path. The prior linear
        health-slope extrapolation is retained ONLY as a labelled crude fallback: C-MAPSS
        validation (2026-07-02) proved it is 300-900% wrong on real nonlinear degradation,
        so it must never be presented as reliable."""
        if not self.microchanges:
            return None
        # Preferred: data-driven survival RUL learned from real run-to-failure curves.
        if self.rul_model is not None and getattr(self.rul_model, "fitted", False):
            steps = self.rul_model.predict(self._worst_severity())  # RUL in observation steps
            if steps == steps and steps >= 0:  # not NaN, non-negative
                return float(steps * window_s)
        # Fallback (CRUDE, known-unreliable on nonlinear degradation): linear slope-to-zero.
        full = np.asarray(self.health_history, dtype=float)
        if len(full) < 20:
            return None
        t_full = np.arange(len(full)) * window_s
        # informative window: from the last time health was healthy (>=0.97) up to the
        # last point still above the floor (>0.03). This is the active-decline segment.
        healthy_idx = np.where(full >= 0.97)[0]
        start = int(healthy_idx[-1]) if len(healthy_idx) else 0
        seg = full[start:]
        seg_t = t_full[start:]
        above_floor = seg > 0.03
        if above_floor.sum() < 5:
            # already declined and now pinned at the floor -> failure predicted imminent/passed
            return 0.0 if full[-1] <= 0.03 else None
        seg, seg_t = seg[above_floor], seg_t[above_floor]
        slope, intercept = np.polyfit(seg_t, seg, 1)
        if slope >= -1e-9:            # not declining
            return None
        t_zero = -intercept / slope   # time at which fitted health hits 0
        rul = t_zero - t_full[-1]     # measured from NOW
        return float(max(rul, 0.0))


# Which prognostic Component each signature-classifier FaultCause localizes to.
# Used ONLY for the cross-engine agreement check below, not to constrain either engine.
_CAUSE_TO_COMPONENT: dict[str, Component] = {
    "tim_degradation":   Component.TIM,
    "mounting_event":    Component.TIM,          # contact loss -> same conduction path
    "dust_accumulation": Component.DIE_COOLING,
    "airflow_blockage":  Component.FAN_ACTUATOR,
    "fan_bearing_wear":  Component.FAN_ACTUATOR,
    "hbm_thermal":       Component.HBM_MEMORY,
    "fabric_link":       Component.FABRIC,
    "power_delivery":    Component.POWER_DELIVERY,
}


@dataclass
class GpuPrognosis:
    """Fused per-GPU prognosis across all components.

    Two engines compose here:
      - the prognostic monitors (this module) localize to a SUBSYSTEM, grade its
        health, and predict WHEN it fails (RUL);
      - the signature classifier (signature.classify) names the exact FAULT MODE
        within that subsystem and reports what observation would make the call exact.
    When a component is in alarm, both run and are cross-checked: agreement raises
    confidence, disagreement is surfaced (not silently resolved) -- the same
    two-independent-methods discipline that made the F7/F15 validation result strong."""
    gpu_id: str
    gpu_class: str = "unknown"           # e.g. "H100-SXM5" -- keys calibration lookups
    calibration: object = None           # optional PrognosticCalibration (dormant if None)
    rul_models: dict = field(default_factory=dict)  # {Component: fitted SeverityRULModel}
    monitors: dict[Component, ComponentMonitor] = field(default_factory=dict)
    _last_fv: object = None
    _normalizers: dict = field(default_factory=dict)  # {channel_name: WorkloadNormalizer}

    def observe(self, fv, power: Optional[float] = None) -> None:
        """Feed one telemetry window. If `power` (the operating point) is given, the
        workload-dependent channels are normalized against it BEFORE the monitors see
        them (generalized R_theta trick) so no channel confuses workload with degradation
        -- GWDG validation showed raw channels are untrustworthy on real varying workloads.
        The RAW fv is kept for the signature classifier, which does its own thing."""
        self._last_fv = fv
        fv_for_monitors = fv if power is None else self._normalize(fv, power)
        for comp in Component:
            if comp not in self.monitors:
                self.monitors[comp] = ComponentMonitor(
                    comp, gpu_class=self.gpu_class, calibration=self.calibration,
                    rul_model=self.rul_models.get(comp))
            self.monitors[comp].observe(fv_for_monitors)

    def _normalize(self, fv, power: float):
        """Return a shallow copy of fv with workload-dependent channels replaced by their
        power-conditioned residual (None during a normalizer's warmup / out-of-range,
        which the monitors correctly treat as unobserved)."""
        from types import SimpleNamespace
        try:
            from .workload_normalizer import WorkloadNormalizer
        except ImportError:
            from workload_normalizer import WorkloadNormalizer
        data = {a: getattr(fv, a, None) for a in _FV_ATTRS}
        for ch in WORKLOAD_DEPENDENT_CHANNELS:
            raw = data.get(ch)
            if raw is None:
                continue
            wn = self._normalizers.setdefault(ch, WorkloadNormalizer())
            data[ch] = wn.normalize(float(raw), power)  # residual, or None if unassessable
        return SimpleNamespace(**data)

    def _attribution(self) -> Optional[dict]:
        """Run the signature classifier on the latest window. Imported lazily so the
        prognostic layer stays usable (and testable) without pulling signature's deps."""
        if self._last_fv is None:
            return None
        try:
            from .signature import classify
        except ImportError:
            try:
                from signature import classify  # standalone / test path
            except ImportError:
                return None
        return classify(self._last_fv).as_dict()

    def report(self, window_s: float = 300.0) -> dict:
        """The forward-looking answer: which component is worst, its health, its
        micro-changes, its (uncalibrated) RUL -- and, when in alarm, the exact fault
        mode from the signature classifier plus a cross-engine agreement check."""
        rows = []
        for comp, mon in self.monitors.items():
            rows.append({
                "component": comp.value,
                "health": round(mon.health, 4),
                "microchanges": mon.microchanges,
                "rul_s": mon.remaining_useful_life(window_s),
                "rul_confidence": mon.confidence_tier(),  # per-component: CALIBRATED once earned
            })
        rows.sort(key=lambda r: r["health"])  # worst (lowest health) first
        worst = rows[0]
        in_alarm = bool(worst["microchanges"])  # a confirmed drift on the worst component

        attribution = None
        agreement = None
        if in_alarm:
            attribution = self._attribution()
            if attribution:
                cause = attribution.get("headline_cause")
                mapped = _CAUSE_TO_COMPONENT.get(cause)
                if mapped is None:
                    agreement = "n/a"          # NOMINAL / INSUFFICIENT_DATA / unmapped
                elif mapped.value == worst["component"]:
                    agreement = "agree"        # both engines point at the same subsystem
                else:
                    # both fired but at different subsystems -- surface, don't hide
                    agreement = f"conflict: prognostic={worst['component']} vs signature={mapped.value}"

        return {
            "gpu_id": self.gpu_id,
            "worst_component": worst["component"],
            "worst_health": worst["health"],
            "in_alarm": in_alarm,
            "rul_confidence": worst["rul_confidence"],  # real per-component tier (via calibration store)
            "attribution": attribution,        # exact fault mode + missing axes (None if not in alarm)
            "engine_agreement": agreement,     # agree | conflict:... | n/a | None(not in alarm)
            "components": rows,
        }
