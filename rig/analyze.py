"""
Lead-time analysis for one E-MR run: lead_time = t_throttle - t_anomaly.

t_anomaly is produced by the SHIPPED detector (theta.agent.prognostic.ChannelCUSUM),
so the number the rig reports is the number the product would have produced -- no
lab-only detector. t_throttle is the first sustained thermal throttle (throttle.py).
R_theta = (T_j - T_ambient) / P, gated at low power (the R_theta=(T-amb)/P blow-up as
P->0 is a known false-positive mode; below the gate R_theta is undefined, not trusted).

A k-sigma sweep (baseline + k*sigma, sustained) is also reported, to compare against the
E-LT/Q_lead_time framing and to produce the lead-time-vs-threshold curve.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from theta.agent.prognostic import ChannelCUSUM
from .throttle import first_sustained_true

MIN_POWER_W = 20.0   # below this, R_theta is not trusted (low-power FP mode)


@dataclass
class RunTrace:
    """One degradation run, sampled ~1 Hz. Arrays are parallel and equal length."""
    t: np.ndarray            # seconds from run start
    t_junction: np.ndarray   # GPU junction/primary temp (C)
    t_ambient: np.ndarray    # MEASURED inlet ambient (C)
    power_w: np.ndarray      # board power (W)
    thermal_throttle: np.ndarray  # bool: thermal-slowdown active
    hotspot: Optional[np.ndarray] = None   # hotspot temp (C), if the card exposes it

    def rtheta(self, min_power_w: float = MIN_POWER_W) -> np.ndarray:
        """R_theta per sample; NaN where power is below the trust gate."""
        p = np.asarray(self.power_w, float)
        dt = np.asarray(self.t_junction, float) - np.asarray(self.t_ambient, float)
        r = np.where(p >= min_power_w, dt / np.where(p == 0, np.nan, p), np.nan)
        return r


@dataclass
class RunResult:
    t_anomaly: Optional[float]        # first shipped-CUSUM micro-change (s)
    t_throttle: Optional[float]       # first sustained thermal throttle (s)
    lead_time_s: Optional[float]      # t_throttle - t_anomaly
    detected: bool                    # did the detector fire before throttle?
    severity_at_throttle: float       # CUSUM severity (sigmas) at t_throttle -> calibration
    peak_rtheta: float
    ksigma_lead_s: dict = field(default_factory=dict)   # {k: lead_time_s} sweep


def analyze_run(trace: RunTrace, min_power_w: float = MIN_POWER_W,
                throttle_min_run: int = 3) -> RunResult:
    """Run the shipped CUSUM over the run's R_theta, find the throttle event, and
    compute lead time. Also sweeps a baseline+k*sigma detector for k in {2,3,4}."""
    t = np.asarray(trace.t, float)
    r = trace.rtheta(min_power_w)

    # t_throttle: first sustained thermal throttle
    t_throttle = first_sustained_true(trace.thermal_throttle, t, min_run=throttle_min_run)

    # t_anomaly: feed R_theta into the shipped CUSUM (direction +1: higher R_theta = worse).
    # ChannelCUSUM keeps its own online baseline, so we pass raw R_theta and read severity.
    cusum = ChannelCUSUM()
    t_anomaly = None
    sev_at = {}                       # t -> severity, to read severity at t_throttle
    peak = 0.0
    for ti, ri in zip(t, r):
        if np.isnan(ri):
            continue                  # gated / undefined: skip (unobserved)
        cusum.update(float(ri), +1)
        sev_at[float(ti)] = cusum.severity
        peak = max(peak, float(ri))
        if t_anomaly is None and cusum.microchange:
            t_anomaly = float(ti)

    detected = (t_anomaly is not None and t_throttle is not None
                and t_anomaly <= t_throttle)
    lead = (t_throttle - t_anomaly) if detected else None
    sev_throttle = _severity_at(sev_at, t_throttle)

    return RunResult(
        t_anomaly=t_anomaly,
        t_throttle=t_throttle,
        lead_time_s=lead,
        detected=detected,
        severity_at_throttle=sev_throttle,
        peak_rtheta=peak,
        ksigma_lead_s=_ksigma_sweep(t, r, t_throttle),
    )


def _severity_at(sev_at: dict, t_throttle: Optional[float]) -> float:
    if t_throttle is None or not sev_at:
        return 0.0
    # nearest logged sample at or before the throttle
    keys = sorted(k for k in sev_at if k <= t_throttle)
    return sev_at[keys[-1]] if keys else 0.0


def _ksigma_sweep(t: np.ndarray, r: np.ndarray, t_throttle: Optional[float],
                  warm_n: int = 60, ks=(2.0, 3.0, 4.0), sustain: int = 5) -> dict:
    """Baseline+k*sigma detector (E-LT framing): baseline mean/std over the first warm_n
    valid samples, then first time (R_theta - mean)/std > k sustained for `sustain`
    samples. Returns {k: lead_time_s} (lead vs t_throttle; None if it never fires or fires
    after throttle)."""
    valid = ~np.isnan(r)
    tv, rv = t[valid], r[valid]
    out = {k: None for k in ks}
    if rv.size < warm_n + sustain or t_throttle is None:
        return out
    base = rv[:warm_n]
    mu, sd = float(np.mean(base)), max(float(np.std(base)), 1e-9)
    z = (rv - mu) / sd
    for k in ks:
        run = 0
        for i in range(warm_n, z.size):
            if z[i] > k:
                run += 1
                if run >= sustain:
                    ta = float(tv[i - sustain + 1])
                    out[k] = (t_throttle - ta) if ta <= t_throttle else None
                    break
            else:
                run = 0
    return out


def severity_trajectory(trace: RunTrace, min_power_w: float = MIN_POWER_W):
    """Return (t_valid, severity, t_throttle): the shipped-CUSUM severity at each valid
    sample plus the throttle time. This is the basis for survival-RUL training points
    (at each sample, actual RUL = t_throttle - t)."""
    t = np.asarray(trace.t, float)
    r = trace.rtheta(min_power_w)
    cusum = ChannelCUSUM()
    ts, sev = [], []
    for ti, ri in zip(t, r):
        if np.isnan(ri):
            continue
        cusum.update(float(ri), +1)
        ts.append(float(ti))
        sev.append(cusum.severity)
    t_throttle = first_sustained_true(trace.thermal_throttle, t)
    return np.asarray(ts), np.asarray(sev), t_throttle


# ── synthetic run generator (for tests + dry-runs before hardware) ────────────
def synthetic_run(seed: int = 0, n: int = 1000, degrade_start: int = 300,
                  throttle_temp_c: float = 87.0, ambient_c: float = 25.0,
                  power_w: float = 150.0, base_rtheta: float = 0.30,
                  degrade_rate: float = 0.0007, noise_c: float = 0.4) -> RunTrace:
    """A fixed-load run whose cooling degrades from `degrade_start`: R_theta rises linearly,
    so T_j = ambient + R_theta*P climbs until it crosses the throttle temp (thermal
    slowdown asserts). Mirrors the rig's fan/TIM arm at constant power. Defaults model a
    SLOW arm (TIM/fan-duty, hundreds of samples of runway) -- for a fast acute fault, pass
    a large degrade_rate + small degrade_start (the shipped conservative CUSUM then fires
    late, which is the real E-LT fan-mode property, caught by the k-sigma sweep)."""
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=float)
    rtheta = base_rtheta + np.clip(t - degrade_start, 0, None) * degrade_rate
    tj = ambient_c + rtheta * power_w + rng.normal(0, noise_c, n)
    amb = np.full(n, ambient_c) + rng.normal(0, 0.1, n)
    pw = np.full(n, power_w) + rng.normal(0, 2.0, n)
    throttle = tj >= throttle_temp_c   # thermal slowdown asserts above the limit
    return RunTrace(t=t, t_junction=tj, t_ambient=amb, power_w=pw, thermal_throttle=throttle)
