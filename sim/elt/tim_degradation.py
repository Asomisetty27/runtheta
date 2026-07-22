"""
TIM-degradation submodel for the Tier-1 DCTM (dctm.py), grounded in published
accelerated-aging magnitudes rather than an assumed Arrhenius rate (the previous
elt_simulation's weakest, fully-guessed parameter — see
wiki/synthesis/sim_model_recommendation_2026_06_30.md and
wiki/synthesis/external_data_library_2026_06_30.md §D).

Literature anchors (TIM-aging studies, accelerated/HAST testing):
  - Thermal resistance rise under HAST aging: +7.5%, +11.5%, +14.8% (3 reported
    sample conditions) -> treated as a triangular-ish empirical distribution for
    the TERMINAL (fully-aged) TIM resistance multiplier.
  - Thermal conductivity decline is NONLINEAR: ~10.8% initial drop, total ~26.6%
    decline by 4 cycles, then stabilizes -- i.e. fast-then-plateau, not a single
    exponential and not linear. Modelled here as a saturating curve (1 - exp)
    in degradation-PROGRESS space, not directly in wall-clock time (see below).
  - Pump-out (CTE-mismatch shear squeezing paste from the interface) is reported
    as the dominant mechanism, not pure dry-out -- supports a progress variable
    driven by THERMAL CYCLING / power transients, not just elapsed time at
    temperature (a refinement over pure Arrhenius dry-out kinetics).

Because none of these sources report a wall-clock RATE for a GPU-class TIM under
GPU-class duty cycles (they're accelerated-test conditions on different TIM
formulations), the time-to-terminal-degradation is the genuinely unknown
quantity -- exactly the thing E-LT measures. This module therefore separates
two honestly-different things:
  1. MAGNITUDE (terminal R_ct multiplier): literature-grounded, real numbers.
  2. RATE (time to reach it): unknown, swept as a wide prior with explicit UQ.
This separation is the point: we are not pretending to know the rate.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

try:
    from .dctm import FosterDCTM
except ImportError:  # allow running as a standalone script (no package context)
    from dctm import FosterDCTM


# ─────────────────────────────────────────────────────────────────────────────
# Magnitude: terminal R_ct multiplier, from literature (HAST aging studies)
# ─────────────────────────────────────────────────────────────────────────────
HAST_RCT_RISE_PCT = (7.5, 11.5, 14.8)  # [LIT] reported HAST thermal-resistance rises
CONDUCTIVITY_DECLINE_PCT = 26.6        # [LIT] nonlinear, ~10.8% initial + plateau by 4 cycles


def terminal_rct_multiplier(rng: np.random.Generator) -> float:
    """Sample a literature-grounded terminal TIM-resistance multiplier.
    Triangular over the 3 reported HAST values; occasionally extrapolate beyond
    the reported range using the conductivity-decline figure as an upper anchor
    (1/(1-decline) ~ 1.36x), since real GPU TIM under years of duty cycling may
    exceed short accelerated-test conditions."""
    lo, mode, hi = HAST_RCT_RISE_PCT
    base = rng.triangular(lo, mode, hi) / 100.0
    upper_anchor = 1.0 / (1.0 - CONDUCTIVITY_DECLINE_PCT / 100.0) - 1.0  # ~0.362
    # 80% draws from the HAST triangular, 20% extrapolate toward the conductivity anchor
    if rng.random() < 0.2:
        base = rng.uniform(base, upper_anchor)
    return 1.0 + base


def degradation_progress_pumpout(t_s: np.ndarray, t_terminal_s: float,
                                  shape: float = 1.6) -> np.ndarray:
    """Saturating (NOT linear, NOT pure-exponential) progress curve x(t) in [0,1],
    reflecting the literature's fast-initial-then-plateau pump-out behaviour.
    x(t) = 1 - (1 - (t/t_terminal))^shape for t<t_terminal, clipped to [0,1].
    shape>1 front-loads degradation (matches the ~10.8%-of-26.6% initial-drop
    fraction reported, i.e. ~40% of total decline happens in the first ~25% of
    the observed cycles)."""
    frac = np.clip(t_s / t_terminal_s, 0.0, 1.0)
    return 1.0 - (1.0 - frac) ** shape


@dataclass
class TIMDegradationDraw:
    """One Monte Carlo draw: a terminal multiplier + an (unknown) time-to-terminal."""
    terminal_mult: float
    t_terminal_s: float
    shape: float

    def rct_multiplier(self, t_s: np.ndarray) -> np.ndarray:
        x = degradation_progress_pumpout(t_s, self.t_terminal_s, self.shape)
        return 1.0 + x * (self.terminal_mult - 1.0)


def sample_draw(rng: np.random.Generator,
                 t_terminal_range_s=(3600.0, 30 * 24 * 3600.0)) -> TIMDegradationDraw:
    """t_terminal swept log-uniform across 1 hour to 30 days -- an explicitly WIDE
    prior, because no source gives a GPU-relevant rate (see module docstring).
    This wide prior is what produces honest (wide) lead-time uncertainty bands,
    rather than a falsely-confident point estimate."""
    lo, hi = t_terminal_range_s
    t_terminal = float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
    shape = float(rng.uniform(1.2, 2.2))
    return TIMDegradationDraw(terminal_mult=terminal_rct_multiplier(rng),
                              t_terminal_s=t_terminal, shape=shape)


# ─────────────────────────────────────────────────────────────────────────────
# Apply to a FosterDCTM: identify which branch is "the TIM" and degrade it
# ─────────────────────────────────────────────────────────────────────────────
def apply_tim_degradation(model: FosterDCTM, draw: TIMDegradationDraw,
                          t_s: np.ndarray, tim_branch_idx: int | None = None) -> np.ndarray:
    """Return R_theta(t) for `model` with one branch's R scaled by the TIM
    degradation trajectory. Defaults to the branch with the SECOND-shortest tau
    (the 'case' branch in our identified T4 DCTM, tau~109s -- physically the
    TIM/IHS layer sits between the fast die response (~9s) and the slow
    heatsink (~800s); see calibrate_dctm.py)."""
    if tim_branch_idx is None:
        tim_branch_idx = int(np.argsort(model.tau)[len(model.tau) // 2]) if len(model.tau) > 2 else 0
    mult = draw.rct_multiplier(t_s)
    r_theta_t = np.empty_like(t_s, dtype=float)
    base_other = np.sum(model.R) - model.R[tim_branch_idx]
    for i, m in enumerate(mult):
        r_theta_t[i] = base_other + model.R[tim_branch_idx] * m
    return r_theta_t


def monte_carlo_lead_time(model: FosterDCTM, n_draws: int, k_sigma: float,
                          noise_floor_frac: float, duration_s: float,
                          dt_s: float = 60.0, seed: int = 0, sustain: int = 5) -> dict:
    """Monte Carlo over the literature-grounded magnitude + wide-prior rate to
    produce a LEAD-TIME DISTRIBUTION with honest uncertainty, rather than a
    single number. lead_time = time from anomaly-detect (R_theta crosses
    healthy_mean + k_sigma*noise, SUSTAINED for `sustain` consecutive samples)
    to terminal degradation (proxy for throttle risk, since we have no
    calibrated throttle controller here -- see
    sim_model_recommendation_2026_06_30.md Tier-1/Tier-2 split).

    sustain matches the shipped detector's convention (see detect_crossing in
    multivariate_detector.py, default sustain=5) -- NOT optional. A prior
    version of this function used single-sample crossing (`np.where(...)[0]`),
    which is silently unsound: over a 30-day/dt=60s window (43200 samples) at
    k_sigma=3, the EXPECTED number of pure-noise exceedances is
    N*P(Z>3)=43200*0.00135~58, so a single-sample crossing test "detects" a
    zero-degradation null run ~100% of the time (verified empirically) -- the
    reported "lead time" was mostly measuring noise-crossing timing, not
    degradation. See wiki/synthesis/deep_research audit, 2026-07-01."""
    rng = np.random.default_rng(seed)
    t = np.arange(0.0, duration_s, dt_s)
    healthy_r = model.r_theta
    noise = healthy_r * noise_floor_frac
    thresh = healthy_r + k_sigma * noise
    lead_times = []
    detect_fail = 0
    for _ in range(n_draws):
        draw = sample_draw(rng)
        r_t = apply_tim_degradation(model, draw, t)
        r_t_noisy = r_t + rng.normal(0, noise, size=r_t.shape)
        above = r_t_noisy > thresh
        run = 0
        t_detect = None
        for i, a in enumerate(above):
            run = run + 1 if a else 0
            if run >= sustain:
                t_detect = t[i - sustain + 1]
                break
        if t_detect is None:
            detect_fail += 1
            continue
        lead = draw.t_terminal_s - t_detect
        lead_times.append(lead)
    lt = np.array(lead_times)
    return {
        "n_draws": n_draws, "n_detected": len(lt), "n_never_detected": detect_fail,
        "frac_never_detected": detect_fail / n_draws,
        "lead_time_s": lt,
        "median_h": float(np.median(lt) / 3600) if len(lt) else None,
        "p10_h": float(np.percentile(lt, 10) / 3600) if len(lt) else None,
        "p90_h": float(np.percentile(lt, 90) / 3600) if len(lt) else None,
        "frac_negative_or_late": float(np.mean(lt < 0)) if len(lt) else None,
    }


def null_control_check(model: FosterDCTM, n_draws: int, k_sigma: float,
                       noise_floor_frac: float, duration_s: float,
                       dt_s: float = 60.0, seed: int = 1, sustain: int = 5) -> float:
    """Run the SAME detector with zero degradation injected. Returns the false-
    detection rate. This should be near k_sigma's nominal false-positive rate,
    not near 1.0 -- a high value here means the detector (or its sustain
    setting) is not actually gated on the degradation signal. Always run this
    alongside monte_carlo_lead_time and report it; a lead-time number without
    this control is not trustworthy."""
    rng = np.random.default_rng(seed)
    t = np.arange(0.0, duration_s, dt_s)
    healthy_r = model.r_theta
    noise = healthy_r * noise_floor_frac
    thresh = healthy_r + k_sigma * noise
    false_detects = 0
    for _ in range(n_draws):
        r_t_noisy = healthy_r + rng.normal(0, noise, size=t.shape)
        above = r_t_noisy > thresh
        run = 0
        for a in above:
            run = run + 1 if a else 0
            if run >= sustain:
                false_detects += 1
                break
    return false_detects / n_draws
