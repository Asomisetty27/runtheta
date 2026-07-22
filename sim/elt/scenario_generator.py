"""
Synthetic multi-GPU fleet scenario generator -- produces MANY labeled
degradation scenarios (varied mode, rate, severity, fleet size, noise) so a
trained detector can be properly cross-validated, rather than tuned against
the 2 hand-built scenarios that bracketed both class-imbalance failure modes
in multivariate_detector_test_2026_06_30.md.

Deliberately reuses real, already-grounded pieces rather than inventing new
assumptions:
  - Severity: tim_degradation.terminal_rct_multiplier (literature HAST magnitudes)
  - Progress shape: tim_degradation.degradation_progress_pumpout (front-loaded,
    matches the reported nonlinear conductivity decline)
  - Noise: non-stationary multiplicative workload variance (the real behavior
    found by inspecting the original synthetic data while debugging the
    Mahalanobis detector -- a fixed-variance noise model was part of why that
    detector's baseline mis-calibrated)
  - Fleet heterogeneity: small per-GPU baseline R_theta offsets (position/bin
    effects, motivated by F7/F10's node/position structure in the real E009 data)

Output schema matches the existing E-LT synthetic scenario files
(timestamp, node, gpu_ordinal, nvidia_gpu_temperature_celsius,
nvidia_gpu_power_usage_milliwatts, nvidia_gpu_duty_cycle) plus a groundtruth
per-gpu multiplier series, so it is a drop-in extension usable by both the
existing univariate detector code and multivariate_detector.py.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

try:
    from .tim_degradation import terminal_rct_multiplier, degradation_progress_pumpout
except ImportError:
    from tim_degradation import terminal_rct_multiplier, degradation_progress_pumpout


@dataclass
class ScenarioConfig:
    seed: int
    fleet_size: int                 # 4-16
    duration_days: float             # 7-30
    cadence_s: float = 300.0          # 5 min, matches GWDG-style cadence
    mode: str = "gradual"            # "gradual" | "step" | "intermittent"
    t_onset_frac: float = 0.5        # degradation begins at this fraction of duration
    t_terminal_frac: float = 1.0     # reaches terminal severity by this fraction (gradual/intermittent)
    base_power_w: float = 600.0
    base_ambient_c: float = 25.0
    healthy_r_theta_mean: float = 0.06   # H100-like default; vary per call
    healthy_r_theta_cv_floor: float = 0.10   # baseline noise floor (per-sample)
    workload_variance_regime_h: float = 36.0  # hours per noise-variance regime block
    fleet_heterogeneity_pct: float = 8.0      # per-GPU baseline R_theta spread (position/bin)


@dataclass
class Scenario:
    config: ScenarioConfig
    degraded_gpu: int
    t: np.ndarray                     # (n,) seconds
    temp: np.ndarray                  # (n_gpus, n) deg C
    power_w: np.ndarray               # (n_gpus, n) W
    duty: np.ndarray                  # (n_gpus, n) pct
    r_theta_mult: np.ndarray          # (n_gpus, n) ground-truth degradation multiplier (1.0 = healthy)
    onset_t: float                    # seconds, ground-truth onset (mult crosses 1.01)


def _nonstationary_noise(rng: np.random.Generator, n: int, dt_s: float,
                         regime_h: float, lo: float, hi: float) -> np.ndarray:
    """Piecewise-constant noise-scale regime, resampled every `regime_h` hours,
    so variance itself drifts over the series (the real behavior found in the
    original data -- a single fixed noise level is not realistic)."""
    block = max(1, int(regime_h * 3600 / dt_s))
    n_blocks = n // block + 1
    levels = rng.uniform(lo, hi, n_blocks)
    return np.repeat(levels, block)[:n]


def generate(cfg: ScenarioConfig) -> Scenario:
    rng = np.random.default_rng(cfg.seed)
    n = int(cfg.duration_days * 86400 / cfg.cadence_s)
    t = np.arange(n) * cfg.cadence_s

    # per-GPU baseline R_theta heterogeneity (position/bin-like spread)
    het = rng.normal(0, cfg.fleet_heterogeneity_pct / 100.0, cfg.fleet_size)
    healthy_r = cfg.healthy_r_theta_mean * (1.0 + het)

    degraded_gpu = int(rng.integers(0, cfg.fleet_size))
    onset_t = cfg.t_onset_frac * t[-1]
    terminal_t = cfg.t_terminal_frac * t[-1]

    terminal_mult = terminal_rct_multiplier(rng)
    mult = np.ones((cfg.fleet_size, n))
    if cfg.mode == "step":
        prog = (t >= onset_t).astype(float)
    elif cfg.mode == "intermittent":
        # degrades, partially recovers (e.g. transient airflow clearing), degrades again
        base_prog = degradation_progress_pumpout(np.maximum(t - onset_t, 0),
                                                  max(terminal_t - onset_t, 1.0))
        ripple = 0.15 * np.sin(2 * np.pi * (t - onset_t) / (3 * 86400))
        prog = np.clip(base_prog + np.where(t >= onset_t, ripple, 0.0), 0.0, 1.0)
    else:  # gradual (default)
        prog = degradation_progress_pumpout(np.maximum(t - onset_t, 0),
                                            max(terminal_t - onset_t, 1.0))
    mult[degraded_gpu] = 1.0 + prog * (terminal_mult - 1.0)

    # non-stationary operational noise (per-GPU, independent regimes)
    power = np.empty((cfg.fleet_size, n))
    temp = np.empty((cfg.fleet_size, n))
    duty = np.empty((cfg.fleet_size, n))
    for g in range(cfg.fleet_size):
        noise_scale = _nonstationary_noise(rng, n, cfg.cadence_s,
                                           cfg.workload_variance_regime_h, 0.04, 0.18)
        p = cfg.base_power_w * (1.0 + rng.normal(0, 1, n) * noise_scale)
        p = np.clip(p, 0.2 * cfg.base_power_w, 1.15 * cfg.base_power_w)
        d = np.clip(80 + 15 * (p / cfg.base_power_w - 1.0) + rng.normal(0, 3, n), 0, 100)
        r_t = healthy_r[g] * mult[g] * (1.0 + rng.normal(0, cfg.healthy_r_theta_cv_floor, n) * 0.3)
        temp_g = cfg.base_ambient_c + r_t * p
        power[g] = p
        duty[g] = d
        temp[g] = temp_g
    return Scenario(config=cfg, degraded_gpu=degraded_gpu, t=t, temp=temp, power_w=power,
                    duty=duty, r_theta_mult=mult, onset_t=onset_t)


def random_config(rng: np.random.Generator, idx: int) -> ScenarioConfig:
    """Sample a randomized config spanning the diversity we actually need:
    mode, rate, severity (via tim_degradation), fleet size, GPU-type R_theta regime."""
    mode = rng.choice(["gradual", "step", "intermittent"], p=[0.5, 0.3, 0.2])
    fleet_size = int(rng.choice([4, 6, 8, 12, 16]))
    duration = float(rng.uniform(7, 30))
    onset_frac = float(rng.uniform(0.15, 0.6))
    terminal_frac = min(1.0, onset_frac + float(rng.uniform(0.1, 0.6)))
    r_theta_regime = float(rng.choice([0.66, 0.169, 0.06, 0.038]))  # T4/V100/H100/A100, this session's F8/F11/F13
    return ScenarioConfig(seed=1000 + idx, fleet_size=fleet_size, duration_days=duration,
                          mode=mode, t_onset_frac=onset_frac, t_terminal_frac=terminal_frac,
                          healthy_r_theta_mean=r_theta_regime)


def generate_batch(n_scenarios: int, master_seed: int = 0) -> list[Scenario]:
    rng = np.random.default_rng(master_seed)
    return [generate(random_config(rng, i)) for i in range(n_scenarios)]
