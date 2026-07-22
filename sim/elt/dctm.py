"""
Grey-box Dynamic Compact Thermal Model (DCTM).

A Foster-network compact thermal model whose parameters are IDENTIFIED from real
telemetry, rather than assumed (cf. params.py, which calibrates a physical Cauer
ladder to a single Stage-1 operating point). This is the Tier-1 engine from the
simulation-model recommendation: a DELPHI-style boundary-condition-independent
dynamic compact model, calibrated by grey-box system identification.

Foster network (behavioural, not physical-ladder):

    T_j(t) - T_amb = sum_i  T_i(t),   dT_i/dt = (P(t)*R_i - T_i) / tau_i

  - n parallel RC branches, each (R_i [C/W], tau_i [s]); tau_i = R_i * C_i.
  - Steady state: T_j - T_amb = P * sum_i R_i = P * R_total  ->  R_theta = sum R_i.
  - The branch time constants tau_i are the multi-timescale thermal response
    (fast die, slow heatsink); they are what a single-RC model cannot capture.

Identification (grey-box): fit a sum-of-exponentials to a measured power-step
RESPONSE (load->idle recovery is cleanest). For a step of magnitude dP at t=0,
the cooling curve is  T(t) = T_inf + sum_i (dP*R_i) * exp(-t/tau_i).  With taus
fixed the amplitudes are linear (numpy lstsq); we grid-search the taus. Model
ORDER (1 vs 2 vs 3 branches) is chosen by the BIC of the fit, which directly
answers the single- vs multi-time-constant question.

scipy-free (numpy only) so it runs without the sim venv.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass
class FosterDCTM:
    """A Foster-network compact thermal model, identified from telemetry."""

    R: np.ndarray          # branch resistances [C/W], shape (n,)
    tau: np.ndarray        # branch time constants [s], shape (n,)
    t_amb_c: float = 25.0  # ambient used during identification
    label: str = "dctm"

    @property
    def r_theta(self) -> float:
        """Steady-state effective thermal resistance R_theta = sum R_i [C/W]."""
        return float(np.sum(self.R))

    def scale_to_rtheta(self, r_theta_target: float) -> "FosterDCTM":
        """Anchor total R_theta to a measured per-GPU value (e.g. A100 0.038),
        keeping the time-constant STRUCTURE identified from finer-grained data.
        Capacitances are held physical (C_i = tau_i/R_i), so scaling R rescales
        tau proportionally only if C is held; here we keep tau (dynamics) fixed
        and rescale R (steady gain) — the BCI-DCTM separation of gain vs dynamics."""
        s = r_theta_target / self.r_theta
        return FosterDCTM(R=self.R * s, tau=self.tau.copy(),
                          t_amb_c=self.t_amb_c, label=f"{self.label}_R{r_theta_target:.3f}")

    def simulate(self, t_s: np.ndarray, power_w: np.ndarray,
                 t_amb_c: np.ndarray | float, tj0_c: float | None = None) -> np.ndarray:
        """Integrate the Foster network for an arbitrary power/ambient series.
        Exact per-step exponential integrator (stable for any step size)."""
        t_s = np.asarray(t_s, float)
        power_w = np.asarray(power_w, float)
        amb = np.full_like(t_s, t_amb_c, float) if np.isscalar(t_amb_c) else np.asarray(t_amb_c, float)
        # initial branch temps: steady at first power if tj0 not given
        Ti = self.R * power_w[0]
        if tj0_c is not None:  # distribute an observed initial offset across branches by R-weight
            off = (tj0_c - amb[0]) - Ti.sum()
            Ti = Ti + off * (self.R / self.R.sum())
        out = np.empty_like(t_s)
        out[0] = amb[0] + Ti.sum()
        for k in range(1, len(t_s)):
            dt = t_s[k] - t_s[k - 1]
            P = power_w[k]
            decay = np.exp(-dt / self.tau)              # per-branch
            Ti = Ti * decay + (self.R * P) * (1.0 - decay)  # exact for piecewise-const P
            out[k] = amb[k] + Ti.sum()
        return out


# ─────────────────────────────────────────────────────────────────────────────
# Grey-box identification from a measured step (cooling) response
# ─────────────────────────────────────────────────────────────────────────────
def _fit_fixed_taus(t: np.ndarray, y: np.ndarray, taus: Sequence[float]):
    """Linear LSQ for [offset, A_1..A_n] given fixed taus. Returns (offset, amps, rmse)."""
    X = np.column_stack([np.ones_like(t)] + [np.exp(-t / tau) for tau in taus])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    rmse = float(np.sqrt(np.mean(resid ** 2)))
    return coef[0], coef[1:], rmse


def fit_step_response(t: np.ndarray, y: np.ndarray, n_branches: int,
                      tau_grid: np.ndarray | None = None):
    """Fit y(t) = offset + sum_i A_i*exp(-t/tau_i) to a cooling curve.
    Grid-search taus (greedy add), linear amplitudes. Returns dict with taus, amps,
    offset (asymptote), rmse, bic, k (free params)."""
    t = np.asarray(t, float); y = np.asarray(y, float)
    if tau_grid is None:
        tau_grid = np.geomspace(2.0, 800.0, 40)
    chosen: list[float] = []
    for _ in range(n_branches):
        best = None
        for tau in tau_grid:
            if any(abs(np.log(tau / c)) < 0.15 for c in chosen):  # avoid near-duplicates
                continue
            off, amps, rmse = _fit_fixed_taus(t, y, chosen + [tau])
            if (amps > 0).all() and (best is None or rmse < best[0]):
                best = (rmse, tau)
        if best is None:
            break
        chosen.append(best[1])
    off, amps, rmse = _fit_fixed_taus(t, y, chosen)
    n = len(t); k = 2 * len(chosen) + 1                      # taus + amps + offset
    bic = n * np.log(max(rmse, 1e-9) ** 2) + k * np.log(n)   # Gaussian BIC
    order = np.argsort(chosen)
    taus_sorted = np.array(chosen)[order]
    # A tau within 5% of the grid's own ceiling is not a measured timescale -- it is
    # the fit pinned against the search boundary (verified empirically: widening the
    # ceiling from 800s to 3000s to 10000s moves this branch to match every time,
    # while genuine branches stay put). Flag rather than silently report it as real.
    grid_ceiling = float(tau_grid[-1])
    at_edge = [bool(tv >= 0.95 * grid_ceiling) for tv in taus_sorted]
    return {"taus": taus_sorted, "amps": np.array(amps)[order],
            "offset": float(off), "rmse": rmse, "bic": float(bic), "n": n,
            "tau_at_grid_edge": at_edge}


def identify_from_recovery(t: np.ndarray, temp_c: np.ndarray, delta_power_w: float,
                           max_branches: int = 3, t_amb_c: float = 25.0):
    """Identify a FosterDCTM from a load->idle recovery (cooling) transient.
    Picks model order by BIC. R_i = A_i / delta_power_w, C_i = tau_i/R_i implicit.
    Returns (best_model: FosterDCTM, report: dict per order)."""
    t = np.asarray(t, float) - float(np.asarray(t, float)[0])
    report = {}
    for nb in range(1, max_branches + 1):
        fit = fit_step_response(t, temp_c, nb)
        report[nb] = fit
    # choose order with lowest BIC
    best_n = min(report, key=lambda k: report[k]["bic"])
    f = report[best_n]
    R = f["amps"] / delta_power_w
    model = FosterDCTM(R=R, tau=f["taus"], t_amb_c=t_amb_c, label=f"T4_recovery_{best_n}branch")
    return model, report, best_n
