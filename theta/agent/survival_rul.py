"""
Survival-analysis RUL model -- replaces prognostic.py's linear health-slope-to-zero
extrapolation, which C-MAPSS validation (2026-07-02) showed fails on real nonlinear
degradation (relative RUL error 300-900%, worsening toward failure).

Approach: a data-driven, nonparametric survival regression on the degradation SEVERITY
signal. Instead of extrapolating a slope (which assumes linear decline and overshoots
accelerating degradation), it learns from real run-to-failure trajectories the empirical
relationship "when severity was at level s, how many cycles/seconds did units actually
have left?" -- then predicts the held-out unit's RUL by that learned curve. This is the
standard degradation-based / similarity-based prognostic; it captures nonlinearity for
free because it fits the observed shape rather than assuming one.

This is the PHM-literature answer the deep-research pass identified (Cox PH / Random
Survival Forests) reduced to its numpy-only essence for our single dominant degradation
covariate (severity). Two survival essentials are kept:
  - CENSORING: units still alive at last observation contribute a LOWER BOUND on RUL at
    their severity (they lasted at least this long), not a data point to be averaged as
    if failed -- ignoring this is exactly the mistake the naive multivariate attempt made.
  - a monotone constraint (RUL non-increasing in severity) via pool-adjacent-violators,
    so the learned curve can't imply "more degraded = more life left".

scipy-free / sklearn-free (numpy only).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class SeverityRULModel:
    """Learned severity -> remaining-useful-life curve, from run-to-failure trajectories."""
    sev_grid: np.ndarray = field(default_factory=lambda: np.array([]))
    rul_grid: np.ndarray = field(default_factory=lambda: np.array([]))  # monotone non-increasing
    n_failures: int = 0
    n_censored: int = 0
    k: int = 80          # neighbourhood size in severity for the local RUL estimate

    @property
    def fitted(self) -> bool:
        return self.sev_grid.size > 0

    def fit(self, failed: list[tuple[float, float]],
            censored: list[tuple[float, float]] | None = None) -> "SeverityRULModel":
        """failed:   (severity, true_RUL) pairs from units that ran to failure.
        censored: (severity, RUL_lower_bound) pairs from units still alive -- the unit
                  had AT LEAST this much life left at that severity. Incorporated as a
                  floor so the curve isn't biased short by survivors (Kaplan-Meier spirit).
        Builds a monotone-decreasing severity->RUL curve by local (k-NN in severity)
        median of failed points, floored by censored lower bounds, then isotonized."""
        censored = censored or []
        self.n_failures, self.n_censored = len(failed), len(censored)
        if len(failed) < 10:
            return self  # not enough to learn a curve
        f = np.array(failed, float)
        fs, fr = f[:, 0], f[:, 1]
        c = np.array(censored, float) if censored else np.empty((0, 2))
        # evaluation grid across the observed severity range
        lo, hi = float(fs.min()), float(fs.max())
        grid = np.linspace(lo, hi, 60)
        rul = np.empty_like(grid)
        for i, s in enumerate(grid):
            idx = np.argsort(np.abs(fs - s))[: self.k]
            est = float(np.median(fr[idx]))
            if c.size:  # a survivor near this severity had >= its bound left -> floor the estimate
                near = c[np.abs(c[:, 0] - s) <= (hi - lo) / 20 + 1e-9]
                if near.size:
                    est = max(est, float(np.median(near[:, 1])))
            rul[i] = est
        # enforce RUL non-increasing in severity (pool-adjacent-violators, decreasing)
        self.sev_grid = grid
        self.rul_grid = _isotonic_decreasing(rul)
        return self

    def predict(self, severity: float) -> float:
        """Predicted RUL at the current severity (same units as training RUL)."""
        if not self.fitted:
            return float("nan")
        return float(np.interp(severity, self.sev_grid, self.rul_grid[::-1][::-1]))


def _isotonic_decreasing(y: np.ndarray) -> np.ndarray:
    """Pool-adjacent-violators for a NON-INCREASING fit (RUL must not rise with severity).
    Isotonic-increasing PAVA on the reversed series, reversed back."""
    y = y.astype(float).copy()
    n = len(y)
    # fit non-decreasing on reversed array, then reverse -> non-increasing on original
    r = y[::-1].copy()
    # standard PAVA (non-decreasing)
    lvls = [[r[0], 1.0]]
    for j in range(1, n):
        lvls.append([r[j], 1.0])
        while len(lvls) > 1 and lvls[-2][0] > lvls[-1][0]:
            v2, w2 = lvls.pop()
            v1, w1 = lvls.pop()
            lvls.append([(v1 * w1 + v2 * w2) / (w1 + w2), w1 + w2])
    out = []
    for v, w_ in lvls:
        out.extend([v] * int(w_))
    return np.array(out[::-1])
