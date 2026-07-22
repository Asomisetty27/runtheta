"""
Workload normalization -- generalize the R_theta trick to every prognostic channel.

R_theta = dT/P is trustworthy because it expresses temperature as a RESIDUAL from what
power predicts: when workload (and therefore power) swings, temperature swings with it,
but the RATIO stays put, so only cooling degradation moves it. Every other channel Theta
tracks (memory-to-core delta, clock, ECC rate, ...) has the same problem R_theta solves:
its raw value moves with workload, so feeding the raw value into a change detector makes
the detector confuse a workload phase change with degradation. The GWDG real-data run
(2026-07-02) is exactly why this matters -- raw channels risk firing on workload, not fault.

This module makes the R_theta idea general and learned instead of closed-form: during a
healthy baseline it fits the expected channel value as a function of the operating point
(power, the dominant workload covariate -- the same covariate R_theta divides by), then
reports the STANDARDIZED RESIDUAL (observed minus expected-given-power, over baseline
residual sigma). On healthy hardware the residual is ~N(0,1) regardless of workload;
under degradation it drifts. That residual -- not the raw value -- is what should feed the
CUSUM detector, so no channel can confuse workload variation with degradation.

Binned (quantile) regression on power is used rather than linear, because the healthy
relationship is generally nonlinear (R_theta itself is a curve, 0.12->0.06 across
120-700W per E009/F8) -- a linear fit would leave workload structure in the residual.

scipy-free / numpy-only.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class WorkloadNormalizer:
    """Learns E[channel | power] on a healthy baseline; emits standardized residuals."""
    n_bins: int = 8
    warm_n: int = 60
    _raw: list[float] = field(default_factory=list)      # baseline channel values
    _pow: list[float] = field(default_factory=list)      # baseline power (operating point)
    range_margin: float = 0.05   # allow this fraction beyond the trained power range
    bin_edges: np.ndarray = field(default_factory=lambda: np.array([]))
    bin_means: np.ndarray = field(default_factory=lambda: np.array([]))
    resid_sigma: float = 1.0
    p_lo: float = 0.0
    p_hi: float = 0.0
    fitted: bool = False
    n_unseen: int = 0            # count of windows skipped as out-of-trained-range

    def _expected(self, power: float) -> float:
        """Expected healthy channel value at this operating point (power)."""
        b = int(np.clip(np.searchsorted(self.bin_edges, power, side="right") - 1,
                        0, len(self.bin_means) - 1))
        return float(self.bin_means[b])

    def fit(self) -> None:
        p = np.asarray(self._pow, float)
        r = np.asarray(self._raw, float)
        # quantile bin edges over observed power so each bin is populated
        qs = np.linspace(0, 100, self.n_bins + 1)
        self.bin_edges = np.unique(np.percentile(p, qs))
        if len(self.bin_edges) < 2:                       # power was ~constant: no conditioning needed
            self.bin_edges = np.array([p.min() - 1, p.max() + 1])
        means = []
        for i in range(len(self.bin_edges) - 1):
            m = (p >= self.bin_edges[i]) & (p <= self.bin_edges[i + 1])
            means.append(float(np.median(r[m])) if m.any() else float(np.median(r)))
        self.bin_means = np.array(means)
        resid = r - np.array([self._expected(pi) for pi in p])
        self.resid_sigma = max(float(np.std(resid)), 1e-6)
        self.p_lo, self.p_hi = float(p.min()), float(p.max())
        self.fitted = True

    def normalize(self, raw: float, power: float) -> float | None:
        """Return the standardized workload-conditioned residual, or None if we cannot
        assess this window: during warmup (no healthy baseline yet), OR when the operating
        point is outside the trained power range. Declining on an unseen operating point is
        deliberate -- we have no healthy reference there, so a residual would be fabricated
        (that was the scenario-1 false alarm). Same honesty as R_theta needing loaded
        samples; the baseline extends to new regimes only as they're observed healthy."""
        if not self.fitted:
            self._raw.append(raw)
            self._pow.append(power)
            if len(self._raw) >= self.warm_n:
                self.fit()
            return None
        span = max(self.p_hi - self.p_lo, 1e-6)
        if power < self.p_lo - self.range_margin * span or power > self.p_hi + self.range_margin * span:
            self.n_unseen += 1
            return None
        return (raw - self._expected(power)) / self.resid_sigma
