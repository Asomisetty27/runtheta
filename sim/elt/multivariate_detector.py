"""
Multivariate anomaly detector: generalizes the shipped baseline+k*sigma R_theta
detector (a 1D z-score) to a FEATURE VECTOR via Mahalanobis distance -- the
direct, classic statistical generalization of "how many sigmas from healthy"
to multiple correlated variables at once.

  z_1D   = (x - mu) / sigma                          (current shipped detector)
  D_maha = sqrt((v - mu_vec)^T * Sigma_inv * (v - mu_vec))   (this module)

Sigma_inv is the inverse covariance of the HEALTHY baseline window, so the
distance accounts for correlation between features (e.g. R_theta and power are
not independent) rather than treating each axis as if it were.

Built to answer a direct, falsifiable question: does adding more telemetry
variables (not just R_theta) to the detector actually improve lead-time /
false-positive performance versus the existing single-variable detector, on
data where we have real ground truth (the synthetic gradual/step scenarios)?
This module does NOT assume the answer -- see compare_detectors() below, which
reports both detectors' performance side by side.

numpy only (no sklearn dependency).
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class MahalanobisDetector:
    mu: np.ndarray          # feature means at baseline (raw units), shape (d,)
    scale: np.ndarray       # feature std at baseline (raw units), shape (d,)
    cov_inv: np.ndarray     # inverse covariance of STANDARDIZED baseline, shape (d,d)
    feature_names: list[str]
    n_baseline: int

    @classmethod
    def fit(cls, X_baseline: np.ndarray, feature_names: list[str], ridge: float = 1e-3):
        """X_baseline: (n_samples, d) healthy/baseline feature matrix.
        Features are STANDARDIZED before covariance (R_theta ~0.05, power ~600W,
        temp ~60C live on wildly different scales -- without this, a small ridge
        is meaningless for the small-scale dims and the inverse covariance
        blows up in those directions, causing false alarms on noise). Requires
        n_samples >> d (rule of thumb >10x) for a stable covariance estimate;
        callers should pass a generously long baseline window, not a short one."""
        d = X_baseline.shape[1]
        if X_baseline.shape[0] < 10 * d:
            raise ValueError(f"baseline n={X_baseline.shape[0]} too small for d={d} "
                             f"features (need >= {10*d}); Mahalanobis cov will be unstable")
        mu = X_baseline.mean(axis=0)
        scale = X_baseline.std(axis=0)
        scale = np.where(scale < 1e-9, 1.0, scale)
        Xs = (X_baseline - mu) / scale
        cov = np.atleast_2d(np.cov(Xs, rowvar=False))
        cov += ridge * np.eye(d)   # ridge now meaningful: standardized dims are all O(1)
        return cls(mu=mu, scale=scale, cov_inv=np.linalg.inv(cov),
                  feature_names=feature_names, n_baseline=X_baseline.shape[0])

    def distance(self, X: np.ndarray) -> np.ndarray:
        Xs = (X - self.mu) / self.scale
        return np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", Xs, self.cov_inv, Xs), 0.0))


def build_feature_matrix(t: np.ndarray, r_theta: np.ndarray, temp: np.ndarray,
                         power: np.ndarray, util: np.ndarray,
                         peer_r_theta_median: np.ndarray,
                         roll_window: int = 5) -> tuple[np.ndarray, list[str]]:
    """Engineer the feature vector per timestamp for ONE gpu:
      [R_theta, temp, power, util, dR_theta/dt, rolling_std(R_theta), peer_z]
    peer_z = (this GPU's R_theta - fleet median R_theta at same t) -- requires
    peer_r_theta_median precomputed per-timestamp across the fleet.
    All using only data already available in our datasets; no synthetic inputs."""
    n = len(t)
    # BACKWARD difference only -- np.gradient's central-difference interior points
    # use r_theta[i+1], a one-sample lookahead that isn't available in real-time
    # detection (the whole point of this feature is latency). Verified present in
    # a prior version; fixed 2026-07-01 per adversarial audit.
    drdt = np.empty(n)
    drdt[0] = 0.0
    dt = np.diff(t)
    dt = np.where(dt == 0, 1e-9, dt)
    drdt[1:] = np.diff(r_theta) / dt
    roll_std = np.array([
        np.std(r_theta[max(0, i - roll_window):i + 1]) for i in range(n)
    ])
    peer_z = r_theta - peer_r_theta_median
    X = np.column_stack([r_theta, temp, power, util, drdt, roll_std, peer_z])
    names = ["r_theta", "temp", "power", "util", "drdt", "roll_std", "peer_z"]
    return X, names


def detect_crossing(score: np.ndarray, t: np.ndarray, threshold: float,
                    sustain: int = 3) -> float | None:
    run = 0
    for i, s in enumerate(score):
        run = run + 1 if s > threshold else 0
        if run >= sustain:
            return float(t[i - sustain + 1])
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Supervised (trained) alternative: learn feature WEIGHTS from labels, rather
# than equal-weighting via raw covariance (which the unsupervised Mahalanobis
# test above showed underperforms on real ground truth -- noisy derived
# features get equal say to the core signal). This is what "trained model"
# should mean: weights fit to minimize error against real labeled outcomes.
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class LogisticDetector:
    w: np.ndarray       # learned weights, shape (d,)
    b: float            # bias
    mu: np.ndarray       # standardization mean (fit on TRAIN split only)
    scale: np.ndarray    # standardization std
    feature_names: list[str]

    @classmethod
    def fit(cls, X: np.ndarray, y: np.ndarray, feature_names: list[str],
            lr: float = 0.1, l2: float = 1e-3, epochs: int = 4000, seed: int = 0,
            class_weight: bool = True):
        """y: binary labels (1 = degraded state, per ground truth). Plain numpy
        gradient descent (no sklearn dependency). Features standardized using
        ONLY this call's X (caller must pass a TRAIN split, not the full series,
        to avoid leaking test-period statistics into the fit).
        class_weight: degradation events are rare (few % positive) -- without
        reweighting, unweighted logistic regression on imbalanced data collapses
        to predicting the majority class (a textbook failure mode, not a model
        limitation). Weight positives by n_neg/n_pos so the gradient doesn't
        average the rare class away."""
        rng = np.random.default_rng(seed)
        mu = X.mean(axis=0)
        scale = X.std(axis=0)
        scale = np.where(scale < 1e-9, 1.0, scale)
        Xs = (X - mu) / scale
        n, d = Xs.shape
        n_pos = max(y.sum(), 1.0)
        n_neg = max(n - y.sum(), 1.0)
        sw = np.where(y > 0.5, n_neg / n_pos, 1.0) if class_weight else np.ones(n)
        sw = sw / sw.mean()  # keep effective learning rate stable
        w = rng.normal(0, 0.01, d)
        b = 0.0
        for _ in range(epochs):
            z = Xs @ w + b
            p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
            grad_w = Xs.T @ (sw * (p - y)) / n + l2 * w
            grad_b = np.mean(sw * (p - y))
            w -= lr * grad_w
            b -= lr * grad_b
        return cls(w=w, b=b, mu=mu, scale=scale, feature_names=feature_names)

    def score(self, X: np.ndarray) -> np.ndarray:
        Xs = (X - self.mu) / self.scale
        z = Xs @ self.w + self.b
        return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
