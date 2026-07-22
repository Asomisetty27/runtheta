"""
Leave-one-scenario-out cross-validation: trained multivariate detector vs.
R_theta-only baseline, across MANY generated scenarios (scenario_generator.py)
instead of the 2 hand-built ones that bracketed both class-imbalance failure
modes in multivariate_detector_test_2026_06_30.md.

Methodology (designed to not p-hack):
  For each test scenario:
    - TRAIN the logistic detector's weights on all OTHER scenarios pooled.
    - CALIBRATE the decision threshold via precision-recall on a separate
      VALIDATION scenario (also excluded from test) -- never on the test
      scenario itself.
    - EVALUATE on the held-out test scenario only.
  Repeat for every scenario as the test fold; report the aggregate distribution
  of detection latency and false-positive rate, not a single cherry-picked run.
"""
from __future__ import annotations
import sys
import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from scenario_generator import generate_batch, Scenario          # noqa: E402
from multivariate_detector import (build_feature_matrix, LogisticDetector,  # noqa: E402
                                   detect_crossing)


def scenario_to_features(s: Scenario):
    """Build (X, y, per-gpu series) for every GPU in one scenario."""
    rth = (s.temp - s.config.base_ambient_c) / s.power_w
    n_gpus = s.config.fleet_size
    peer_med = np.median(rth, axis=0)
    out = {}
    for g in range(n_gpus):
        X, names = build_feature_matrix(s.t, rth[g], s.temp[g], s.power_w[g], s.duty[g], peer_med)
        y = (s.r_theta_mult[g] > 1.01).astype(float)
        out[g] = (X, y)
    return out, names


def best_threshold_pr(scores: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Pick the threshold maximizing F1 on a VALIDATION scenario (not test)."""
    cand = np.unique(scores)[::-1]
    best_f1, best_t = 0.0, 0.5
    for thr in cand[::max(1, len(cand) // 200)]:  # subsample candidate thresholds for speed
        pred = (scores >= thr).astype(float)
        tp = np.sum((pred == 1) & (y == 1))
        fp = np.sum((pred == 1) & (y == 0))
        fn = np.sum((pred == 0) & (y == 1))
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        if f1 > best_f1:
            best_f1, best_t = f1, float(thr)
    return best_t, best_f1


def uni_baseline_eval(s: Scenario, k: float = 3.0, sustain: int = 5):
    rth = (s.temp - s.config.base_ambient_c) / s.power_w
    g_bad = s.degraded_gpu
    n_base = int(np.searchsorted(s.t, s.onset_t)) - 5
    n_base = max(20, min(n_base, len(s.t) - 10))
    fp = 0
    lead = None
    for g in range(s.config.fleet_size):
        base = rth[g][:n_base]
        thr = base.mean() + k * base.std()
        t_det = detect_crossing(rth[g], s.t, thr, sustain=sustain)
        if g == g_bad:
            lead = (s.onset_t - t_det) if t_det else None
        elif t_det is not None:
            fp += 1
    return lead, fp, s.config.fleet_size - 1


def run_cv(n_scenarios: int = 24, master_seed: int = 7):
    batch = generate_batch(n_scenarios, master_seed=master_seed)
    feats = [scenario_to_features(s) for s in batch]

    uni_leads, uni_fps, uni_fp_denoms = [], [], []
    multi_leads, multi_fps, multi_fp_denoms = [], [], []
    n_skipped = 0

    for i in range(n_scenarios):
        test_s = batch[i]
        # validation scenario: the next index (wrap-around), excluded from both train and test
        val_idx = (i + 1) % n_scenarios
        train_idx = [j for j in range(n_scenarios) if j not in (i, val_idx)]
        if len(train_idx) < 3:
            n_skipped += 1
            continue

        Xtr, ytr = [], []
        for j in train_idx:
            fdict, names = feats[j]
            for g, (X, y) in fdict.items():
                Xtr.append(X)
                ytr.append(y)
        Xtr = np.vstack(Xtr)
        ytr = np.concatenate(ytr)
        if ytr.sum() < 5:  # degenerate fold, skip
            n_skipped += 1
            continue
        clf = LogisticDetector.fit(Xtr, ytr, names, lr=0.3, l2=1e-3, epochs=1500, class_weight=True)

        # calibrate threshold on the VALIDATION scenario only
        val_fdict, _ = feats[val_idx]
        val_scores, val_y = [], []
        for g, (X, y) in val_fdict.items():
            val_scores.append(clf.score(X))
            val_y.append(y)
        thr, _ = best_threshold_pr(np.concatenate(val_scores), np.concatenate(val_y))

        # evaluate on the TEST scenario
        test_fdict, _ = feats[i]
        g_bad = test_s.degraded_gpu
        fp = 0
        lead = None
        for g, (X, y) in test_fdict.items():
            score = clf.score(X)
            t_det = detect_crossing(score, test_s.t, thr, sustain=5)
            if g == g_bad:
                lead = (test_s.onset_t - t_det) if t_det else None
            elif t_det is not None:
                fp += 1
        multi_leads.append(lead)
        multi_fps.append(fp)
        multi_fp_denoms.append(test_s.config.fleet_size - 1)

        ul, ufp, ufp_d = uni_baseline_eval(test_s)
        uni_leads.append(ul)
        uni_fps.append(ufp)
        uni_fp_denoms.append(ufp_d)

    def summarize(leads, fps, denoms, label):
        det = [x for x in leads if x is not None]
        miss = sum(1 for x in leads if x is None)
        fp_rate = sum(fps) / max(sum(denoms), 1)
        print(f"{label}: n={len(leads)} detected={len(det)} missed={miss} "
              f"fp_rate={fp_rate:.3f} ({sum(fps)}/{sum(denoms)})")
        if det:
            det_h = np.array(det) / 3600
            print(f"  lead-time (h): median={np.median(det_h):.1f} "
                  f"p10={np.percentile(det_h,10):.1f} p90={np.percentile(det_h,90):.1f} "
                  f"frac_negative(late)={np.mean(np.array(det)<0):.2f}")

    print(f"\n=== Leave-one-scenario-out CV, n_scenarios={n_scenarios}, skipped={n_skipped} ===\n")
    summarize(uni_leads, uni_fps, uni_fp_denoms, "R_theta-only (baseline)")
    print()
    summarize(multi_leads, multi_fps, multi_fp_denoms, "Multivariate (trained, threshold calibrated on held-out val scenario)")


if __name__ == "__main__":
    run_cv()
