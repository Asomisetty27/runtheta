"""
Calibrate the grey-box DCTM (dctm.py) against REAL T4 telemetry, validate on
held-out trials, and anchor steady-state R_theta to each measured GPU type.

Run:  python3 calibrate_dctm.py
Reads the Stage-1 CSV directly (numpy only; no scipy / no sim venv needed).
"""
from __future__ import annotations
import csv
import sys
from collections import defaultdict
import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from dctm import identify_from_recovery, _fit_fixed_taus  # noqa: E402

CSV = "/Users/amogh/thermalos-vault/raw/experiments/ThermalOS_Measurements_Raw.csv"
# measured steady R_theta per GPU type (this session's findings)
R_THETA = {"T4_load": 0.66, "H100": 0.060, "A100": 0.038}  # F8/F11/E009


def load_trials():
    rows = list(csv.DictReader(open(CSV)))
    byphase = defaultdict(list)
    for r in rows:
        byphase[r["phase"]].append(r)
    trials = {}
    for n in range(1, 8):
        loadp = byphase.get(f"e004_t{n}_separate_process_load", [])
        recp = byphase.get(f"e004_t{n}_recovery_after_child_exit", [])
        if len(recp) < 40 or not loadp:
            continue
        recp = sorted(recp, key=lambda r: int(r["trial_second"]))
        t = np.array([int(r["trial_second"]) for r in recp], float)
        T = np.array([float(r["temp_c"]) for r in recp], float)
        p_load = np.mean([float(r["power_w"]) for r in loadp])
        p_idle = np.mean([float(r["power_w"]) for r in recp[-10:]])
        trials[n] = {"t": t - t[0], "T": T, "dP": p_load - p_idle}
    return trials


def main():
    trials = load_trials()
    print(f"loaded {len(trials)} T4 E004 recovery trials: {sorted(trials)}")
    # FIT on trial 3 (longest clean window); VALIDATE on the rest
    fit_n = 3 if 3 in trials else sorted(trials)[0]
    ft = trials[fit_n]
    print(f"\n=== IDENTIFY DCTM on trial {fit_n} (dP={ft['dP']:.1f} W) ===")
    model, report, best = identify_from_recovery(ft["t"], ft["T"], ft["dP"])
    print(f"{'order':>5} {'taus (s)':<28} {'RMSE (C)':>9} {'BIC':>9}")
    for nb, f in report.items():
        edge = f.get("tau_at_grid_edge", [False] * len(f["taus"]))
        taus = ", ".join(f"{x:.0f}{'*' if edge[j] else ''}" for j, x in enumerate(f["taus"]))
        mark = " <-- BIC-selected" if nb == best else ""
        print(f"{nb:>5} {taus:<28} {f['rmse']:>9.3f} {f['bic']:>9.1f}{mark}")
    edge_flags = report[best].get("tau_at_grid_edge", [])
    n_edge = sum(edge_flags)
    n_real = len(edge_flags) - n_edge
    print(f"\n(* = tau within 5% of the search-grid ceiling -- NOT a measured timescale, it is"
          f" pinned to the grid boundary; verified by widening the grid and watching it track"
          f" the ceiling instead of converging. {n_edge}/{len(edge_flags)} branch(es) flagged.)")
    print(f"SINGLE vs MULTI time-constant: BIC selects {best} branches, but only {n_real} are"
          f" identifiable from this {ft['t'][-1]:.0f}s window -> "
          f"{'MULTI-timescale confirmed (>=2 real branches)' if n_real > 1 else 'not established beyond 1 branch'}")
    print(f"identified branches: R={np.round(model.R,4)} C/W, tau={np.round(model.tau,1)} s"
          f" (report tau ONLY for non-* branches as measured T4 properties)")
    print(f"DCTM R_theta (sum R_i) = {model.r_theta:.4f} C/W")

    # VALIDATE: do the IDENTIFIED TIME CONSTANTS transfer to held-out trials?
    # (amplitudes must be refit per trial: load phase is only 62s, shorter than
    # the 109s/800s branches, so the GPU never reaches load-steady-state and the
    # true branch amplitudes at t=0 differ trial-to-trial. Holding tau fixed and
    # refitting only the linear amplitudes isolates the real transferability
    # question: are the TIME CONSTANTS a property of the GPU package, reusable
    # across trials, even though the starting thermal state varies?)
    print("\n=== VALIDATE: do identified time-constants (tau) transfer to held-out trials? ===")
    print(f"{'trial':>5} {'dP(W)':>6} {'n':>4} {'refit-amp RMSE (C)':>19} {'well-posed?':>12}")
    rmses_ok, rmses_bad = [], []
    for n, tr in sorted(trials.items()):
        if n == fit_n:
            continue
        off, amps, rmse = _fit_fixed_taus(tr["t"], tr["T"], model.tau)
        ok = (amps > 0).all()
        (rmses_ok if ok else rmses_bad).append(rmse)
        print(f"{n:>5} {tr['dP']:>6.1f} {len(tr['t']):>4} {rmse:>19.3f} {'yes' if ok else 'NO (amp<0)':>12}")
    n_total = len(rmses_ok) + len(rmses_bad)
    print(f"\n{len(rmses_bad)}/{n_total} held-out trials required a NON-PHYSICAL (negative-"
          f"amplitude) fit -- their RMSE does not support the tau values, it shows only that")
    print("4 free linear coefficients (offset+3 amps) can fit almost any smooth cooling curve.")
    print(f"well-posed trials only: mean RMSE = "
          f"{np.mean(rmses_ok) if rmses_ok else float('nan'):.3f} C (n={len(rmses_ok)})")
    print(f"all trials pooled (previously reported, misleading): mean RMSE = "
          f"{np.mean(rmses_ok + rmses_bad):.3f} C (T4 sensor 1C-quantised -> ~0.5C floor)")
    print("Honest interpretation: reaching the noise floor on well-posed trials is consistent")
    print("with the real branches being reusable, but has NO discriminative power once amp<0")
    print("fits are included -- an unconstrained fit reaches the floor regardless of tau accuracy.")

    # ANCHOR steady R_theta per GPU type (dynamics from T4, gain from measured R_theta)
    print("\n=== per-GPU DCTM (dynamics from T4 transients, R_theta anchored to measured) ===")
    for gpu, rth in R_THETA.items():
        m = model.scale_to_rtheta(rth)
        print(f"  {gpu:8s}: R_theta={m.r_theta:.4f}  branches R={np.round(m.R,4)} tau={np.round(m.tau,1)}s")


if __name__ == "__main__":
    main()
