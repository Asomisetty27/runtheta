"""
Active probe — fieldiag-lite functional verification for `theta certify --active`.

The capability audit (2026-07-15) is honest that passive telemetry cannot see
every component: die-level aging and parts of the memory path are only
endpoint-visible. This module closes the software-closable part of that gap:
a short, deliberate load that MEASURES functional health instead of inferring
it — the same category of evidence as a refurbisher's burn-in, but
spec-normalized, throttle-aware, and appended to the device's record rather
than replacing it.

What it runs (torch-gated; ~60 s default):
  1. GEMM throughput  — sustained fp16 matmul, achieved TFLOPS. Verifies the
     compute path end-to-end and exposes silent clock instability.
  2. Memory bandwidth — large triad (c = a + b), achieved GB/s. Exercises the
     full DRAM path; a card with marginal memory shows up here long before
     ECC counters tell the story.
  3. Telemetry watch  — temp/power/clock/throttle sampled during load, so the
     result records the CONDITIONS of the measurement (a probe that throttled
     is reported as such, not as a slow card).

What it does NOT do (printed in the result):
  * No pass/fail oracle beyond a conservative floor. Achieved-vs-spec below
    FLOOR_FRAC flags `below_floor` — everything else is a measured number for
    the record and the cohort corpus to price. Cohort z-scores are emitted as
    null with reason "pending corpus" until enough probes accumulate; we do
    not fake a distribution we have not measured (calibration before
    precision).
  * No claim about components the load cannot reach (VRM internals, solder
    fatigue — see the capability audit; those stay actuarial).

Execution requires torch with CUDA (optional dependency — absent torch, the
CLI degrades with a clear message; nothing else in theta depends on it).
Analysis (`grade_probe`) is pure and fully testable without a GPU.
"""
from __future__ import annotations

import time
from typing import Callable, Optional

# Conservative floor: achieved/spec below this flags the card. Healthy parts
# under proper cooling land 0.75-0.95 of spec on these workloads; 0.60 leaves
# headroom for container/driver overhead so the flag means something is wrong
# with the CARD, not the harness.
FLOOR_FRAC = 0.60

# Vendor-spec reference points for spec-fraction normalization. Values are
# THEORETICAL peaks from vendor datasheets (dense fp16 tensor TFLOPS without
# sparsity; memory bandwidth GB/s). Matching is by substring on the NVML name.
# An unmatched SKU still probes fine — it just reports absolute numbers with
# spec_frac = null ("unknown SKU").
SPECS = {
    # datacenter
    "H100":      {"fp16_tflops": 989.0,  "mem_bw_gbs": 3350.0},   # SXM, HBM3
    "A100":      {"fp16_tflops": 312.0,  "mem_bw_gbs": 2039.0},   # SXM 80GB
    "V100":      {"fp16_tflops": 125.0,  "mem_bw_gbs": 900.0},
    "T4":        {"fp16_tflops": 65.0,   "mem_bw_gbs": 320.0},
    "L40S":      {"fp16_tflops": 362.0,  "mem_bw_gbs": 864.0},
    # consumer (GA102-class, common on marketplaces)
    "RTX 3090":  {"fp16_tflops": 71.0,   "mem_bw_gbs": 936.0},
    "RTX 3080 Ti": {"fp16_tflops": 68.0, "mem_bw_gbs": 912.0},
    "RTX 4090":  {"fp16_tflops": 165.0,  "mem_bw_gbs": 1008.0},
}

THERMAL_BITS = 0x20 | 0x40   # SW/HW thermal slowdown


def lookup_spec(gpu_name: str) -> Optional[dict]:
    """Longest-substring SKU match so 'RTX 3080 Ti' wins over a '3080' entry."""
    best = None
    for key, spec in SPECS.items():
        if key in gpu_name and (best is None or len(key) > len(best[0])):
            best = (key, spec)
    return {"sku": best[0], **best[1]} if best else None


def grade_probe(
    measured: dict,
    gpu_name: str,
    *,
    floor_frac: float = FLOOR_FRAC,
) -> dict:
    """Pure analysis: normalize measured numbers against vendor spec.

    measured: {gemm_tflops, mem_bw_gbs, thermal_throttle_s, max_temp_c,
               mean_power_w, min_sm_mhz, duration_s}
    Returns the certificate-ready active_probe block.
    """
    spec = lookup_spec(gpu_name)
    throttled = measured.get("thermal_throttle_s", 0) > 0

    def _frac(key_meas: str, key_spec: str) -> Optional[float]:
        if not spec or measured.get(key_meas) is None:
            return None
        s = spec.get(key_spec)
        return round(measured[key_meas] / s, 3) if s else None

    gemm_frac = _frac("gemm_tflops", "fp16_tflops")
    bw_frac = _frac("mem_bw_gbs", "mem_bw_gbs")

    flags = []
    # A probe that thermally throttled measures the COOLING, not the silicon —
    # report it and withhold the below-floor verdict on compute.
    if throttled:
        flags.append("thermally_throttled_during_probe")
    else:
        if gemm_frac is not None and gemm_frac < floor_frac:
            flags.append("gemm_below_floor")
        if bw_frac is not None and bw_frac < floor_frac:
            flags.append("mem_bw_below_floor")

    return {
        "kind": "active_probe/v1",
        "sku_match": spec["sku"] if spec else None,
        "measured": {
            "gemm_tflops": measured.get("gemm_tflops"),
            "mem_bw_gbs": measured.get("mem_bw_gbs"),
            "duration_s": measured.get("duration_s"),
            "max_temp_c": measured.get("max_temp_c"),
            "mean_power_w": measured.get("mean_power_w"),
            "min_sm_mhz": measured.get("min_sm_mhz"),
            "thermal_throttle_s": measured.get("thermal_throttle_s", 0),
        },
        "spec_fraction": {
            "gemm": gemm_frac,
            "mem_bw": bw_frac,
            "basis": ("vendor datasheet theoretical peak (dense fp16, no "
                      "sparsity)" if spec else "unknown SKU - absolute values only"),
        },
        "cohort_z": None,
        "cohort_z_reason": "pending corpus - cohort distribution not yet measured",
        "floor_frac": floor_frac,
        "flags": flags,
        "validation": "measured-active",
        "scope_note": ("Functional verification of the compute and DRAM paths "
                       "under load. Does not reach VRM internals or solder "
                       "fatigue (see capability audit); those remain covered "
                       "actuarially."),
    }


def run_active_probe(
    gpu_index: int = 0,
    seconds: float = 60.0,
    sample_fn: Optional[Callable[[], dict]] = None,
) -> dict:
    """Execute the load and return raw measured numbers (torch-gated).

    sample_fn: optional callable returning {temp_c, power_w, sm_mhz, throttle}
    per second (the CLI wires pynvml); without it, telemetry fields are None.
    Raises RuntimeError with an actionable message when torch/CUDA is absent.
    """
    try:
        import torch
    except ImportError as e:
        raise RuntimeError(
            "active probe requires torch with CUDA (pip install torch). "
            "The passive certificate works without it."
        ) from e
    if not torch.cuda.is_available():
        raise RuntimeError("torch sees no CUDA device - cannot run the active probe")

    dev = torch.device(f"cuda:{gpu_index}")
    torch.cuda.set_device(dev)

    # telemetry accumulator
    temps, powers, clocks = [], [], []
    throttle_s = 0

    def _sample():
        nonlocal throttle_s
        if sample_fn is None:
            return
        d = sample_fn()
        if d.get("temp_c") is not None:
            temps.append(d["temp_c"])
        if d.get("power_w") is not None:
            powers.append(d["power_w"])
        if d.get("sm_mhz") is not None:
            clocks.append(d["sm_mhz"])
        if d.get("throttle") and (d["throttle"] & THERMAL_BITS):
            throttle_s += 1

    # ── phase 1: GEMM throughput (2/3 of the budget) ────────────────────────
    n = 8192
    a = torch.randn(n, n, device=dev, dtype=torch.float16)
    b = torch.randn(n, n, device=dev, dtype=torch.float16)
    for _ in range(3):                       # warm-up / cuBLAS heuristics
        a @ b
    torch.cuda.synchronize()
    gemm_budget = seconds * (2 / 3)
    flops_per = 2 * n ** 3
    iters = 0
    t0 = time.perf_counter()
    next_sample = t0 + 1.0
    while time.perf_counter() - t0 < gemm_budget:
        a @ b
        iters += 1
        if iters % 4 == 0:
            torch.cuda.synchronize()
            if time.perf_counter() >= next_sample:
                _sample()
                next_sample += 1.0
    torch.cuda.synchronize()
    gemm_s = time.perf_counter() - t0
    gemm_tflops = flops_per * iters / gemm_s / 1e12

    # ── phase 2: memory bandwidth, triad c = a + b (1/3 of the budget) ─────
    m = 2 ** 26                              # 64M fp32 elements = 256 MB/tensor
    x = torch.randn(m, device=dev, dtype=torch.float32)
    y = torch.randn(m, device=dev, dtype=torch.float32)
    z = torch.empty_like(x)
    torch.add(x, y, out=z)                   # warm-up
    torch.cuda.synchronize()
    bw_budget = seconds / 3
    bytes_per = 3 * m * 4                    # read a, read b, write c
    iters = 0
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < bw_budget:
        torch.add(x, y, out=z)
        iters += 1
        if iters % 8 == 0:
            torch.cuda.synchronize()
            if time.perf_counter() >= next_sample:
                _sample()
                next_sample += 1.0
    torch.cuda.synchronize()
    bw_s = time.perf_counter() - t0
    mem_bw_gbs = bytes_per * iters / bw_s / 1e9

    return {
        "gemm_tflops": round(gemm_tflops, 1),
        "mem_bw_gbs": round(mem_bw_gbs, 1),
        "duration_s": round(gemm_s + bw_s, 1),
        "max_temp_c": max(temps) if temps else None,
        "mean_power_w": round(sum(powers) / len(powers), 1) if powers else None,
        "min_sm_mhz": min(clocks) if clocks else None,
        "thermal_throttle_s": throttle_s,
    }
