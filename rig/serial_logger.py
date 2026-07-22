"""
Host side of the rig log: read the Pico ambient stream, and join measured ambient onto
the NVML capture so `analyze.py` gets a complete RunTrace with a REAL T_ambient.

Two jobs:
  - read_serial(): timestamp each Pico line with host epoch and write CSV (needs pyserial).
  - merge_ambient() / build_run_trace(): join the Pico ambient onto the NVML capture by
    host time (nearest sample) -> the (T_j, T_ambient, P, throttle) trace R_theta needs.

The parsing + merge are pure and testable offline; only read_serial touches hardware.
"""
from __future__ import annotations

import csv
import time
from typing import Optional

import numpy as np

from .analyze import RunTrace

PICO_COLUMNS = ["host_epoch", "elapsed_ms", "inlet_c", "exhaust_c", "case_c",
                "fan_duty", "fan_rpm"]


def parse_line(line: str) -> Optional[dict]:
    """Parse one Pico CSV line 'elapsed_ms,inlet,exhaust,case,duty,rpm'. Empty fields ->
    None. Returns None on a malformed/partial line so serial noise never crashes the log."""
    parts = line.strip().split(",")
    if len(parts) != 6:
        return None
    try:
        elapsed = int(parts[0])
    except ValueError:
        return None

    def f(x):
        x = x.strip()
        return float(x) if x not in ("", "None") else None

    return {"elapsed_ms": elapsed, "inlet_c": f(parts[1]), "exhaust_c": f(parts[2]),
            "case_c": f(parts[3]), "fan_duty": f(parts[4]), "fan_rpm": f(parts[5])}


def read_serial(port: str, seconds: int, out: str, baud: int = 115200) -> int:
    """Read the Pico stream for `seconds`, stamp each valid line with host epoch, write CSV.
    Returns the number of rows written. Needs pyserial (imported here so the module stays
    importable without it)."""
    import serial

    n = 0
    t_end = time.time() + seconds
    with serial.Serial(port, baud, timeout=1) as ser, open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(PICO_COLUMNS)
        while time.time() < t_end:
            raw = ser.readline().decode(errors="ignore")
            rec = parse_line(raw)
            if rec is None:
                continue
            w.writerow([round(time.time(), 3), rec["elapsed_ms"], rec["inlet_c"],
                        rec["exhaust_c"], rec["case_c"], rec["fan_duty"], rec["fan_rpm"]])
            fh.flush()
            n += 1
    return n


def merge_ambient(nvml_rows: list[dict], pico_rows: list[dict],
                  nvml_start_epoch: float, ambient_key: str = "inlet_c") -> list[dict]:
    """Attach measured ambient to each NVML sample by nearest host time.

    nvml_rows: dicts with 't' (elapsed s from capture start) + gpu_temp_c/power_w/etc.
    pico_rows: dicts with 'host_epoch' + inlet_c/... (from the logged Pico CSV).
    nvml_start_epoch: host epoch at NVML capture t=0 (so nvml host time = start + t).
    Returns nvml_rows each with an added 'ambient_c' (None if no pico data)."""
    if not pico_rows:
        return [{**r, "ambient_c": None} for r in nvml_rows]
    p_epoch = np.array([float(p["host_epoch"]) for p in pico_rows])
    p_amb = np.array([np.nan if p.get(ambient_key) in (None, "") else float(p[ambient_key])
                      for p in pico_rows])
    out = []
    for r in nvml_rows:
        host_t = nvml_start_epoch + float(r["t"])
        j = int(np.argmin(np.abs(p_epoch - host_t)))
        amb = p_amb[j]
        out.append({**r, "ambient_c": (None if np.isnan(amb) else float(amb))})
    return out


def build_run_trace(nvml_rows: list[dict], pico_rows: list[dict],
                    nvml_start_epoch: float, ambient_key: str = "inlet_c") -> RunTrace:
    """Join NVML + Pico into a RunTrace ready for analyze_run. Samples with no measured
    ambient are dropped (R_theta needs a real T_ambient -- fabricating one is the very
    thing the rig exists to avoid)."""
    merged = merge_ambient(nvml_rows, pico_rows, nvml_start_epoch, ambient_key)
    keep = [m for m in merged if m["ambient_c"] is not None]
    t = np.array([float(m["t"]) for m in keep])
    tj = np.array([float(m["gpu_temp_c"]) for m in keep])
    amb = np.array([float(m["ambient_c"]) for m in keep])
    pw = np.array([float(m["power_w"]) for m in keep])
    thr = np.array([str(m.get("thermal_throttle")).strip().lower() == "true" for m in keep])
    hot = None
    if keep and keep[0].get("hotspot_temp_c") not in (None, "", "None"):
        hot = np.array([float(m["hotspot_temp_c"]) for m in keep])
    return RunTrace(t=t, t_junction=tj, t_ambient=amb, power_w=pw,
                    thermal_throttle=thr, hotspot=hot)
