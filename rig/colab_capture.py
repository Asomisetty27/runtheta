"""
Capture the full NVML telemetry surface to CSV -- run on Colab (or the rig host) today.

Two jobs:
  1. DE-RISK THE PURCHASE (do this now, before the Friday spend): run this on a Colab T4
     under load and read the printed summary. It tells us what a real T4 actually exposes
     -- crucially, whether the throttle-reason bitmask and a hotspot/memory temperature are
     available, and at what rate. We are ASSUMING the T4 reports these; better to confirm on
     the T4 we already have access to than to discover a gap mid-degradation-run.
  2. BE THE RIG CAPTURE: on the real rig, this is the NVML half of the log (the Pico streams
     the measured-ambient half); the two are joined by timestamp in analysis.

Colab quick start (new cell):
    !pip -q install pynvml
    !git clone <this repo> && cd <repo>
    # in another cell, put the GPU under load:  !git clone https://github.com/wilicc/gpu-burn && cd gpu-burn && make && ./gpu_burn 120 &
    !python -m rig.colab_capture --seconds 120 --out t4_surface.csv

The script logs, per second: t, gpu_temp, hotspot_temp (if exposed), power_w, sm_clock,
util, throttle_bitmask, thermal_throttle. It NEVER assumes a field exists -- missing fields
log as empty and are reported in the summary. Pure-stdlib CSV; pynvml is the only extra dep.
"""
from __future__ import annotations

import argparse
import csv
import time
from typing import Optional

from .throttle import is_thermal_throttle

# Hotspot / memory-junction field ids (present on newer NVML; absent on some Turing).
# We probe them and degrade gracefully rather than assume.
_HOTSPOT_FIELD_CANDIDATES = (
    "NVML_FIELD_ID_GPU_HOTSPOT_TEMP",   # naming varies by pynvml version
    "NVML_FIELD_ID_MEMORY_TEMP",
)

CSV_COLUMNS = [
    "t", "gpu_temp_c", "hotspot_temp_c", "power_w", "sm_clock_mhz",
    "util_pct", "throttle_bitmask", "thermal_throttle",
]


def _throttle_reasons(pynvml, handle) -> Optional[int]:
    """Read the current clock throttle/event reason bitmask. NVIDIA renamed this call
    from ...ClocksThrottleReasons to ...ClocksEventReasons in recent NVML, so try both."""
    for fn in ("nvmlDeviceGetCurrentClocksEventReasons",
               "nvmlDeviceGetCurrentClocksThrottleReasons"):
        f = getattr(pynvml, fn, None)
        if f is not None:
            return f(handle)
    return None


def _try_hotspot(pynvml, handle) -> Optional[float]:
    """Attempt to read a hotspot/memory temperature via the field-values API. Returns None
    if this NVML/card does not expose it (common on Turing/T4) -- reported in the summary."""
    for name in _HOTSPOT_FIELD_CANDIDATES:
        fid = getattr(pynvml, name, None)
        if fid is None:
            continue
        try:
            vals = pynvml.nvmlDeviceGetFieldValues(handle, [fid])
            v = vals[0]
            if getattr(v, "nvmlReturn", 0) == 0:
                return float(v.value.siVal if hasattr(v.value, "siVal") else v.value.uiVal)
        except Exception:
            continue
    return None


def capture(seconds: int = 120, hz: float = 1.0, out: str = "nvml_surface.csv",
            gpu_index: int = 0) -> dict:
    """Poll NVML at `hz` for `seconds`, write CSV, and return a capability summary dict."""
    import pynvml   # imported here so the module stays importable without a GPU

    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
    name = pynvml.nvmlDeviceGetName(handle)
    name = name.decode() if isinstance(name, bytes) else name

    seen = {"hotspot": False, "throttle_reasons": False, "n": 0,
            "max_temp": 0.0, "thermal_throttle_samples": 0}
    period = 1.0 / hz
    t0 = time.time()

    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(CSV_COLUMNS)
        while time.time() - t0 < seconds:
            row_t = round(time.time() - t0, 3)
            temp = _safe(lambda: pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU))
            hot = _try_hotspot(pynvml, handle)
            power = _safe(lambda: pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0)
            clock = _safe(lambda: pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM))
            util = _safe(lambda: pynvml.nvmlDeviceGetUtilizationRates(handle).gpu)
            mask = _safe(lambda: _throttle_reasons(pynvml, handle))
            thermal = is_thermal_throttle(mask) if mask is not None else None

            if hot is not None:
                seen["hotspot"] = True
            if mask is not None:
                seen["throttle_reasons"] = True
            if temp is not None:
                seen["max_temp"] = max(seen["max_temp"], temp)
            if thermal:
                seen["thermal_throttle_samples"] += 1
            seen["n"] += 1

            w.writerow([row_t, temp, hot, power, clock, util, mask, thermal])
            fh.flush()
            time.sleep(max(0.0, period - (time.time() - t0 - row_t)))

    pynvml.nvmlShutdown()
    summary = {
        "gpu": name, "samples": seen["n"], "csv": out,
        "hotspot_exposed": seen["hotspot"],
        "throttle_reasons_exposed": seen["throttle_reasons"],
        "max_gpu_temp_c": round(seen["max_temp"], 1),
        "thermal_throttle_samples": seen["thermal_throttle_samples"],
    }
    _print_summary(summary)
    return summary


def _safe(fn):
    try:
        return fn()
    except Exception:
        return None


def _print_summary(s: dict) -> None:
    print("\n=== NVML capability summary (de-risk report) ===")
    print(f"GPU:                    {s['gpu']}")
    print(f"samples logged:         {s['samples']}  -> {s['csv']}")
    print(f"throttle-reason bitmask exposed: {s['throttle_reasons_exposed']}  "
          "(needed for the ground-truth throttle label)")
    print(f"hotspot temp exposed:   {s['hotspot_exposed']}  "
          "(needed for the F16 hotspot-vs-average gap arm; a thermocouple substitutes if not)")
    print(f"max GPU temp seen:      {s['max_gpu_temp_c']} C")
    print(f"thermal-throttle samples: {s['thermal_throttle_samples']}  "
          "(Colab cooling usually prevents thermal throttle -- 0 here is expected and fine;"
          " this run is to characterize the sensor surface, not to degrade)")


def main() -> None:
    ap = argparse.ArgumentParser(description="Capture the NVML telemetry surface to CSV.")
    ap.add_argument("--seconds", type=int, default=120)
    ap.add_argument("--hz", type=float, default=1.0)
    ap.add_argument("--out", type=str, default="nvml_surface.csv")
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()
    capture(seconds=args.seconds, hz=args.hz, out=args.out, gpu_index=args.gpu)


if __name__ == "__main__":
    main()
