"""
Thermal-throttle event detection from NVML clock-throttle reasons.

The ground-truth failure label for E-MR is the first sustained THERMAL throttle. NVML
exposes current throttle reasons as a bitmask (nvmlDeviceGetCurrentClocksThrottleReasons)
/ nvidia-smi as per-reason Active flags. We care ONLY about thermal slowdown -- a
POWER-CAP throttle is normal operation, not a cooling failure, and counting it would
fabricate throttle events. This module is pure (no NVML dependency) so it is testable
offline and reused by both the live capture and the analysis.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

# nvmlClocksThrottleReasons bits (from nvml.h). Only the thermal ones are failure signal.
THERMAL_SW_SLOWDOWN = 0x0000000000000020   # nvmlClocksThrottleReasonSwThermalSlowdown
THERMAL_HW_SLOWDOWN = 0x0000000000000040   # nvmlClocksThrottleReasonHwThermalSlowdown
# NOT counted as thermal failure (normal / different cause):
SW_POWER_CAP        = 0x0000000000000004   # hitting the power limit -- normal
HW_POWER_BRAKE      = 0x0000000000000080   # external power-brake assert

_THERMAL_MASK = THERMAL_SW_SLOWDOWN | THERMAL_HW_SLOWDOWN


def is_thermal_throttle(bitmask: int) -> bool:
    """True iff a THERMAL slowdown bit is set. Power-cap throttling returns False."""
    return bool(int(bitmask) & _THERMAL_MASK)


def thermal_flags_from_bitmasks(bitmasks: Sequence[int]) -> np.ndarray:
    """Vectorize a series of raw NVML throttle bitmasks to a bool thermal-throttle flag."""
    arr = np.asarray(bitmasks, dtype=np.int64)
    return (arr & _THERMAL_MASK) != 0


def first_sustained_true(flags: Sequence[bool], t: Sequence[float],
                         min_run: int = 3) -> Optional[float]:
    """Timestamp of the first thermal-throttle event that persists >= min_run consecutive
    samples. Requiring a short sustained run rejects a single transient blip (a one-sample
    throttle flag during a workload transient is not a cooling failure)."""
    f = np.asarray(flags, dtype=bool)
    t = np.asarray(t, dtype=float)
    if f.size == 0 or f.size != t.size:
        return None
    run = 0
    for i in range(f.size):
        if f[i]:
            run += 1
            if run >= min_run:
                return float(t[i - min_run + 1])   # start of the sustained run
        else:
            run = 0
    return None
