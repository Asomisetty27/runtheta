"""Fleet economics — turn R_theta detections into kilowatts and dollars.

The gap this closes: theta *detects* degraded cooling (elevated R_theta,
peer-relative) but never says what it COSTS. Detection is a nice-to-have;
a P&L line is a need. This module is the accounting layer that converts the
signals the agent already computes into three defensible numbers an operator
acts on:

  1. REALIZED loss  — throughput lost right now to thermal throttling,
     attributed to the units whose cooling is degraded. Measured as the
     clock deficit during *thermally* (not power-cap) throttled time, valued
     at the GPU-hour price. This is money already leaving the building.

  2. HEADROOM risk  — how close a degraded unit is to the throttle cliff
     (thermal margin, C). A unit at +X% R_theta reaches its thermal limit at
     a lower ambient/load than its peers; it is the one that throttles first
     on the next hot day. Leading indicator, before realized loss spikes.

  3. RECURRENCE exposure — F23: a unit that had a hardware incident has a
     weeks-scale elevated re-fault hazard (GWDG median 26 d, 56% within 30 d).
     Expected downtime cost = P(re-incident in horizon) x downtime x GPU-hour.

  4. STRAGGLER exposure — in synchronous training the slowest GPU sets the
     pace of every step ("even a single straggler can slow down thousands of
     other GPUs" — Llama 3 paper; ByteDance measured 10.4% of GPU-hours lost
     to stragglers fleet-wide). When a unit is OBSERVED thermally throttled
     while part of a synchronous job, its clock deficit taxes every peer
     waiting at the barrier: exposure = deficit x sync_peers x GPU-hour price.
     This is an explicit UPPER BOUND (assumes a fully synchronous workload
     with no slack); it is zero unless the caller supplies the job's peer
     count, and it accrues ONLY on observed thermally-throttled time — per
     F25, R_theta elevation alone must never be projected into a loss number
     (healthy GPUs sit below the throttle knee where dClock/dR_theta ~ 0, so
     no stable R_theta->loss coefficient exists).

Only theta produces (1) with *attribution* (this throttle is because THIS unit's
cooling is degraded vs its peers, not because the workload is heavy) and (2)/(3)
at all. DCGM sees the throttle; it does not see the peer-relative cause.

Pure accounting: no NVML, no I/O. Fed once per tick from the daemon's existing
per-GPU state; reported on demand. Fully testable without a GPU.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

# Thermal-throttle reason bits (NVML nvmlClocksThrottleReasons*). We count only
# thermally-caused throttling toward realized loss — power-cap throttling is an
# intentional operator/BMC choice, not a cooling defect, and blaming it on
# cooling would inflate the number dishonestly.
_THERMAL_THROTTLE_BITS = 0x20 | 0x40  # SW thermal slowdown | HW thermal slowdown

# F23 recurrence: fraction of units that re-incident within the default 30-day
# hazard window (GWDG measured 56% within 30 d). Used for expected-cost weighting.
_RECURRENCE_P_WITHIN_WINDOW = 0.56


@dataclass(frozen=True)
class PriceConfig:
    """Operator-supplied prices. Defaults are deliberately conservative."""
    gpu_hour_usd: float = 2.0     # rental / opportunity cost of one GPU-hour
    kwh_usd: float = 0.12         # electricity price
    pue: float = 1.3              # facility overhead multiplier on IT power
    incident_downtime_hours: float = 4.0  # expected node downtime per hardware incident


@dataclass
class GpuEconomics:
    gpu_index: int
    observed_seconds: float = 0.0
    thermal_throttle_seconds: float = 0.0
    # clock-deficit-weighted throttle time: 60 s throttled at 90% clock = 6 lost GPU-s
    lost_gpu_seconds: float = 0.0
    # deficit x sync-peer-count weighted time: the barrier tax this straggler
    # imposes on the REST of its synchronous job (upper bound; 0 if no job info)
    straggler_gpu_seconds: float = 0.0
    # worst (smallest) thermal margin to the throttle limit seen while loaded, C
    min_thermal_margin_c: Optional[float] = None
    degraded_cooling: bool = False       # elevated R_theta vs peers (theta's unique call)
    on_recurrence_watch: bool = False     # F23 post-incident hazard window
    _last_ts: Optional[float] = field(default=None, repr=False)


class FleetEconomics:
    """Accumulates per-GPU cost signals and rolls up a fleet report."""

    def __init__(self, prices: Optional[PriceConfig] = None):
        self.prices = prices or PriceConfig()
        self._g: dict[int, GpuEconomics] = {}

    def _gpu(self, i: int) -> GpuEconomics:
        if i not in self._g:
            self._g[i] = GpuEconomics(i)
        return self._g[i]

    def observe(
        self,
        gpu_index: int,
        ts: float,
        *,
        power_w: float,
        throttle_reasons: int,
        sm_clock_mhz: int,
        sm_clock_max_mhz: int,
        thermal_margin_c: Optional[float] = None,
        degraded_cooling: bool = False,
        on_recurrence_watch: bool = False,
        sync_peers: int = 0,
    ) -> None:
        """Feed one tick of a GPU's state. dt is inferred from successive ts.

        sync_peers: number of OTHER GPUs in the same synchronous job (e.g. 7 on
        an 8-GPU data-parallel job; job_gpus - 1 fleet-wide). Enables the
        straggler upper bound; leave 0 when job topology is unknown.
        """
        g = self._gpu(gpu_index)
        g.degraded_cooling = g.degraded_cooling or degraded_cooling
        g.on_recurrence_watch = on_recurrence_watch
        if thermal_margin_c is not None and power_w > 0:
            g.min_thermal_margin_c = (thermal_margin_c if g.min_thermal_margin_c is None
                                      else min(g.min_thermal_margin_c, thermal_margin_c))
        if g._last_ts is None:
            g._last_ts = ts
            return
        dt = ts - g._last_ts
        g._last_ts = ts
        if dt <= 0 or dt > 3600:   # ignore gaps / clock jumps
            return
        g.observed_seconds += dt
        thermally_throttled = bool(throttle_reasons & _THERMAL_THROTTLE_BITS)
        if thermally_throttled and sm_clock_max_mhz > 0 and sm_clock_mhz > 0:
            g.thermal_throttle_seconds += dt
            deficit = max(0.0, 1.0 - sm_clock_mhz / sm_clock_max_mhz)
            g.lost_gpu_seconds += dt * deficit
            if sync_peers > 0:
                # barrier tax: every peer runs at this straggler's pace
                g.straggler_gpu_seconds += dt * deficit * sync_peers

    # ── reporting ────────────────────────────────────────────────────────────
    def gpu_report(self, i: int) -> dict:
        g = self._g[i]
        p = self.prices
        realized_usd = g.lost_gpu_seconds / 3600.0 * p.gpu_hour_usd
        straggler_usd = g.straggler_gpu_seconds / 3600.0 * p.gpu_hour_usd
        recurrence_usd = (
            _RECURRENCE_P_WITHIN_WINDOW * p.incident_downtime_hours * p.gpu_hour_usd
            if g.on_recurrence_watch else 0.0
        )
        return {
            "gpu_index": i,
            "observed_hours": round(g.observed_seconds / 3600.0, 2),
            "thermal_throttle_hours": round(g.thermal_throttle_seconds / 3600.0, 3),
            "lost_gpu_hours": round(g.lost_gpu_seconds / 3600.0, 3),
            "realized_loss_usd": round(realized_usd, 2),
            "min_thermal_margin_c": g.min_thermal_margin_c,
            "degraded_cooling": g.degraded_cooling,
            "on_recurrence_watch": g.on_recurrence_watch,
            "recurrence_exposure_usd": round(recurrence_usd, 2),
            # upper bound: barrier tax on synchronous-job peers (0 without job info)
            "straggler_exposure_usd": round(straggler_usd, 2),
        }

    def fleet_report(self) -> dict:
        """Fleet roll-up: the numbers that go on an operator's dashboard."""
        gpus = [self.gpu_report(i) for i in sorted(self._g)]
        realized = sum(g["realized_loss_usd"] for g in gpus)
        exposure = sum(g["recurrence_exposure_usd"] for g in gpus)
        straggler = sum(g["straggler_exposure_usd"] for g in gpus)
        lost_hours = sum(g["lost_gpu_hours"] for g in gpus)
        obs_hours = sum(g["observed_hours"] for g in gpus)
        degraded = [g for g in gpus if g["degraded_cooling"]]
        # capacity-at-risk: degraded units near the throttle cliff (< 5 C margin)
        at_cliff = [g for g in degraded
                    if g["min_thermal_margin_c"] is not None and g["min_thermal_margin_c"] < 5.0]
        return {
            "gpus": len(gpus),
            "observed_gpu_hours": round(obs_hours, 1),
            "realized_loss_usd": round(realized, 2),
            "lost_gpu_hours": round(lost_hours, 3),
            "throttle_loss_pct": round(100.0 * lost_hours / obs_hours, 3) if obs_hours else 0.0,
            "degraded_cooling_units": len(degraded),
            "capacity_at_cliff_units": len(at_cliff),
            "recurrence_exposure_usd": round(exposure, 2),
            "straggler_exposure_usd": round(straggler, 2),
            # the single number for the dashboard: money in play right now
            "total_exposure_usd": round(realized + exposure + straggler, 2),
            "ranked_units": sorted(
                [g for g in gpus if g["realized_loss_usd"] > 0 or g["degraded_cooling"]
                 or g["on_recurrence_watch"]],
                key=lambda g: (g["realized_loss_usd"] + g["recurrence_exposure_usd"]
                               + g["straggler_exposure_usd"]),
                reverse=True,
            )[:20],
        }
