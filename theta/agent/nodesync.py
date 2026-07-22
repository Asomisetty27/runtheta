"""
Node-synchronous event discrimination — one node event is not N dead GPUs.

The evidence (cross-dataset, 2026-07-15): 10 of 14 analyzable GWDG detachment
incidents were node-synchronous — every GPU on the host dropped within ~30
minutes — i.e. host/power-domain events, not device failures. Treating such a
burst as N independent device incidents manufactures device-level history that
belongs to no device: it opens N false recurrence-hazard windows (F23), skews
fleet failure accounting, and tells the operator "4 GPUs failed" when the
truth is "the node blipped."

This module classifies hardware-incident events (detachment / reset-required /
memory fault) as device-scoped or node-synchronous:

  * events accumulate per category in a sliding window (default 30 min, the
    GWDG-measured burst width);
  * when the number of distinct GPUs in-window crosses the threshold
    (>= min_gpus AND >= min_frac of the fleet), the burst is node-synchronous:
    the crossing event reports `first_of_burst` so the daemon can emit ONE
    node-scoped summary alert, and `burst_gpus` so it can void the per-device
    recurrence records made before the burst became visible;
  * below threshold, events stay device-scoped (record_incident as usual).

Per-GPU XID alerts still fire immediately — ground truth is never delayed or
suppressed. What changes is the *interpretation layer*: recurrence watches and
the summary story. Pure logic, no NVML; fully testable without a GPU.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

# GWDG measurement: node-synchronous detachments land within ~30 minutes.
DEFAULT_WINDOW_S = 1800.0
# Both gates must pass: an absolute floor (2 GPUs co-failing is common chance
# on big fleets) and a fleet fraction (on a 4-GPU node, 3 of 4 = node event;
# on an 8-GPU node it takes 6).
DEFAULT_MIN_GPUS = 3
DEFAULT_MIN_FRAC = 0.75


@dataclass
class NodeSyncVerdict:
    node_synchronous: bool
    first_of_burst: bool = False          # this event crossed the threshold
    burst_gpus: list[int] = field(default_factory=list)  # distinct GPUs in-window
    category: str = ""
    window_s: float = DEFAULT_WINDOW_S


class NodeSyncDiscriminator:
    """Sliding-window burst classifier for hardware-incident events."""

    def __init__(self, fleet_size: int,
                 window_s: float = DEFAULT_WINDOW_S,
                 min_gpus: int = DEFAULT_MIN_GPUS,
                 min_frac: float = DEFAULT_MIN_FRAC):
        self.fleet_size = max(1, fleet_size)
        self.window_s = window_s
        self.min_gpus = min_gpus
        self.min_frac = min_frac
        # category -> deque[(ts, gpu)]
        self._events: dict[str, deque] = {}
        # category -> ts of the burst already announced (so a burst summarizes once)
        self._announced: dict[str, float] = {}

    def _threshold(self) -> int:
        return max(self.min_gpus, math.ceil(self.min_frac * self.fleet_size))

    def record(self, gpu: int, ts: float, category: str) -> NodeSyncVerdict:
        q = self._events.setdefault(category, deque())
        q.append((ts, gpu))
        cutoff = ts - self.window_s
        while q and q[0][0] < cutoff:
            q.popleft()
        gpus = sorted({g for _, g in q})
        sync = len(gpus) >= self._threshold()
        first = False
        if sync:
            # a burst is "the same burst" while its window overlaps the last
            # announcement; a later, disjoint burst announces again
            last = self._announced.get(category)
            if last is None or ts - last > self.window_s:
                first = True
            self._announced[category] = ts
        return NodeSyncVerdict(node_synchronous=sync, first_of_burst=first,
                               burst_gpus=gpus, category=category,
                               window_s=self.window_s)
