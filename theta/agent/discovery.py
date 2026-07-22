"""
Discovery engine — Exploration and Discovery pattern (Gulli Ch 21).

The signature classifier already identifies UNKNOWN axes and names, per fault, the
observation that would make the call exact (its missing-axis ledger). Today that output
is passive: the agent says "I can't be sure, I'd need a coolant-outlet sensor" and stops.
This module makes it ACTIVE -- it turns the fleet's accumulated missing-axis gaps into a
ranked set of concrete, approvable recommendations to go acquire the missing signal, so
Theta constantly seeks finer detail instead of waiting for it to arrive.

Design (composes two patterns, deliberately):
  - Exploration/Discovery (Ch 21): proactively identify what would resolve an ambiguity.
  - Prioritization (Ch 20): rank the gaps by value/cost, so the operator is asked to
    acquire the ONE signal that unblocks the most urgent, most widespread ambiguity first.

Human-in-the-loop by construction (Ch 13): this PROPOSES; it never auto-runs a diagnostic
workload on someone's production cluster or provisions hardware. Proposals are ranked and
surfaced; a human approves. Same trust discipline as the governor.

Consumes GpuPrognosis.report() dicts. scipy-free, no LLM -- ranking is transparent
arithmetic, every proposal traceable to the units and gaps that produced it.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

# Acquisition-cost tiers, keyed by the classifier's `via` field. Lower = cheaper/faster
# to obtain, so a proposal that resolves the same ambiguity via a cheap route ranks above
# one needing new hardware. These are relative effort weights, not dollar costs.
_ACQUISITION_COST: dict[str, float] = {
    "service-log": 1.0,   # query existing maintenance records -- nearly free
    "workload":    2.0,   # run a short controlled workload (e.g. a power sweep) -- minutes
    "probe":       2.5,   # active probe / brief diagnostic -- minutes, mild intrusion
    "sensor":      5.0,   # install a physical sensor -- capital + downtime
}
_DEFAULT_COST = 3.0


@dataclass
class DiscoveryProposal:
    """One ranked recommendation to acquire a missing signal."""
    needs:          str            # the observation to acquire (from MissingAxis.needs)
    via:            str            # how: service-log | workload | probe | sensor
    resolves:       str            # which fault causes it would disambiguate
    units_blocked:  int            # how many GPUs are currently stuck on this same gap
    min_rul_s:      Optional[float]  # most urgent blocked unit's RUL (None if unknown)
    example_gpus:   list[str]      # a few blocked GPU ids, for traceability
    priority:       float          # value / cost -- higher = do first

    def recommendation(self) -> str:
        rul = ("unknown" if self.min_rul_s is None
               else f"~{self.min_rul_s/3600:.0f}h to predicted failure on the most urgent unit")
        return (f"Acquire '{self.needs}' via {self.via}: unblocks an exact-cause call on "
                f"{self.units_blocked} unit(s) ({rul}), resolving {self.resolves}. Approve?")


@dataclass
class _Gap:
    via: str
    resolves: set[str] = field(default_factory=set)
    gpus: list[str] = field(default_factory=list)
    ruls: list[float] = field(default_factory=list)


class DiscoveryEngine:
    """Fleet-level exploration planner. Feed it the current per-GPU prognoses; it returns
    the ranked acquisitions that would most improve diagnostic certainty."""

    @staticmethod
    def propose(reports: list[dict]) -> list[DiscoveryProposal]:
        # 1. Generation: collect every open missing-axis gap across the fleet. A gap is
        #    "open" only when a unit is in alarm AND the classifier could NOT make an
        #    exact call (identifiable == False) -- i.e. certainty is genuinely blocked.
        gaps: dict[str, _Gap] = {}
        for rep in reports:
            if not rep.get("in_alarm"):
                continue
            attr = rep.get("attribution")
            if not attr or attr.get("identifiable"):
                continue  # either no attribution, or already exact -> nothing to discover
            gpu = rep.get("gpu_id", "?")
            rul = None
            worst = rep.get("worst_component")
            for c in rep.get("components", []):
                if c["component"] == worst:
                    rul = c.get("rul_s")
                    break
            for ax in attr.get("missing_axes", []):
                g = gaps.setdefault(ax["needs"], _Gap(via=ax.get("via", "sensor")))
                g.resolves.add(ax.get("resolves", ""))
                g.gpus.append(gpu)
                if rul is not None:
                    g.ruls.append(rul)

        # 2. Prioritization: value = breadth (units blocked) x urgency (lower RUL ->
        #    higher), divided by acquisition cost. Transparent, inspectable arithmetic.
        proposals: list[DiscoveryProposal] = []
        for needs, g in gaps.items():
            cost = _ACQUISITION_COST.get(g.via, _DEFAULT_COST)
            min_rul = min(g.ruls) if g.ruls else None
            # urgency: a unit ~10h from predicted failure is far more urgent than ~1000h.
            # log-scaled so it doesn't dominate breadth entirely; 1.0 when RUL unknown.
            if min_rul is None:
                urgency = 1.0
            else:
                urgency = 1.0 + max(0.0, 3.0 - math.log10(max(min_rul / 3600.0, 0.1)))
            value = len(g.gpus) * urgency
            proposals.append(DiscoveryProposal(
                needs=needs, via=g.via,
                resolves=", ".join(sorted(x for x in g.resolves if x)),
                units_blocked=len(g.gpus), min_rul_s=min_rul,
                example_gpus=g.gpus[:5], priority=round(value / cost, 3),
            ))
        proposals.sort(key=lambda p: p.priority, reverse=True)
        return proposals
