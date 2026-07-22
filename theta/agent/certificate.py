"""
theta certify — the verified GPU health certificate (schema v1).

The product decision this implements (2026-07-15, Amogh + Sam): theta issues a
VERIFIED HEALTH SCORE + component attribution + history certificate that FEEDS
a valuation — it never claims to BE the price. Market comps, warranty, and
demand set the baseline; this certificate is the condition pillar, the input
that makes the three classical appraisal approaches (market / income / cost)
computable for a GPU.

Design rules, each load-bearing:

  * HISTORY IS THE PRODUCT. A certificate backed by a continuous record states
    its span and coverage; gaps are EXPLICIT ("no record before install", like
    a CarFax coverage gap). With no agent history at all, the certificate is
    issued but downgraded to grade "point-in-time" — a snapshot cannot
    reproduce history (F1 thermal memory, F10 rank persistence, F25
    fault-before-symptom), and the certificate says so on its face.
  * COMPONENT ATTRIBUTION CARRIES A VALIDATION GRADE. The fault-curve
    classifier's TIM/cooling-path call is ground-truthed (GA102 optical
    repaste study); other causes are physics-guided heuristics. The
    certificate states which is which instead of implying uniform confidence.
  * RISK IS ACTUARIAL, NEVER PROPHETIC. The recurrence tier quotes the
    two-fleet measurement (56% re-fault within 30 d of an incident, GWDG;
    25.7% re-detachment, Summit) with its window arithmetic. No failure dates.
  * THE REFUSALS ARE PART OF THE CERTIFICATE. What theta will not certify —
    performance loss projected from R_theta (F25: no stable coefficient),
    failure dates (five-fleet nulls), HBM2 memory-delta health (F26) — is
    printed on the document. The refusal list is the credibility.

Pure assembly: no NVML, no I/O. Inputs are plain dicts gathered by the CLI
(live NVML identity, the running agent's health/econ state when available);
output is a canonical dict with a SHA-256 integrity hash over its sorted-JSON
payload. Fully testable without a GPU.
"""
from __future__ import annotations

import hashlib
import json
from typing import Optional

SCHEMA = "theta-certificate/v1"

# F23, two fleets, both cooling regimes: P(re-fault <= 30 d | hardware incident).
RECURRENCE_P_30D = 0.56
RECURRENCE_WINDOW_DAYS = 30.0

# Validation grade per fault-curve cause (2026-07-15 evidence state). The
# TIM/cooling-path call is ground-truthed against the GA102 optical repaste
# study; the other thermal causes are physics-guided curve-shape heuristics;
# the non-thermal causes come from fabric/power telemetry (measured counters,
# but the *attribution* is heuristic).
VALIDATION_GRADE = {
    "nominal":           "measured",
    "tim_degradation":   "ground-truthed",
    "dust_accumulation": "heuristic",
    "fan_bearing_wear":  "heuristic",
    "airflow_blockage":  "heuristic",
    "mounting_event":    "heuristic",
    "hbm_thermal":       "heuristic",
    "fabric_link":       "measured-counters",
    "power_delivery":    "measured-counters",
    "insufficient_data": "n/a",
}

# Printed on every certificate. Each refusal cites the evidence that mandates it.
REFUSALS = [
    "No failure-date prediction: five-fleet analysis found no thermal precursor "
    "for uncorrectable-memory or detachment failures (permutation-nulled).",
    "No performance-loss projection from R_theta elevation: healthy GPUs operate "
    "below the firmware throttle knee where no stable R_theta-to-clock "
    "coefficient exists (F25). Loss appears only as OBSERVED throttle residency.",
    "No HBM2 memory-delta health claims: the die-to-memory differential is a "
    "device signature, not a health signal, at fleet scale (F26).",
    "Risk statements are actuarial (measured cohort frequencies under stated "
    "conditions), never per-unit prophecy.",
]


def _integrity_hash(payload: dict) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(canonical.encode()).hexdigest()


def build_certificate(
    *,
    now: float,
    agent_version: str,
    identity: dict,
    record_span: Optional[tuple[float, float]] = None,
    observed_hours: Optional[float] = None,
    gaps: Optional[list[dict]] = None,
    health: Optional[dict] = None,        # GpuHealth.as_dict() from the agent
    econ: Optional[dict] = None,          # FleetEconomics.gpu_report()
    diagnosis: Optional[dict] = None,     # {cause, confidence, curve_slope, intercept}
    incident: Optional[dict] = None,      # {last_incident_ts, incident_count, kind}
    active_probe: Optional[dict] = None,  # activeprobe.grade_probe() output
) -> dict:
    """Assemble a v1 certificate from whatever evidence is available.

    Every section is present in the output; absent evidence is an explicit
    null with a reason, never a silent omission — an appraiser reading the
    document must be able to tell "healthy" from "not measured".
    """
    has_history = record_span is not None and observed_hours not in (None, 0.0)

    # ── record block: what this certificate is actually backed by ──────────
    if has_history:
        span_days = (record_span[1] - record_span[0]) / 86400.0
        coverage = None
        if span_days > 0 and observed_hours is not None:
            coverage = round(min(1.0, (observed_hours / 24.0) / span_days), 3)
        record = {
            "grade": "continuous-record",
            "span_start": record_span[0],
            "span_end": record_span[1],
            "span_days": round(span_days, 1),
            "observed_hours": round(observed_hours, 1),
            "coverage_frac": coverage,
            "gaps": gaps or [],
            "note": "History before span_start is not certified (no record).",
        }
    else:
        record = {
            "grade": "point-in-time",
            "span_start": None, "span_end": None, "span_days": 0.0,
            "observed_hours": 0.0, "coverage_frac": 0.0, "gaps": [],
            "note": (
                "POINT-IN-TIME INSPECTION ONLY - no continuous record backs this "
                "certificate. A snapshot cannot reproduce operating history "
                "(thermal memory, rank persistence, faults-before-symptoms); "
                "full-grade certificates require the theta agent recording since "
                "install."
            ),
        }

    # ── condition block ─────────────────────────────────────────────────────
    cause = (diagnosis or {}).get("cause", "insufficient_data")
    condition = {
        "health_status": (health or {}).get("status"),
        "schedulable": (health or {}).get("schedulable"),
        "active_conditions": [c["name"] for c in (health or {}).get("conditions", [])],
        "fault_attribution": {
            "cause": cause,
            "validation": VALIDATION_GRADE.get(cause, "heuristic"),
            "confidence": (diagnosis or {}).get("confidence"),
            "rtheta_curve_slope": (diagnosis or {}).get("curve_slope"),
            "rtheta_intercept": (diagnosis or {}).get("intercept"),
        },
        "min_thermal_margin_c": (econ or {}).get("min_thermal_margin_c"),
        "thermal_throttle_hours": (econ or {}).get("thermal_throttle_hours"),
        "throttle_residency_frac": (
            round((econ["thermal_throttle_hours"] / econ["observed_hours"]), 4)
            if econ and econ.get("observed_hours") else None
        ),
        "degraded_cooling": (econ or {}).get("degraded_cooling"),
    }

    # ── risk block: actuarial, condition-scoped ────────────────────────────
    on_watch = False
    window_remaining_days = None
    if incident and incident.get("last_incident_ts") is not None:
        elapsed_d = (now - incident["last_incident_ts"]) / 86400.0
        if elapsed_d < RECURRENCE_WINDOW_DAYS:
            on_watch = True
            window_remaining_days = round(RECURRENCE_WINDOW_DAYS - elapsed_d, 1)
    risk = {
        "recurrence_watch": on_watch,
        "window_remaining_days": window_remaining_days,
        "incident_count": (incident or {}).get("incident_count", 0),
        "last_incident_kind": (incident or {}).get("kind") or (incident or {}).get("last_incident_kind", ""),
        "p_refault_within_30d_given_incident": RECURRENCE_P_30D if on_watch else None,
        "basis": (
            "Two-fleet measurement (air + liquid cooling): 56% of post-incident "
            "units re-fault within 30 days; applies ONLY while the watch window "
            "is open. Outside the window, no elevated hazard is claimed."
        ) if on_watch else "No hardware incident in the last 30 days of record.",
    }

    payload = {
        "schema": SCHEMA,
        "generated_at": now,
        "agent_version": agent_version,
        "identity": identity,
        "record": record,
        "condition": condition,
        "active_probe": active_probe or {
            "kind": "active_probe/v1", "run": False,
            "note": "not run - passive certificate; run `theta certify --active` "
                    "for measured functional verification of compute and DRAM paths",
        },
        "risk": risk,
        "scope_refusals": REFUSALS,
    }
    return {**payload, "integrity": _integrity_hash(payload)}


def verify_certificate(cert: dict) -> bool:
    """Recompute the integrity hash. True iff the document is unmodified."""
    body = {k: v for k, v in cert.items() if k != "integrity"}
    return _integrity_hash(body) == cert.get("integrity")
