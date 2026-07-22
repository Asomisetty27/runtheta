"""
Theta grounding knowledge base -- the corpus behind Agentic RAG (Gulli Ch 14).

Ch 14 distinguishes static context-injection RAG from *Agentic RAG*, where an
intelligent agent CALLS retrieval on demand when it detects a knowledge gap, then
reasons over what it gets back. That is the right fit for Theta: the operator
agent already reads live numbers from the deterministic tools; this corpus lets it
retrieve WHAT those numbers mean -- which validated finding backs a signature, the
per-generation R_theta context, and the concrete repair -- instead of inventing an
explanation from parametric memory. Exposed as the read-only `theta_knowledge_lookup`
tool the agent invokes itself.

Discipline: this corpus contains ONLY validated ground (F7/F15 detection, F16 TIM
magnitude, cross-generation R_theta, the honesty tiers, component repair playbooks).
It deliberately excludes the findings retracted by adversarial audit, so the agent
can never ground an answer on a claim we walked back.

Retrieval is keyword/token-overlap scoring over a small curated corpus -- honest and
sufficient at this size. The scale-up path (embeddings + hybrid BM25/vector search,
Ch 14's recommendation) matters only when the corpus grows past hand-curation.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass(frozen=True)
class KnowledgeEntry:
    topic: str
    content: str
    source: str
    keywords: tuple[str, ...] = field(default_factory=tuple)


# Curated grounding corpus. Every entry is validated ground; sources name the
# finding so the agent can cite it. No retracted findings appear here.
CORPUS: tuple[KnowledgeEntry, ...] = (
    KnowledgeEntry(
        topic="What R_theta is and why it separates busy-hot from failing-hot",
        content=(
            "R_theta = (T_junction - T_ambient) / P_GPU, in C/W. It expresses "
            "temperature as a residual from what power predicts, so a GPU that is "
            "merely under heavy load (high T, high P) has a normal R_theta, while a "
            "GPU whose cooling is degrading shows a rising R_theta at the same power. "
            "No incumbent (DCGM, Mission Control, Phaidra) computes this ratio."),
        source="README / core thesis",
        keywords=("rtheta", "r_theta", "thermal", "resistance", "cooling", "metric", "ratio"),
    ),
    KnowledgeEntry(
        topic="Peer-relative detection is validated on a real production fleet (F7)",
        content=(
            "On a real 64-GPU production H100 fleet, peer-relative R_theta (median-"
            "polish robust-z vs matched-power node-mates) blind-flags a degraded GPU "
            "with NO temporal warm-up, including units degraded before monitoring "
            "began. Rank stability rho=0.986 over time. This is the E009 method in "
            "peer.py. Detection and attribution are the VALIDATED capability."),
        source="F7 (E009 validation fleet)",
        keywords=("peer", "detection", "validated", "fleet", "e009", "blind", "flag", "robust-z"),
    ),
    KnowledgeEntry(
        topic="Independent re-confirmation of a blind-flagged GPU (F15)",
        content=(
            "The single strongest real-world validation to date: a major research "
            "university's own diagnostics independently re-confirmed the specific GPU "
            "that Theta had blind-flagged (GPU7), using a separate method. Two "
            "independent methods agreeing on the same unit is the discipline that "
            "makes the result strong -- the same two-engine cross-check the prognostic "
            "engine applies (engine_agreement)."),
        source="F15",
        keywords=("f15", "reconfirm", "independent", "validation", "gpu7", "cross-check", "agreement"),
    ),
    KnowledgeEntry(
        topic="Real TIM degradation magnitude (F16)",
        content=(
            "A real consumer-GPU TIM repaste dataset showed the thermal swing TIM "
            "pump-out/dry-out actually produces: on the order of an 18 C hotspot swing "
            "and a 29-39% R_theta change. This grounds the TIM-degradation signature "
            "in a measured magnitude, and de-risks the ~$1000 mini-rig approach."),
        source="F16",
        keywords=("tim", "repaste", "pump-out", "dry-out", "magnitude", "f16", "hotspot", "paste"),
    ),
    KnowledgeEntry(
        topic="R_theta is per-generation, not a power-scaling formula",
        content=(
            "Measured on real telemetry across four GPU generations: T4 ~0.66, V100 "
            "~0.17, A100 ~0.038, H100 ~0.05-0.10 C/W. It is NOT monotonic with TDP "
            "(H100 > A100 despite more power). Therefore threshold calibration must be "
            "per-GPU-generation; a single universal threshold or a power-scaling law "
            "does not transfer. This is why the bundled classifier refuses to start on "
            "non-T4 hardware without a calibration file."),
        source="F8-F13 cross-generation",
        keywords=("generation", "t4", "v100", "a100", "h100", "tdp", "calibration", "threshold", "transfer"),
    ),
    KnowledgeEntry(
        topic="The honesty tiers: what a prediction is allowed to claim",
        content=(
            "Detection/attribution is validated; forward lead-time is not. An RUL "
            "labeled UNCALIBRATED is a physically-plausible estimate with an assumed "
            "threshold -- never a firm time-to-failure. It becomes CALIBRATED only once "
            "real labeled failures for that component+GPU class move the boundary. "
            "Separately, the signature classifier reports whether a cause is "
            "'identifiable' (uniquely pinned) or only 'discriminated' (narrowed), and "
            "lists the missing axis that would sharpen it."),
        source="signature.py honesty tiers / calibration.py",
        keywords=("uncalibrated", "calibrated", "honesty", "tier", "lead-time", "rul", "identifiable", "confidence"),
    ),
    KnowledgeEntry(
        topic="Why there is no firm lead-time claim yet",
        content=(
            "No real GPU cooling-degradation-to-failure dataset exists anywhere "
            "(confirmed exhaustively). The prognostic stack is validated on synthetic + "
            "cross-domain run-to-failure (NASA C-MAPSS turbofan) + partial real "
            "production telemetry, but its calibration loop stays dormant until real "
            "GPU failure labels exist. Only a real install/rig/fleet partner produces "
            "those. Until then, RUL ships UNCALIBRATED by design."),
        source="Q_lead_time / route-forward synthesis",
        keywords=("lead-time", "leadtime", "dataset", "dormant", "cmapss", "calibration", "labels", "predict"),
    ),
    KnowledgeEntry(
        topic="Repair playbook: TIM / die-cooling degradation",
        content=(
            "Signal: rising R_theta / beta slope / recovery time-constant, mem-core "
            "delta drift. Mechanism: thermal-interface-material pump-out or dry-out, or "
            "lost die contact. Action for a human to verify: schedule a repaste / "
            "heatsink reseat on the flagged unit; confirm mounting pressure. Expected "
            "magnitude if it is TIM: ~29-39% R_theta swing (F16)."),
        source="playbook + F16",
        keywords=("tim", "repaste", "die", "cooling", "repair", "playbook", "reseat", "heatsink", "action"),
    ),
    KnowledgeEntry(
        topic="Repair playbook: fan / cooling-actuator wear",
        content=(
            "Signal: fan RPM residual diverges from duty-cycle command; recovery "
            "time-constant lengthens. Mechanism: bearing wear or duty-vs-RPM "
            "divergence. Action to verify: inspect/replace the fan module; check for "
            "obstruction. Distinguish from airflow blockage, which shows dust "
            "accumulation and inlet-delta rise without an RPM-command divergence."),
        source="playbook / fault_classifier",
        keywords=("fan", "bearing", "rpm", "actuator", "airflow", "blockage", "dust", "repair", "wear"),
    ),
    KnowledgeEntry(
        topic="Repair playbook: HBM / memory, power-delivery, fabric",
        content=(
            "HBM thermal: mem-core temp delta + ECC single-bit-error rate rise -> "
            "memory thermal stress; verify memory-side cooling. Power delivery/VRM: "
            "power-violation rate + clock-efficiency / perf-per-watt drop -> VRM aging "
            "or power instability; verify power stage. Fabric/interconnect: NVLink "
            "error rate + PCIe replay rate -> link degradation or connector wear; "
            "reseat the link/connector."),
        source="playbook / signature axes",
        keywords=("hbm", "memory", "ecc", "power", "vrm", "clock", "fabric", "nvlink", "pcie", "interconnect", "repair"),
    ),
    KnowledgeEntry(
        topic="Known detector failure mode: low-power false positive",
        content=(
            "R_theta = (T-amb)/P blows up as P -> 0, so a baseline+k-sigma detector "
            "can false-positive at very low power. The fix is a minimum-power gate "
            "before R_theta is trusted; classification also only runs on steady-state "
            "windows (sigma < 0.03 C/W). If a GPU is idle, decline to assess rather "
            "than flag."),
        source="detector.py / window.py invariants",
        keywords=("low-power", "false", "positive", "idle", "gate", "minimum", "power", "steady-state"),
    ),
)


_TOKEN = re.compile(r"[a-z0-9_]+")

# Drop high-frequency function words so an incidental "for"/"the" overlap cannot
# manufacture a false grounding match (a miss must return []).
_STOPWORDS = frozenset((
    "the", "a", "an", "and", "or", "of", "to", "is", "are", "in", "it", "its", "at",
    "on", "as", "by", "be", "for", "with", "that", "this", "from", "so", "not", "no",
    "if", "but", "than", "then", "what", "when", "why", "how", "does", "do", "has",
    "have", "was", "were", "will", "can", "get", "any", "all", "into", "out", "up",
    "about", "over", "per", "vs", "via",
))


def _tokens(text: str) -> set[str]:
    return {t for t in _TOKEN.findall(text.lower()) if t not in _STOPWORDS}


def lookup(query: str, k: int = 3) -> list[dict]:
    """Retrieve the top-k grounding entries for a query by keyword/token overlap.

    Scoring: explicit keyword hits weigh double (curated, high-signal), plus raw
    token overlap against topic+content. Returns [] on no match so the agent knows
    the corpus has nothing and must say so rather than fabricate grounding."""
    q = _tokens(query)
    if not q:
        return []
    scored = []
    for e in CORPUS:
        kw_hits = sum(1 for kw in e.keywords if kw in q or any(kw in t for t in q))
        overlap = len(q & _tokens(e.topic + " " + e.content))
        score = 2 * kw_hits + overlap
        if score > 0:
            scored.append((score, e))
    scored.sort(key=lambda s: s[0], reverse=True)
    return [
        {"topic": e.topic, "content": e.content, "source": e.source}
        for _, e in scored[:k]
    ]
