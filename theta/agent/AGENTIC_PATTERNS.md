# Theta x Agentic Design Patterns (Antonio Gulli)

How Theta maps to all 21 patterns in the book. This is a design map, not a
checklist to satisfy: patterns are used where the reliability problem warrants
them, and where a pattern does not fit a single-fleet reliability agent that is
stated plainly rather than faked. The organizing principle is Theta's one
non-negotiable: **the deterministic core makes every trust-critical numerical
call; the LLM layer orchestrates and explains, and can never fire an alert or
change the fleet.** Several patterns below exist specifically to enforce that.

Legend: [CORE] load-bearing and built - [PARTIAL] present, with a concrete next
step - [THIN] deliberately minimal today, honest reason given.

## Part One: Foundational

**1. Prompt Chaining [CORE].** Two chains. The deterministic tick is a fixed
chain: collector -> window sigma-gate -> R_theta -> classify -> detect -> govern
-> export (`daemon.py`). The operator agent is a *dynamic* chain: each turn's
tool result conditions the next reasoning step (`operator.py::_react`).

**2. Routing [CORE].** The governor routes alerts by trust tier - ground-truth
hardware faults (ECC/Xid/throttle) fire immediately, anything inferred from
R_theta statistics is held and rate-budgeted (`governor.py`). The operator routes
attention: start broad (`theta_fleet_summary`) then drill into the anomalous GPU.
Next step: explicit operator-question-type routing (status vs. diagnose vs.
explain) if question volume justifies it.

**3. Parallelization [CORE].** The prognostic engine fans out one monitor per
subsystem (`prognostic.py`), and when a turn requests several independent
read-only tools (e.g. per-GPU prognosis across the fleet) the agent executes them
concurrently on a thread pool, preserving request order so tool_result ids stay
aligned (`operator.py::_execute_tools`). A single call runs inline (no thread
overhead).

**4. Reflection [CORE].** Producer-Critic, not self-review. After the ReAct loop
drafts an answer, an independent critic persona (`CRITIC_SYSTEM_PROMPT`) audits it
against the honesty contract and the exact tool evidence; a flag triggers one
bounded refinement pass (`operator.py::_reflect`). Ch 4's role-separation point is
the whole reason it is a separate persona, and its single-cycle bound is Ch 4's
cost/latency tradeoff honored.

**5. Tool Use [CORE].** The agent answers from live tools, never parametric
memory (`operator.py` + `mcp_server.py::call_tool`). Implements Ch 5's six-step
mechanism exactly: define -> LLM decides -> structured call -> orchestrator
executes -> result returns -> LLM formulates. Every tool is read-only.

**6. Planning [CORE].** A fleet question is decomposed into fleet-scan then
per-GPU drill-downs; the plan is LLM-driven and encoded in the system prompt
rather than a separate planner agent - deliberate, matching the problem scale.

**7. Multi-Agent Collaboration [CORE].** Seven specialized `ComponentMonitor`s
(die/TIM/HBM/power/fan/fabric/silicon), each an expert on its subsystem's normal
micro-behavior, fused by the `GpuPrognosis` coordinator, then cross-checked
against the independent signature classifier (`engine_agreement`). Specialized-
monitors-fused-by-coordinator, not a swarm - see Ch 15 for why not full A2A.

## Part Two: Advanced Systems

**8. Memory Management [CORE].** Episodic: `incident_store` (before/after features
+ operator labels). Longitudinal: per-component health and micro-change history
in each monitor. Learned baselines: `baseline.py` (per-GPU virtual ambient),
`calibration.py` (JSONL-persisted failure boundaries).

**9. Learning and Adaptation [CORE].** `calibration.py` moves per-component
thresholds UNCALIBRATED -> CALIBRATED from operator-confirmed failures;
`workload_normalizer.py` learns E[channel|power]; `survival_rul.py` fits the RUL
curve from data. The incident_store is the flywheel substrate.

**10. Model Context Protocol (MCP) [CORE].** `mcp_server.py` is the theta-mcp
stdio JSON-RPC server; the operator agent consumes the *same* tool registry, so
an external MCP client and the internal agent see one identical toolset.

**11. Goal Setting and Monitoring [CORE].** The standing goal (keep compute near
100% by catching degradation early) is encoded in the operator system prompt and
is the lens it reasons under; `health.py` conditions are the level-state being
monitored; the per-component health index is the monitored goal variable.

## Part Three: Production Concerns

**12. Exception Handling and Recovery [CORE].** The governor circuit-breaks noisy
GPUs and budgets false positives; `daemon.py::_stage` isolates each pipeline stage
so one failure does not abort the tick; the integrity gate blocks a diagnosis on
untrustworthy telemetry rather than narrating garbage; `operator.py::_safe_call`
turns a tool exception into data the LLM can reason about.

**13. Human in the Loop [CORE].** `theta label` is the only path to CONFIRMED_CAUSE
ground truth; the operator agent is advisory and structurally cannot act;
`discovery.py` proposes human-approvable diagnostics; the governor gates inferential
alerts.

**14. Knowledge Retrieval (RAG) [CORE].** Agentic RAG (Ch 14's stronger variant):
the agent CALLS `theta_knowledge_lookup` on demand when it needs to ground an
explanation, retrieving from a curated corpus of validated findings (F7/F15
detection, F16 TIM magnitude, per-generation R_theta), the honesty tiers, and
per-component repair playbooks (`knowledge.py`). The corpus contains only validated
ground - retracted findings are deliberately excluded so the agent can never ground
an answer on a claim we walked back. Retrieval is keyword/token scoring over a small
curated set; embeddings + hybrid search is the scale-up path when it outgrows
hand-curation.

## Part Four: Multi-Agent Architectures

**15. Inter-Agent Communication (A2A) [THIN, deliberate].** Theta is a heavy
central brain plus light read-only edge collectors: one logical agent. There is no
second autonomous agent to negotiate with, so A2A would be architecture theater
today. It becomes real only with a fleet-of-agents topology (per-datacenter agents
coordinating a global reliability picture) - a plausible future, explicitly not the
current single-fleet product. Stated rather than shimmed.

**16. Resource-Aware Optimization [PARTIAL, deliberate].** The core split *is* this
pattern at the system level: heavy central reasoning, lightweight edge collector
(the process pitched to partners), and optional deps stay inert when absent so the
base agent is dependency-light. The operator `model` is configurable. Auto model
tiering (cheap model for status, strong for diagnosis) was considered and
deliberately NOT added: on a reliability product, under-thinking a real degradation
question is the expensive failure, and Ch 4/18 (quality over speed when correctness
matters) outweighs Ch 16's cost optimization here. The capable default stays; the
seam is left for a human to opt into a cheaper model explicitly.

**17. Reasoning Techniques [CORE].** The operator loop is ReAct (reason -> act ->
observe -> reason). The signature classifier does structured multi-hypothesis
reasoning: generate candidate causes across axes -> critic each against physics
evidence with a missing-axis ledger -> rank with honesty tiers. Same
generate/critic/rank shape that carried the data-mining work.

**18. Guardrails and Safety [CORE].** Multi-layered exactly as Ch 18 prescribes.
Behavioral: the honesty-contract system prompt. Tool-level: read-only tools by
construction (no drain/alert/threshold tool exists). Output-level: the reflection
critic. Oversight: HITL advisory + a fully inspectable tool-call audit trail. Plus
the deterministic governor, integrity gate, and signature honesty tiers underneath.

**19. Evaluation and Monitoring [CORE].** `metrics.py` + Prometheus/OTLP exporters;
`health_api` conditions; incident_store turns accuracy into a *measured* number via
labels; the 371-test characterization suite is the behavioral eval that pins real
incidents (E009) so a regression cannot silently re-baseline them.

**20. Prioritization [CORE].** `discovery.py` ranks proposals by
breadth x urgency / acquisition-cost; the governor prioritizes which alerts route
under budget; `GpuPrognosis.report` sorts components worst-health-first.

**21. Exploration and Discovery [CORE].** `discovery.py` actively proposes the
diagnostic that would resolve an ambiguous component (close the missing axis)
instead of waiting for the next incident.

## Honest scorecard

20 of 21 patterns are genuinely built, several deeply (Guardrails, Multi-Agent,
Exception Handling, Learning, RAG, Reflection). One is deliberately thin with a
stated reason: A2A (15) does not fit a single-agent product yet - it becomes real
only with a fleet-of-agents topology. One is partial by explicit choice:
Resource-Aware auto model-tiering (16) was rejected because quality-over-speed wins
on a reliability product. No pattern is claimed that is not in the code. The
patterns that matter most for a reliability product - the ones that keep the LLM
from ever making the trust-critical call - are the load-bearing ones: Guardrails
(read-only tools + critic), Human-in-the-Loop, Reflection, and Tool Use restricted
to read-only.
