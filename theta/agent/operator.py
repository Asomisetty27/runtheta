"""
Theta operator-reasoning agent -- the LLM layer that sits ON TOP of the
deterministic prognostic core.

Design follows Antonio Gulli's "Agentic Design Patterns", applied where the
problem actually warrants it (not pattern-for-its-own-sake):

  - Tool Use (Ch: Tool Use / Function Calling): the agent answers operator
    questions by calling Theta's live tools (theta.mcp_server.call_tool) rather
    than guessing from parametric memory. The daemon's numbers are ground truth.
  - Reasoning Techniques / ReAct (Ch 17): the loop is reason -> act (call a
    tool) -> observe (tool result) -> reason, until the question is answered.
    Multi-hypothesis diagnosis: when a GPU is ambiguous, gather prognosis AND
    details AND risk, then reconcile, instead of committing to the first read.
  - Planning (Ch 6): a fleet-wide question is decomposed into per-GPU tool calls;
    the agent plans which GPUs to drill into from the fleet summary first.
  - Reflection (Ch 4): after the loop drafts an answer, an independent critic
    persona audits it against the honesty contract before it reaches the operator,
    with one bounded refinement pass (see _reflect / CRITIC_SYSTEM_PROMPT).
  - Parallelization (Ch 3): independent read-only tool calls in a single turn run
    concurrently (per-GPU fan-out), order preserved (see _execute_tools).
  - Guardrails / Safety (Ch 18): the STRONGEST guardrail here is structural, not
    prompted -- every tool the agent can call is READ-ONLY. It has no tool to
    fire an alert, drain a node, or change a threshold. So it physically cannot
    make the trust-critical failure/alert decision; that stays in the
    deterministic governor. The agent reads and explains; it never acts. The
    system prompt adds the honesty contract on top (UNCALIBRATED RUL is not a
    lead-time; distinguish validated detection from unproven prediction).
  - Human-in-the-Loop (Ch 13): output is advisory. Recommendations are framed as
    "consider / verify", and anything actionable is handed to a human operator,
    never auto-executed.
  - Goal Setting & Monitoring (Ch 11): the standing goal -- keep compute as close
    to 100% as possible by catching degradation early -- is the lens the agent
    reasons under, encoded in the system prompt.
  - Knowledge Retrieval / Agentic RAG (Ch 14): the agent CALLS theta_knowledge_lookup
    to ground an explanation in a validated finding + repair playbook when it hits a
    knowledge gap, rather than injecting a static corpus or inventing the mechanism.
  - Memory Management (Ch 8): OperatorSession carries short-term (in-context)
    conversation memory so follow-ups keep context; long-term memory is the separate
    persistent tier already served by incident_store / calibration.

`anthropic` is an OPTIONAL dependency (extra: runtheta[agent]); this module is
inert and importable without it, matching the repo's otlp pattern. The agentic
loop takes an injectable client so it is fully testable without a network call.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

DEFAULT_MODEL = "claude-sonnet-4-6"   # balanced product default; override per call
MAX_TURNS = 8                          # reason/act cycles before we stop and summarize
MAX_TOKENS = 1536


OPERATOR_SYSTEM_PROMPT = """\
You are Theta's fleet-reliability operator assistant. Theta is a GPU
thermal-power prognostic agent built on one signal, R_theta = (T_junction -
T_ambient) / P, plus a multi-component prognostic engine that grades each
subsystem's health and estimates remaining-useful-life (RUL).

YOUR STANDING GOAL: help the operator keep compute as close to 100% as possible
by surfacing degradation early and explaining it clearly -- which specific GPU
and which subsystem, how confident, and what a human should look at next.

HOW YOU WORK:
- Answer from the live tools, never from memory. The daemon's numbers are ground
  truth; you orchestrate and explain them. If a tool says the daemon is
  unreachable, say so plainly -- do not invent a fleet state.
- Reason then act: start broad (theta_fleet_summary / theta_fleet_status), then
  drill into any GPU that looks off with theta_gpu_prognosis and
  theta_gpu_details. Cross-check engines -- if the prognostic worst-component and
  the signature fault classifier disagree (engine_agreement: conflict), surface
  the disagreement, do not paper over it.
- Ground your explanations: before you explain what a signature MEANS or
  recommend a repair, call theta_knowledge_lookup to pull the validated finding
  and repair playbook behind it, and cite the source. If the lookup returns
  nothing, say you lack grounding rather than inventing the mechanism.

HONESTY CONTRACT (non-negotiable):
- Detection and peer-relative attribution are validated. Forward lead-time is
  NOT. An RUL whose rul_confidence is "UNCALIBRATED" is a physically-plausible
  estimate with an assumed threshold -- present it as such, never as a firm
  time-to-failure. Only call an RUL actionable when in_alarm is true AND the tier
  is "CALIBRATED".
- You are advisory only. You have no ability to drain a node, fire an alert, or
  change a threshold, and you must not imply you did. Recommend what a human
  should verify or watch; leave the action to them and to the deterministic
  governor.
- Prefer "I don't have enough signal to say" over a confident guess. Declining to
  assess is a valid, correct answer.

Keep answers tight and operator-grade: lead with the bottom line, then the
evidence, then what to check. No filler."""


# Reflection (Ch 4) as an independent Producer-Critic, which also serves as the
# output-level guardrail (Ch 18). A SEPARATE critic persona -- not the producer
# reviewing itself -- audits the draft against the honesty contract and the tool
# evidence, because Ch 4 is explicit that role separation avoids the cognitive
# bias of an agent grading its own work. Verdict is machine-readable so one
# bounded refinement pass can act on it (Ch 4: single cycle, bound the cost).
CRITIC_SYSTEM_PROMPT = """\
You are a skeptical GPU-reliability reviewer auditing a DRAFT answer from Theta's
operator agent before it reaches a human. You did NOT write the draft; your only
job is to catch violations of Theta's honesty contract and drift from the tool
evidence you are given. Flag the draft (ok=false) if ANY of these hold:
- it presents an UNCALIBRATED RUL or lead-time as a firm/definite time-to-failure
- it states a number, fault, or fleet state not supported by the tool evidence
- it implies it took an action (drained, alerted, throttled) -- the agent is
  advisory only and has no such power
- it papers over an engine_agreement "conflict" instead of surfacing it
- it narrates a fleet state when the tools reported the daemon unreachable/error

Respond with ONLY a JSON object, no prose:
{"ok": <bool>, "issues": [<short strings>], "fix_hint": "<one sentence>"}
If the draft is faithful and honest, return {"ok": true, "issues": [], "fix_hint": ""}."""


def anthropic_tools() -> list[dict]:
    """Theta's read-only tools in Anthropic tool-use format (input_schema).

    Derived from the SAME registry the MCP server exposes, so the agent and any
    external MCP client see one identical toolset."""
    from ..mcp_server import TOOLS
    return [
        {
            "name": t["name"],
            "description": t["description"],
            "input_schema": t["inputSchema"],
        }
        for t in TOOLS
    ]


@dataclass
class OperatorSession:
    """Short-term (in-context) session memory -- Ch 8's ephemeral tier.

    Carries the conversation so an operator can ask follow-ups ("what about GPU
    5?", "why?") and the agent keeps context. Long-term memory is a SEPARATE tier
    that Theta already serves through persistent stores (incident_store,
    calibration) -- this holds only the current conversation thread, lost when the
    session ends, exactly as Ch 8 frames short-term memory."""
    messages: list[dict] = field(default_factory=list)   # prior user/assistant turns
    turns: int = 0


@dataclass
class ToolCall:
    """One audit-trail entry: what the agent asked for and what it got back."""
    name: str
    args: dict
    result: dict


@dataclass
class OperatorAnswer:
    """Result of a reasoning session: the final answer plus a full audit trail of
    every deterministic tool call the LLM made to reach it (so the reasoning is
    inspectable, per the same trust discipline as the rest of Theta)."""
    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    turns: int = 0
    stopped_reason: str = "end_turn"   # end_turn | max_turns | unavailable
    reflected: bool = False            # a critic pass audited this answer (Ch 4/18)
    refined: bool = False              # the critic flagged it and it was rewritten
    critic_issues: list[str] = field(default_factory=list)


class OperatorAgent:
    """LLM reason-act loop over Theta's read-only tools.

    Pass a pre-built Anthropic client, or leave it None to construct one from
    ANTHROPIC_API_KEY. `available()` reports whether the agent can actually run
    (dependency + key present) so callers degrade gracefully."""

    def __init__(
        self,
        client: Any = None,
        model: str = DEFAULT_MODEL,
        max_turns: int = MAX_TURNS,
        tool_fn=None,
    ):
        self.model = model
        self.max_turns = max_turns
        self._client = client
        # Injection seam for tests: default to the shared MCP dispatch.
        if tool_fn is None:
            from ..mcp_server import call_tool
            tool_fn = call_tool
        self._tool_fn = tool_fn

    # ── availability (optional-dep + key), so the CLI can degrade cleanly ──
    @staticmethod
    def dependency_present() -> bool:
        try:
            import anthropic  # noqa: F401
            return True
        except ImportError:
            return False

    def available(self) -> tuple[bool, str]:
        if self._client is not None:
            return True, "ok"
        if not self.dependency_present():
            return False, ("The agent LLM layer needs the optional dependency. "
                           "Install it with: pip install 'runtheta[agent]'")
        if not os.environ.get("ANTHROPIC_API_KEY"):
            return False, "Set ANTHROPIC_API_KEY to use the operator agent."
        return True, "ok"

    def _ensure_client(self):
        if self._client is None:
            import anthropic
            self._client = anthropic.Anthropic()
        return self._client

    # ── public entry: ReAct loop (Ch 17/5) then Reflection guardrail (Ch 4/18) ──
    def ask(self, question: str, reflect: bool = True,
            session: "OperatorSession | None" = None) -> OperatorAnswer:
        ok, reason = self.available()
        if not ok:
            return OperatorAnswer(text=reason, stopped_reason="unavailable")

        client = self._ensure_client()
        # Ch 8: prepend the session's prior turns so follow-ups keep context.
        history = list(session.messages) if session is not None else []
        messages = history + [{"role": "user", "content": question}]
        answer = self._react(client, messages)
        # Only reflect on a completed answer -- a truncated (max_turns) answer is
        # already self-labeled partial, and reflecting a partial invites the critic
        # to fault it for incompleteness rather than for honesty.
        if reflect and answer.stopped_reason == "end_turn":
            answer = self._reflect(client, question, answer)
        if session is not None:
            session.messages.append({"role": "user", "content": question})
            session.messages.append({"role": "assistant", "content": answer.text})
            session.turns += 1
        return answer

    def _react(self, client, messages: list) -> OperatorAnswer:
        tools = anthropic_tools()
        messages = list(messages)   # own copy; do not mutate the caller's session
        audit: list[ToolCall] = []

        for turn in range(1, self.max_turns + 1):
            resp = client.messages.create(
                model=self.model,
                max_tokens=MAX_TOKENS,
                system=OPERATOR_SYSTEM_PROMPT,
                tools=tools,
                messages=messages,
            )
            blocks = list(resp.content)

            if getattr(resp, "stop_reason", None) != "tool_use":
                return OperatorAnswer(
                    text=_text_of(blocks),
                    tool_calls=audit,
                    turns=turn,
                    stopped_reason="end_turn",
                )

            # execute every requested tool, feed results back (ReAct: observe).
            # Parallelization (Ch 3): when a turn requests several INDEPENDENT
            # read-only tools (e.g. per-GPU prognosis across the fleet), run them
            # concurrently -- latency only, no correctness change, and order is
            # preserved so the tool_result blocks still line up with their calls.
            messages.append({"role": "assistant", "content": blocks})
            tool_blocks = [b for b in blocks if getattr(b, "type", None) == "tool_use"]
            executed = self._execute_tools(tool_blocks)
            tool_results = []
            for b, args, result in executed:
                audit.append(ToolCall(name=b.name, args=args, result=result))
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": b.id,
                    "content": json.dumps(result),
                })
            messages.append({"role": "user", "content": tool_results})

        # ran out of turns: one final, tool-free pass to summarize what we have
        final = client.messages.create(
            model=self.model,
            max_tokens=MAX_TOKENS,
            system=OPERATOR_SYSTEM_PROMPT,
            messages=messages + [{
                "role": "user",
                "content": "Turn budget reached. Give your best operator-grade "
                           "answer from the tool results so far, and say clearly "
                           "if it is incomplete.",
            }],
        )
        return OperatorAnswer(
            text=_text_of(list(final.content)),
            tool_calls=audit,
            turns=self.max_turns,
            stopped_reason="max_turns",
        )

    def _reflect(self, client, question: str, answer: OperatorAnswer) -> OperatorAnswer:
        """Producer-Critic reflection (Ch 4) = the output-level guardrail (Ch 18).

        An independent critic persona audits the draft against the honesty contract
        and the exact tool evidence the producer saw. On a clean verdict the draft
        passes through (marked reflected). On a flag, ONE bounded refinement pass
        rewrites it -- single cycle, so the safety gain does not turn into unbounded
        latency/cost (Ch 4's explicit tradeoff)."""
        evidence = json.dumps(
            [{"tool": c.name, "args": c.args, "result": c.result} for c in answer.tool_calls]
        )[:6000]
        verdict = client.messages.create(
            model=self.model,
            max_tokens=512,
            system=CRITIC_SYSTEM_PROMPT,
            messages=[{"role": "user", "content":
                       f"DRAFT ANSWER:\n{answer.text}\n\nTOOL EVIDENCE:\n{evidence}"}],
        )
        crit = _parse_critic(_text_of(list(verdict.content)))
        answer.reflected = True
        answer.critic_issues = list(crit.get("issues") or [])

        if crit.get("ok", True):
            return answer

        revised = client.messages.create(
            model=self.model,
            max_tokens=MAX_TOKENS,
            system=OPERATOR_SYSTEM_PROMPT,
            messages=[
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer.text},
                {"role": "user", "content":
                    "A reviewer flagged honesty-contract issues with your answer: "
                    f"{answer.critic_issues}. {crit.get('fix_hint', '')} "
                    "Rewrite the answer to fix these. Use only the evidence you already "
                    "gathered; make no new claims."},
            ],
        )
        answer.text = _text_of(list(revised.content))
        answer.refined = True
        return answer

    def _execute_tools(self, tool_blocks: list) -> list:
        """Run a turn's tool calls and return (block, args, result) in REQUEST order.

        Parallelization pattern (Ch 3): a single tool runs inline (no thread
        overhead); multiple independent read-only calls run concurrently on a small
        thread pool. Results are re-sorted to the original order so the
        tool_result blocks stay aligned with their tool_use ids."""
        prepared = [(b, dict(b.input or {})) for b in tool_blocks]
        if len(prepared) <= 1:
            return [(b, args, self._safe_call(b.name, args)) for b, args in prepared]

        from concurrent.futures import ThreadPoolExecutor
        results: list = [None] * len(prepared)
        with ThreadPoolExecutor(max_workers=min(len(prepared), 8)) as pool:
            futures = {
                pool.submit(self._safe_call, b.name, args): i
                for i, (b, args) in enumerate(prepared)
            }
            for fut in futures:
                i = futures[fut]
                b, args = prepared[i]
                results[i] = (b, args, fut.result())
        return results

    def _safe_call(self, name: str, args: dict) -> dict:
        """Never let a tool exception break the loop -- return it as data so the
        LLM can reason about the failure (and the audit trail records it)."""
        try:
            return self._tool_fn(name, args)
        except Exception as exc:   # pragma: no cover - defensive
            return {"error": f"tool {name} failed: {exc}"}


def _parse_critic(text: str) -> dict:
    """Parse the critic's JSON verdict robustly. If it can't be parsed, FAIL OPEN
    to ok=true: reflection is a safety net, not a gate, and a malformed critic
    verdict must not silently suppress a valid answer. The behavioral + tool-level
    guardrails still hold regardless."""
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        pass
    lo, hi = text.find("{"), text.rfind("}")
    if 0 <= lo < hi:
        try:
            return json.loads(text[lo:hi + 1])
        except json.JSONDecodeError:
            pass
    return {"ok": True, "issues": [], "fix_hint": ""}


def _text_of(blocks: list) -> str:
    """Concatenate the text blocks of an Anthropic response."""
    out = []
    for b in blocks:
        if getattr(b, "type", None) == "text":
            out.append(b.text)
    return "\n".join(out).strip() or "(no answer)"
