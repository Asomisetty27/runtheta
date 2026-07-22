"""
Characterization tests for the LLM operator-reasoning agent
(theta/agent/operator.py).

The Anthropic client is INJECTED as a scripted fake, so the ReAct loop, the
tool-dispatch audit trail, the turn-budget fallback, and the honesty guardrails
are all exercised with zero network calls. The point of these tests: pin that
the agent orchestrates the deterministic read-only tools correctly and never
depends on a live model to be validated.
"""
from types import SimpleNamespace

from theta.agent.operator import (
    OperatorAgent, OperatorAnswer, OperatorSession, anthropic_tools,
    OPERATOR_SYSTEM_PROMPT, CRITIC_SYSTEM_PROMPT, _parse_critic,
)


# ── fake Anthropic client: returns a scripted list of responses in order ──
def _text(t):
    return SimpleNamespace(type="text", text=t)


def _tool(tid, name, inp):
    return SimpleNamespace(type="tool_use", id=tid, name=name, input=inp)


def _resp(stop_reason, blocks):
    return SimpleNamespace(stop_reason=stop_reason, content=blocks)


class _Msgs:
    def __init__(self, scripted):
        self._scripted = list(scripted)
        self.calls = []

    def create(self, **kw):
        self.calls.append(kw)
        return self._scripted.pop(0)


class FakeClient:
    def __init__(self, scripted):
        self.messages = _Msgs(scripted)


# ── tool schema exposed to the model ──
def test_anthropic_tools_shape():
    tools = anthropic_tools()
    assert len(tools) == 6
    names = {t["name"] for t in tools}
    assert "theta_gpu_prognosis" in names
    assert "theta_knowledge_lookup" in names
    for t in tools:
        assert set(t) == {"name", "description", "input_schema"}  # Anthropic format


def test_system_prompt_encodes_honesty_contract():
    p = OPERATOR_SYSTEM_PROMPT
    assert "UNCALIBRATED" in p              # never present as firm lead-time
    assert "advisory" in p.lower()          # no action authority
    assert "never from memory" in p.lower() # answer from tools, not parametric memory


# ── availability / graceful degradation ──
def test_unavailable_without_client_dep_or_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(OperatorAgent, "dependency_present", staticmethod(lambda: False))
    agent = OperatorAgent()
    ok, reason = agent.available()
    assert ok is False
    assert "runtheta[agent]" in reason


def test_ask_when_unavailable_returns_message(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(OperatorAgent, "dependency_present", staticmethod(lambda: False))
    ans = OperatorAgent().ask("is anything degrading?")
    assert isinstance(ans, OperatorAnswer)
    assert ans.stopped_reason == "unavailable"
    assert ans.tool_calls == []


# ── the ReAct loop ──
def test_loop_calls_tool_then_answers():
    client = FakeClient([
        _resp("tool_use", [_tool("t1", "theta_fleet_summary", {})]),
        _resp("end_turn", [_text("All 8 GPUs nominal.")]),
    ])
    calls = []

    def fake_tool(name, args):
        calls.append((name, args))
        return {"summary": "8 GPUs nominal"}

    ans = OperatorAgent(client=client, tool_fn=fake_tool).ask("status?", reflect=False)
    assert ans.text == "All 8 GPUs nominal."
    assert ans.stopped_reason == "end_turn"
    assert ans.turns == 2
    assert calls == [("theta_fleet_summary", {})]
    # audit trail records the tool call AND its result
    assert len(ans.tool_calls) == 1
    assert ans.tool_calls[0].name == "theta_fleet_summary"
    assert ans.tool_calls[0].result == {"summary": "8 GPUs nominal"}


def test_loop_feeds_tool_result_back_to_model():
    # the second create() call must carry the tool_result in its messages
    client = FakeClient([
        _resp("tool_use", [_tool("t1", "theta_gpu_prognosis", {"gpu_index": 3})]),
        _resp("end_turn", [_text("GPU 3 TIM drifting, UNCALIBRATED RUL.")]),
    ])
    OperatorAgent(client=client, tool_fn=lambda n, a: {"worst_component": "tim"}).ask(
        "gpu 3?", reflect=False)
    second_call_msgs = client.messages.calls[1]["messages"]
    # last message is the user turn carrying the tool_result block
    tr = second_call_msgs[-1]["content"][0]
    assert tr["type"] == "tool_result"
    assert tr["tool_use_id"] == "t1"
    assert "tim" in tr["content"]


def test_tool_exception_becomes_data_not_crash():
    client = FakeClient([
        _resp("tool_use", [_tool("t1", "theta_gpu_details", {"gpu_index": 0})]),
        _resp("end_turn", [_text("Could not read GPU 0.")]),
    ])

    def boom(name, args):
        raise RuntimeError("daemon exploded")

    ans = OperatorAgent(client=client, tool_fn=boom).ask("gpu 0?", reflect=False)
    assert ans.stopped_reason == "end_turn"
    assert "error" in ans.tool_calls[0].result   # exception captured as data


def test_turn_budget_triggers_final_summary():
    # model never stops calling tools -> loop must cap at max_turns and do one
    # final tool-free summarizing pass.
    scripted = [_resp("tool_use", [_tool(f"t{i}", "theta_fleet_status", {})]) for i in range(2)]
    scripted.append(_resp("end_turn", [_text("Partial: fleet mostly nominal.")]))
    client = FakeClient(scripted)
    ans = OperatorAgent(client=client, tool_fn=lambda n, a: {"ok": True},
                        max_turns=2).ask("deep dive everything")
    assert ans.stopped_reason == "max_turns"
    assert ans.turns == 2
    assert "Partial" in ans.text
    # the final summarizing create() call must be tool-free (no tools kwarg)
    assert "tools" not in client.messages.calls[-1]


# ── Reflection (Ch 4) / output guardrail (Ch 18) ──
def test_reflection_clean_verdict_passes_through():
    client = FakeClient([
        _resp("end_turn", [_text("GPU 3 TIM health 0.71, UNCALIBRATED RUL — advisory only.")]),
        _resp("end_turn", [_text('{"ok": true, "issues": [], "fix_hint": ""}')]),  # critic
    ])
    ans = OperatorAgent(client=client, tool_fn=lambda n, a: {}).ask("gpu 3?")
    assert ans.reflected is True
    assert ans.refined is False
    assert ans.critic_issues == []
    assert ans.text.startswith("GPU 3 TIM")   # unchanged
    # exactly two model calls: draft + critic (no refinement)
    assert len(client.messages.calls) == 2


def test_reflection_flag_triggers_one_refinement():
    client = FakeClient([
        _resp("end_turn", [_text("GPU 3 will fail in 18 hours.")]),  # overstates UNCALIBRATED RUL
        _resp("end_turn", [_text('{"ok": false, "issues": ["states UNCALIBRATED RUL as firm '
                                  'time-to-failure"], "fix_hint": "label it uncalibrated"}')]),
        _resp("end_turn", [_text("GPU 3 shows an UNCALIBRATED RUL (~18h, not a firm estimate); "
                                 "have a human verify.")]),
    ])
    ans = OperatorAgent(client=client, tool_fn=lambda n, a: {}).ask("when does gpu 3 fail?")
    assert ans.reflected is True
    assert ans.refined is True
    assert "UNCALIBRATED" in ans.text          # the corrected answer
    assert ans.critic_issues and "UNCALIBRATED" in ans.critic_issues[0]
    assert len(client.messages.calls) == 3     # draft + critic + refine


def test_reflection_skipped_on_max_turns_answer():
    # a truncated answer is already labeled partial; do not reflect it
    scripted = [_resp("tool_use", [_tool(f"t{i}", "theta_fleet_status", {})]) for i in range(2)]
    scripted.append(_resp("end_turn", [_text("Partial answer.")]))
    client = FakeClient(scripted)
    ans = OperatorAgent(client=client, tool_fn=lambda n, a: {"ok": True},
                        max_turns=2).ask("dig", reflect=True)
    assert ans.stopped_reason == "max_turns"
    assert ans.reflected is False              # reflection correctly skipped


def test_critic_persona_is_independent():
    # the critic prompt must establish it did NOT write the draft (Ch 4 role separation)
    assert "did NOT write" in CRITIC_SYSTEM_PROMPT
    assert "advisory only" in CRITIC_SYSTEM_PROMPT


def test_parse_critic_robust():
    assert _parse_critic('{"ok": true, "issues": []}')["ok"] is True
    # JSON embedded in prose
    v = _parse_critic('Here is my verdict: {"ok": false, "issues": ["x"]} done.')
    assert v["ok"] is False and v["issues"] == ["x"]
    # garbage -> fail open (ok=true), never suppress a valid answer on a parse error
    assert _parse_critic("not json at all")["ok"] is True


# ── Memory (Ch 8): session carries context across turns ──
def test_session_memory_carries_prior_turns():
    session = OperatorSession()
    agent = OperatorAgent(client=FakeClient([_resp("end_turn", [_text("GPU 3 TIM drifting.")])]),
                          tool_fn=lambda n, a: {})
    agent.ask("how is gpu 3?", reflect=False, session=session)
    # first turn recorded as user+assistant
    assert session.turns == 1
    assert session.messages[0] == {"role": "user", "content": "how is gpu 3?"}
    assert session.messages[1]["role"] == "assistant"

    # second turn: the agent must receive the prior turns as leading context
    agent2 = OperatorAgent(client=FakeClient([_resp("end_turn", [_text("It is worsening.")])]),
                           tool_fn=lambda n, a: {})
    agent2.ask("is it getting worse?", reflect=False, session=session)
    sent_messages = agent2._client.messages.calls[0]["messages"]
    assert len(sent_messages) == 3   # 2 prior + the new question
    assert sent_messages[0]["content"] == "how is gpu 3?"
    assert sent_messages[-1]["content"] == "is it getting worse?"
    assert session.turns == 2


# ── Parallelization (Ch 3): multiple tools in one turn, order preserved ──
def test_multiple_tools_in_one_turn_preserve_order():
    client = FakeClient([
        _resp("tool_use", [
            _tool("t1", "theta_gpu_prognosis", {"gpu_index": 0}),
            _tool("t2", "theta_gpu_prognosis", {"gpu_index": 1}),
            _tool("t3", "theta_gpu_prognosis", {"gpu_index": 2}),
        ]),
        _resp("end_turn", [_text("GPUs 0-2 checked.")]),
    ])

    def fake_tool(name, args):
        return {"gpu_index": args["gpu_index"], "worst_health": 0.9}

    ans = OperatorAgent(client=client, tool_fn=fake_tool).ask("check 0,1,2", reflect=False)
    # all three executed, audit in request order
    assert [c.args["gpu_index"] for c in ans.tool_calls] == [0, 1, 2]
    # the tool_result blocks fed back must align with their tool_use ids in order
    second_msgs = client.messages.calls[1]["messages"]
    tool_result_ids = [tr["tool_use_id"] for tr in second_msgs[-1]["content"]]
    assert tool_result_ids == ["t1", "t2", "t3"]
