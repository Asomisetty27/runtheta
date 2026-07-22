"""
Characterization tests for the Theta MCP server dispatch (theta/mcp_server.py).

_handle() is pure JSON-RPC over the daemon's HTTP API, so it is testable without
a live daemon: with the daemon down, _api() returns an "unreachable" dict and every
tool must degrade gracefully (never raise, always return the content envelope). The
prognostic tool (added with the daemon prognostic wiring) is pinned here.
"""
import json

import theta.mcp_server as m


def test_prognosis_tool_is_registered():
    names = [t["name"] for t in m.TOOLS]
    assert "theta_gpu_prognosis" in names
    tool = next(t for t in m.TOOLS if t["name"] == "theta_gpu_prognosis")
    # the honesty contract must be stated in the description the LLM sees
    assert "UNCALIBRATED" in tool["description"]
    assert tool["inputSchema"]["required"] == ["gpu_index"]


def test_extract_prognosis_from_details():
    details = {"causal_explanation": {"prognosis": {
        "worst_component": "tim", "worst_health": 0.72, "in_alarm": True,
        "rul_confidence": "UNCALIBRATED", "components": []}}}
    out = m._prognosis_from_details(details, 2)
    assert out["gpu_index"] == 2
    assert out["worst_component"] == "tim"
    assert out["rul_confidence"] == "UNCALIBRATED"


def test_extract_prognosis_absent_returns_clear_note():
    out = m._prognosis_from_details({"causal_explanation": {}}, 3)
    assert out["prognosis"] is None
    assert "note" in out


def test_extract_prognosis_passes_through_api_error():
    err = {"error": "conn refused", "daemon_status": "unreachable"}
    assert m._prognosis_from_details(err, 0) == err


def test_tools_list_includes_all_tools():
    resp = m._handle({"method": "tools/list"})
    assert len(resp["tools"]) == 6
    names = {t["name"] for t in resp["tools"]}
    assert "theta_knowledge_lookup" in names


def test_knowledge_lookup_tool_returns_grounding():
    # local corpus, no daemon needed -> must return real grounding entries
    resp = m._handle({"method": "tools/call",
                      "params": {"name": "theta_knowledge_lookup",
                                 "arguments": {"query": "TIM degradation repair"}}})
    body = json.loads(resp["content"][0]["text"])
    assert body["results"]                       # non-empty for a known topic
    assert any("tim" in r["content"].lower() or "tim" in r["topic"].lower()
               for r in body["results"])
    assert all("source" in r for r in body["results"])


def test_prognosis_dispatch_graceful_when_daemon_down():
    # no daemon on the test port -> _api returns an error dict; the tool must still
    # return the content envelope with valid JSON, not raise.
    resp = m._handle({"method": "tools/call",
                      "params": {"name": "theta_gpu_prognosis", "arguments": {"gpu_index": 0}}})
    text = resp["content"][0]["text"]
    body = json.loads(text)            # must be valid JSON
    assert "error" in body or body.get("prognosis") is None


def test_unknown_tool_returns_error_envelope():
    resp = m._handle({"method": "tools/call",
                      "params": {"name": "theta_nope", "arguments": {}}})
    body = json.loads(resp["content"][0]["text"])
    assert "Unknown tool" in body["error"]
