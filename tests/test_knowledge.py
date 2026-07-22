"""
Tests for the Agentic RAG grounding corpus (theta/agent/knowledge.py, Ch 14).

Pins that retrieval returns relevant validated grounding, empties honestly on a
miss, and -- the load-bearing discipline -- contains NO retracted findings, so the
operator agent can never ground an answer on a claim we walked back.
"""
from theta.agent.knowledge import lookup, CORPUS


def test_lookup_returns_relevant_entry():
    res = lookup("TIM degradation repair playbook")
    assert res
    assert any("tim" in r["topic"].lower() or "tim" in r["content"].lower() for r in res)
    assert all({"topic", "content", "source"} <= set(r) for r in res)


def test_lookup_ranks_and_caps_k():
    res = lookup("R_theta per generation calibration threshold", k=2)
    assert 1 <= len(res) <= 2
    assert "generation" in (res[0]["topic"] + res[0]["content"]).lower()


def test_lookup_empty_on_no_match():
    # unrelated query -> corpus has nothing; must return [] so the agent declines
    assert lookup("quarterly revenue forecast for pizza") == []
    assert lookup("") == []


def test_uncalibrated_grounding_present():
    res = lookup("what is UNCALIBRATED RUL lead-time")
    assert res
    assert any("uncalibrated" in r["content"].lower() for r in res)


def test_corpus_excludes_retracted_findings():
    # discipline: no retracted finding may appear as grounding.
    blob = " ".join(e.topic + " " + e.content + " " + e.source for e in CORPUS).lower()
    for retracted in ("f14", "monte carlo", "multivariate", "noise floor", "gwdg precursor"):
        assert retracted not in blob, f"retracted finding leaked into corpus: {retracted}"
