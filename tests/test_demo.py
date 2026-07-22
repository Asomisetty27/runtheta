"""
Characterization tests for `theta demo` (theta/agent/demo.py).

The demo replays the REAL E009 incident export (de-identified) through the
live detectors, so these tests pin the incident the same way test_peer.py
does: the numbers below are what the real fleet produced. If a detector
change breaks them, the burden is to show the new behavior is more correct
than the pinned incident — not to re-baseline.
"""

import pytest
from typer.testing import CliRunner

from theta.agent.demo import load_fixture, run_replay



@pytest.fixture(scope="module")
def fixture():
    return load_fixture()


@pytest.fixture(scope="module")
def result(fixture):
    return run_replay(fixture)


def test_fixture_shape_and_deidentification(fixture):
    assert len(fixture["fleet"]) == 64
    nodes = {k.split(":")[0] for k in fixture["fleet"]}
    assert len(nodes) == 8

    # Pseudonymity is structural: every GPU key is a node-NN:D pseudonym, and
    # the fixture says so. (The against-the-raw-export identifier scan lives
    # with the derivation script, alongside the private export it reads.)
    import re
    assert all(re.fullmatch(r"node-\d{2}:\d", k) for k in fixture["fleet"])
    assert "de-identified" in fixture["about"]


def test_temperature_sees_nothing(fixture, result):
    # The incident's whole point: zero GPUs over the absolute alert threshold.
    assert result.temp_alerts == 0
    assert fixture["alert_threshold_c"] == 85.0


def test_replay_reproduces_the_three_blind_flags(result):
    # E009 pinned outcome: exactly 3 units at robust-z >= 3, in this z-order.
    assert result.flagged == ["node-05:7", "node-03:6", "node-05:2"]

    # Pinned robust-z magnitudes from the real fleet (position-conditioned
    # median polish). Loose tolerance: the shape, not the last digit.
    assert result.polish_z["node-05:7"] == pytest.approx(14.6, abs=0.5)
    assert result.polish_z["node-03:6"] == pytest.approx(4.0, abs=0.3)
    assert result.polish_z["node-05:2"] == pytest.approx(3.0, abs=0.3)


def test_ground_truth_is_a_subset_of_flags(result):
    # Every operator-confirmed unit must be among the flags (else the demo
    # would be claiming credit for a catch the detector didn't make).
    assert result.rma_replaced in result.flagged
    assert result.field_reconfirmed in result.flagged
    assert set(result.unconfirmed) <= set(result.flagged)
    # The honesty split: exactly one RMA replacement, one field
    # re-confirmation, one open flag. Do NOT merge the two confirmations —
    # only the 72C unit was replaced; the extreme unit is re-measured, not
    # RMA'd (see vault F7/F15).
    assert result.rma_replaced == "node-03:6"
    assert result.field_reconfirmed == "node-05:7"
    assert len(result.unconfirmed) == 1


def test_within_node_catches_only_the_extreme_unit(result):
    # Documented asymmetry (fleet-scan docstring: 3/3 vs 1/3): a single node's
    # within-node comparison catches only the +14.6σ unit.
    assert list(result.within_node_flags) == ["node-05:7"]


def test_cli_demo_runs_clean():
    from theta.cli import app

    res = CliRunner().invoke(app, ["demo"])
    assert res.exit_code == 0, res.output
    assert "receipt" in res.output
    # The honest-scope panel must always ship (rich wraps lines inside the
    # panel, so match on whitespace-normalized text).
    flat = " ".join(res.output.split())
    assert "one fleet, one incident" in flat
    # The two confirmations must never be merged into "2 RMA'd".
    assert "two different ways" in flat
    # The no-prediction disclaimer must always ship.
    assert "Condition, not prophecy" in flat
