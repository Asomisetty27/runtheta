"""`theta characterize` — HTML fleet characterization report.

Synthetic multi-node fleet with one hot unit: the report must flag it,
attribute it, render the R_θ(P) tier curve, and stay well-formed
(escaped) HTML. No GPU, no network — per the repo rule.
"""

import math

from theta.agent.characterize import POWER_TIERS, characterize, power_tier_curve

AMBIENT = 25.0


def synth_fleet(n_nodes=2, n_ords=4, bad_key=("n0", 2), n_samples=400):
    """Two-power-tier synthetic job: healthy R_θ ≈ 0.060, bad unit +45%."""
    aligned = {}
    for n in range(n_nodes):
        for o in range(n_ords):
            key = (f"n{n}", o)
            r_true = 0.060 * (1.45 if key == bad_key else 1.0)
            T, P, U, t = [], [], [], []
            for i in range(n_samples):
                # alternate sustained load between two tiers so the curve
                # and the α/β fit both have power range to work with
                power = 300.0 if (i // 50) % 2 == 0 else 620.0
                power += 3.0 * math.sin(i * 0.7)
                T.append(AMBIENT + r_true * power + 0.2 * math.sin(i))
                P.append(power)
                U.append(0.95)
                t.append(1000.0 + i * 30.0)
            aligned[key] = {"t": t, "T": T, "P": P, "U": U}
    return aligned


def test_power_tier_curve_bins_and_orders():
    tiers = power_tier_curve(synth_fleet())
    assert len(tiers) == 2                       # 250–400 and 550–700 tiers hit
    labels = [t.label for t in tiers]
    assert labels == ["250–400 W", "550–700 W"]
    # R_θ here is constant-by-construction, so medians land near 0.060
    for t in tiers:
        assert 0.05 < t.median_r < 0.10
        assert t.n >= 100


def test_characterize_flags_bad_unit_and_renders_html():
    aligned = synth_fleet()
    report, attributions, doc = characterize(
        aligned, label="synthetic <fleet>", generated_at="2026-07-04 12:00 UTC"
    )

    # The degraded unit is detected and attributed.
    assert "n0:2" in report.flagged
    assert any(a.key == "n0:2" and a.tier == "FLAG" for a in attributions)

    # Self-contained document with every section present.
    assert doc.startswith("<!doctype html>")
    for marker in ("Theta fleet characterization", "peer-relative outliers",
                   "power-tier curve", "Per-GPU characterization", "<svg", "n0:2"):
        assert marker in doc
    # Label is escaped, not injected.
    assert "synthetic &lt;fleet&gt;" in doc
    assert "synthetic <fleet>" not in doc

    # Healthy peers are present and nominal.
    assert doc.count("nominal") >= 6


def test_tiers_constant_defined_sanely():
    los = [lo for lo, _ in POWER_TIERS]
    assert los == sorted(los)
