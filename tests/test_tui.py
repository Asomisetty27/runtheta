"""Headless tests for `theta top` (textual Pilot — no TTY needed).

Follows the repo's no-GPU rule: a fake provider feeds deterministic
readings, including one degraded GPU, and we assert the new interface
renders its three panels (thermal field, polish matrix, trajectories),
derives alerts, scrubs, and switches to the demo fleet.
"""

import pytest

textual = pytest.importorskip("textual")

from theta.tui import (  # noqa: E402
    DemoFleetProvider,
    FieldPanel,
    FleetSample,
    GpuReading,
    PolishPanel,
    RemoteProvider,
    ThetaTopApp,
    TrajectoryPanel,
    derive_alerts,
)
from theta.tuifx import DemoFleet, polish, sparkline  # noqa: E402


class FakeProvider:
    """Two healthy GPUs plus one hot, drifting, unschedulable unit."""

    def __init__(self):
        self.calls = 0

    async def sample(self) -> FleetSample:
        self.calls += 1
        return FleetSample(
            source="fake",
            gpus=[
                GpuReading(index=0, name="Tesla T4", temp_c=61.0, power_w=68.0,
                           util_pct=95.0, rtheta=0.53, drift_sigma=0.2,
                           clock_eff=0.99, schedulable=True),
                GpuReading(index=1, name="Tesla T4", temp_c=63.0, power_w=69.0,
                           util_pct=94.0, rtheta=0.55, drift_sigma=0.4,
                           clock_eff=0.98, schedulable=True),
                GpuReading(index=2, name="Tesla T4", temp_c=88.0, power_w=68.5,
                           util_pct=96.0, rtheta=0.92, drift_sigma=4.2,
                           clock_eff=0.78, schedulable=False),
            ],
        )


@pytest.mark.asyncio
async def test_top_renders_panels_alerts_and_scrub(tmp_path):
    app = ThetaTopApp(provider=FakeProvider(), interval=60)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()

        # The three signature panels exist and rendered.
        assert app.query_one(FieldPanel) is not None
        assert app.query_one(PolishPanel) is not None
        assert app.query_one(TrajectoryPanel) is not None

        # Single node → polish panel must REFUSE node-scope verdicts (F21).
        polish_text = str(app.query_one(PolishPanel).render())
        assert "single node" in polish_text

        # History buffered for trajectories.
        assert list(app._history["gpu2"]) == [0.92]

        # Degraded unit produced alerts in the feed (drift, thermal, drain).
        alerts = derive_alerts((await app.provider.sample()).gpus[2])
        assert any("unfit" in a for a in alerts)
        assert any("drift" in a for a in alerts)

        # SVG screenshot artifact (portfolio/docs/demo).
        app.save_screenshot(str(tmp_path / "theta-top.svg"))
        assert (tmp_path / "theta-top.svg").stat().st_size > 10_000

        # Bindings: pause, scrub back into replay, return to live, demo fleet.
        await pilot.press("p")
        assert app.paused is True
        await pilot.press("p")
        await pilot.press("left")
        assert app._scrub_pos is not None
        await pilot.press("right")
        assert app._scrub_pos is None
        await pilot.press("f")
        assert isinstance(app.provider, DemoFleetProvider)
        await pilot.press("q")


@pytest.mark.asyncio
async def test_demo_fleet_polish_catches_the_planted_degradation():
    """The demo story is real math: after the victim's drift ramps in, the
    two-way polish isolates it as the fleet's dominant residual."""
    fleet = DemoFleet()
    fleet.tick(200)  # well past the ramp
    snap = fleet.sample()
    view = polish({(n, s): m["rtheta"] for (n, s), m in snap.items()})
    victim_z = view.residual_z[fleet.victim]
    others = [abs(z) for k, z in view.residual_z.items() if k != fleet.victim]
    assert victim_z > 4.0
    assert victim_z > max(others) + 2.0


def test_sparkline_shape_and_bounds():
    rows = sparkline([1.0, 2.0, 3.0, 2.5, 4.0], width=12, height=2)
    assert len(rows) == 2
    assert all(r.cell_len == 12 for r in rows)


@pytest.mark.asyncio
async def test_remote_provider_parses_prometheus_text():
    text = "\n".join([
        '# HELP theta_gpu_rtheta_cwatt R',
        'theta_gpu_rtheta_cwatt{gpu_index="0"} 0.0601',
        'theta_gpu_temperature_celsius{gpu_index="0"} 66.0',
        'theta_gpu_power_watts{gpu_index="0"} 653.0',
        'theta_gpu_utilization_ratio{gpu_index="0"} 0.97',
        'theta_gpu_schedulable{gpu_index="0"} 1',
        'theta_gpu_rtheta_cwatt{gpu_index="7"} 0.0842',
        'theta_gpu_temperature_celsius{gpu_index="7"} 80.0',
        'theta_gpu_power_watts{gpu_index="7"} 653.0',
        'theta_gpu_drift_sigma{gpu_index="7"} 15.6',
        'theta_gpu_schedulable{gpu_index="7"} 0',
    ])

    provider = RemoteProvider("localhost:9101")

    class FakeResp:
        def __init__(self, t): self.text = t

    class FakeClient:
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        async def get(self, url): return FakeResp(text)

    import httpx

    orig = httpx.AsyncClient
    httpx.AsyncClient = lambda **kw: FakeClient()
    try:
        sample = await provider.sample()
    finally:
        httpx.AsyncClient = orig

    assert [g.index for g in sample.gpus] == [0, 7]
    g7 = sample.gpus[1]
    assert g7.rtheta == pytest.approx(0.0842)
    assert g7.drift_sigma == pytest.approx(15.6)   # the E009 blind-flag z-score
    assert g7.schedulable is False
    assert any("unfit" in a for a in derive_alerts(g7))


class _OneNodeFake:
    """Stands in for a per-node RemoteProvider inside MultiNodeProvider."""

    def __init__(self, n_gpus=2, fail=False):
        self.n_gpus, self.fail = n_gpus, fail

    async def sample(self):
        if self.fail:
            raise ConnectionError("tunnel died")
        return FleetSample(
            source="fake-node",
            gpus=[GpuReading(index=i, temp_c=60 + i, power_w=300.0, rtheta=0.05)
                  for i in range(self.n_gpus)],
        )


@pytest.mark.asyncio
async def test_multinode_provider_merges_and_survives_dead_node():
    from theta.tui import MultiNodeProvider

    mp = MultiNodeProvider([("node-a", "127.0.0.1:1"), ("node-b", "127.0.0.1:2"),
                            ("node-c", "127.0.0.1:3")])
    mp._providers = [("node-a", _OneNodeFake()), ("node-b", _OneNodeFake(fail=True)),
                     ("node-c", _OneNodeFake())]

    s = await mp.sample()

    # Dead node reported, healthy nodes merged with node identity attached.
    assert len(s.gpus) == 4
    assert {g.node for g in s.gpus} == {"node-a", "node-c"}
    assert any("node-b" in a and "scrape failed" in a for a in s.alerts)

    # uids are unique across nodes even though gpu indices collide.
    uids = [g.uid for g in s.gpus]
    assert len(uids) == len(set(uids))
    assert "node-a" in s.gpus[0].label
