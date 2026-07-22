"""The production monitor must never silently substitute synthetic samples."""

import asyncio
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

import theta.cli as cli
from theta.agent.collector import CollectorConfig, NVMLCollector
from theta.agent.daemon import ThetaAgent


def test_monitor_passes_demo_only_when_the_flag_is_present(monkeypatch):
    captured = []

    def fake_init(self, config):
        self.config = config
        self._classifier = SimpleNamespace(mode="test")

    async def fake_run(self):
        captured.append(self.config.demo)

    monkeypatch.setattr(ThetaAgent, "__init__", fake_init)
    monkeypatch.setattr(ThetaAgent, "run", fake_run)
    runner = CliRunner()

    result = runner.invoke(cli.app, ["monitor", "--port", "0"])
    assert result.exit_code == 0, result.output
    assert captured == [False]

    result = runner.invoke(cli.app, ["monitor", "--demo", "--port", "0"])
    assert result.exit_code == 0, result.output
    assert captured == [False, True]


def test_strict_nvml_collector_refuses_startup_fallback(monkeypatch):
    """A driver failure after backend selection cannot become fake T4 data."""
    import theta.agent.collector as collector_mod

    monkeypatch.setattr(collector_mod, "NVML_AVAILABLE", False)
    collector = NVMLCollector(CollectorConfig(allow_demo=False))

    async def enter():
        async with collector:
            pass

    with pytest.raises(RuntimeError, match="demo mode was not requested"):
        asyncio.run(enter())
