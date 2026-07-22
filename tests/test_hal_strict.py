"""Tests for HAL strict mode (allow_demo=False) and vendor-agnostic name probe.

Pins the 2026-07-13 AMD-path audit fixes:
  1. select_collector(allow_demo=False) must RAISE when no real backend exists,
     instead of silently returning NVMLCollector demo mode — demo-mode
     calibration writes plausible-looking but meaningless thresholds to disk.
  2. probe_gpu_name() must degrade to None (never raise) with no backend, so
     callers fall back to a placeholder name rather than crashing — and must
     not be pynvml-only, or AMD hosts silently anchor the default hw profile
     instead of the bundled vendor profile (e.g. mi300x).
"""
import pytest

from theta.agent import hal
from theta.agent.collector import CollectorConfig, NVMLCollector


@pytest.fixture
def no_backends(monkeypatch):
    """Simulate a host where neither pynvml nor amdsmi can talk to a driver."""
    monkeypatch.setattr(hal, "_nvml_available", lambda: False)
    monkeypatch.setattr(hal, "_rocm_available", lambda: False)


def test_auto_detect_falls_back_to_demo_by_default(no_backends):
    # The historical (CI / site-only) behavior is preserved by default.
    coll = hal.select_collector(CollectorConfig(interval_sec=1.0))
    assert isinstance(coll, NVMLCollector)


def test_strict_mode_refuses_demo_fallback(no_backends):
    with pytest.raises(RuntimeError, match="demo mode"):
        hal.select_collector(CollectorConfig(interval_sec=1.0), allow_demo=False)


def test_strict_mode_returns_real_backend_when_available(monkeypatch):
    # With NVML "available", strict mode must return the real collector, not raise.
    monkeypatch.setattr(hal, "_nvml_available", lambda: True)
    monkeypatch.setattr(hal, "_rocm_available", lambda: False)
    coll = hal.select_collector(CollectorConfig(interval_sec=1.0), allow_demo=False)
    assert isinstance(coll, NVMLCollector)


def test_strict_mode_does_not_shadow_prefer(no_backends):
    # prefer="demo" is an explicit operator request — allow_demo only governs
    # the silent auto-detect fall-through, not explicit demo selection.
    coll = hal.select_collector(
        CollectorConfig(interval_sec=1.0), prefer="demo", allow_demo=False)
    assert getattr(coll, "_demo_mode", False) is True


def test_probe_gpu_name_none_without_backends(no_backends):
    assert hal.probe_gpu_name(0) is None


def test_probe_gpu_name_uses_amdsmi_on_amd_host(monkeypatch):
    """On an AMD-only host the probe must return the amdsmi market name —
    this is what lets hw_profiles resolve 'mi300x' instead of the default."""
    monkeypatch.setattr(hal, "_nvml_available", lambda: False)
    monkeypatch.setattr(hal, "_rocm_available", lambda: True)

    class _FakeAmdsmi:
        @staticmethod
        def amdsmi_init():
            pass

        @staticmethod
        def amdsmi_shut_down():
            pass

        @staticmethod
        def amdsmi_get_processor_handles():
            return ["h0"]

        @staticmethod
        def amdsmi_get_gpu_asic_info(handle):
            return {"market_name": "AMD Instinct MI300X OAM"}

    import sys
    monkeypatch.setitem(sys.modules, "amdsmi", _FakeAmdsmi())
    assert hal.probe_gpu_name(0) == "AMD Instinct MI300X OAM"

    # And that name must actually resolve the mi300x profile end-to-end.
    from theta.agent.hw_profiles import resolve_or_default
    profile = resolve_or_default("AMD Instinct MI300X OAM")
    assert profile.canonical_name == "MI300X-OAM"


def test_probe_gpu_name_index_out_of_range(monkeypatch):
    monkeypatch.setattr(hal, "_nvml_available", lambda: False)
    monkeypatch.setattr(hal, "_rocm_available", lambda: True)

    class _FakeAmdsmi:
        @staticmethod
        def amdsmi_init():
            pass

        @staticmethod
        def amdsmi_shut_down():
            pass

        @staticmethod
        def amdsmi_get_processor_handles():
            return ["h0"]

    import sys
    monkeypatch.setitem(sys.modules, "amdsmi", _FakeAmdsmi())
    assert hal.probe_gpu_name(7) is None
