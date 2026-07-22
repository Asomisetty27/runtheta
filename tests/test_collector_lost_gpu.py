"""GPU_IS_LOST quarantine — the driver-failure mode the collector must survive.

Per the no-GPU rule, NVML is fully mocked: a fake pynvml module supplies the
NVMLError type and the GPU_IS_LOST error code, letting us exercise the exact
failure classification a fallen-off-the-bus device produces in production.
"""

import pytest

import theta.agent.collector as collector_mod
from theta.agent.collector import NVMLCollector


class FakeNVMLError(Exception):
    def __init__(self, value):
        super().__init__(f"NVML error {value}")
        self.value = value


class FakePynvml:
    NVMLError = FakeNVMLError
    NVML_ERROR_GPU_IS_LOST = 15
    NVML_ERROR_UNKNOWN = 999


@pytest.fixture
def patched_nvml(monkeypatch):
    monkeypatch.setattr(collector_mod, "pynvml", FakePynvml, raising=False)
    monkeypatch.setattr(collector_mod, "NVML_AVAILABLE", True)
    return FakePynvml


def make_collector():
    from theta.agent.collector import CollectorConfig
    c = NVMLCollector(CollectorConfig())
    c._handles = [object(), object(), object()]
    return c


def test_gpu_is_lost_quarantines_only_that_slot(patched_nvml):
    c = make_collector()
    c._handle_collect_error(1, FakeNVMLError(FakePynvml.NVML_ERROR_GPU_IS_LOST))

    assert 1 in c._lost_gpus
    # Quarantined slot is skipped at tick rate...
    assert c._should_poll(1) is False
    # ...while healthy peers keep polling every tick.
    assert c._should_poll(0) is True
    assert c._should_poll(2) is True


def test_quarantined_slot_probes_periodically(patched_nvml):
    c = make_collector()
    c._lost_probe_every = 5
    c._handle_collect_error(0, FakeNVMLError(FakePynvml.NVML_ERROR_GPU_IS_LOST))

    polls = [c._should_poll(0) for _ in range(10)]
    # Exactly every 5th tick probes the lost device.
    assert polls == [False, False, False, False, True,
                     False, False, False, False, True]


def test_gpu_is_lost_does_not_reinit_loop(patched_nvml):
    """A lost GPU must NOT enter the strike/reinit path (reinit against dead
    hardware at tick rate is the failure mode this change removes)."""
    c = make_collector()
    reinits = []
    c._try_reinit_handle = lambda slot: reinits.append(slot)

    for _ in range(10):
        c._handle_collect_error(2, FakeNVMLError(FakePynvml.NVML_ERROR_GPU_IS_LOST))

    assert reinits == []
    assert c._failure_counts.get(2) is None


def test_transient_errors_still_use_strike_reinit_path(patched_nvml):
    c = make_collector()
    reinits = []
    c._try_reinit_handle = lambda slot: reinits.append(slot)

    for _ in range(c._failure_threshold):
        c._handle_collect_error(0, FakeNVMLError(FakePynvml.NVML_ERROR_UNKNOWN))

    assert reinits == [0]
    assert 0 not in c._lost_gpus
