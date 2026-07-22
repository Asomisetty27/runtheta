"""Lead-time instruments (leadtime_precursor_program 2026-07).

Pins the two properties that make the instruments safe to ship:

1. Probe-once semantics: an unsupported query is disabled after its first
   failure and never retried — a consumer SKU or old driver must not pay a
   failing NVML call every tick, and a disabled instrument must not resurrect.
2. Zero-default honesty: unsupported channels stay None/0 in RawSample.
   mem_temp_c especially must never fabricate 0.0 °C — a fake reading would
   poison R_pkg = (T_mem − T_edge)/P downstream (F22).

Per the no-GPU rule, NVML is never touched: these test the collector's own
bookkeeping plus dataclass defaults.
"""

from theta.agent.collector import CollectorConfig, NVMLCollector
from theta.agent.metrics import RawSample


def _minimal_sample(**overrides):
    base = dict(
        gpu_index=0, timestamp=0.0, temp_junction=50.0, power_w=100.0,
        util_pct=50.0, mem_util_pct=10.0, perf_state=0,
        clock_sm_mhz=1000, clock_mem_mhz=800,
    )
    base.update(overrides)
    return RawSample(**base)


class TestRawSampleDefaults:
    def test_instruments_default_to_absent(self):
        s = _minimal_sample()
        assert s.mem_temp_c is None          # never a fabricated temperature
        assert s.pcie_replay_counter == 0
        assert s.pcie_link_width == 0 and s.pcie_link_width_max == 0
        assert s.pcie_link_gen == 0 and s.pcie_link_gen_max == 0
        assert s.nvlink_errors == 0

    def test_instruments_carry_values(self):
        s = _minimal_sample(
            mem_temp_c=67.0, pcie_replay_counter=12,
            pcie_link_width=8, pcie_link_width_max=16,
            pcie_link_gen=3, pcie_link_gen_max=4, nvlink_errors=5,
        )
        assert s.mem_temp_c == 67.0
        # the downtrain signature the detachment ladder watches for:
        assert s.pcie_link_width < s.pcie_link_width_max
        assert s.pcie_link_gen < s.pcie_link_gen_max


class TestProbeOnceSemantics:
    def _collector(self):
        return NVMLCollector(CollectorConfig())

    def test_instruments_start_enabled(self):
        c = self._collector()
        for name in ("mem_temp", "pcie", "nvlink"):
            assert c._instrument_ok(0, name)

    def test_disable_is_sticky_and_per_slot(self):
        c = self._collector()
        c._instrument_disable(1, "mem_temp")
        assert not c._instrument_ok(1, "mem_temp")
        # never resurrects
        c._instrument_disable(1, "mem_temp")
        assert not c._instrument_ok(1, "mem_temp")
        # other slots and other instruments unaffected
        assert c._instrument_ok(0, "mem_temp")
        assert c._instrument_ok(1, "pcie")
