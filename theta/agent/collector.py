"""
Async GPU telemetry collector via pynvml.

pynvml calls are synchronous C library wrappers — they block the event loop.
All NVML queries are offloaded to threads via asyncio.to_thread() per the
recommendation from monitoring agent best practices (2026).

One collector instance per process. GPU handles are cached after init.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Optional

try:
    import pynvml
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False

from .metrics import RawSample

log = logging.getLogger(__name__)

# Power-plausibility guard thresholds. A GPU under sustained heavy load must
# draw a substantial fraction of its TDP; a reading far below that is almost
# certainly a per-die NVML under-report on a dual-die part (e.g. B200 reports
# one die's ~450 W of a ~900 W module), which would halve the R_θ denominator
# and double R_θ — a spurious degradation signal.
UNDERREPORT_LOAD_UTIL = 80.0   # only judge at clearly-heavy load
UNDERREPORT_TDP_FRAC  = 0.5    # below 50% of TDP at heavy load ⇒ suspect


def power_reading_suspect(
    power_w: float,
    util_pct: float,
    idle_floor_w: float,
    tdp_w: float,
) -> Optional[str]:
    """Return a reason string if a power reading is implausibly low, else None.

    Two tiers, both targeting per-die NVML under-reporting on dual-die GPUs:
      - "near_zero": power below 40% of the idle floor while active — one die
        reads ~0 while the GPU is clearly working.
      - "below_tdp_floor_at_load": power below 50% of TDP at sustained heavy
        load — the fixed idle-floor gate (34 W on a B200) is far too low to
        catch a 450 W per-die report on a 1000 W part, so scale to TDP.

    A True result only ever DROPS the sample (R_θ not computed this tick); it
    never raises an alert and cannot mask real degradation, because degradation
    raises R_θ without lowering power draw.
    """
    if power_w < idle_floor_w * 0.4 and util_pct > 15.0:
        return "near_zero"
    if tdp_w > 0.0 and util_pct >= UNDERREPORT_LOAD_UTIL and power_w < tdp_w * UNDERREPORT_TDP_FRAC:
        return "below_tdp_floor_at_load"
    return None


@dataclass
class CollectorConfig:
    interval_sec: float = 5.0        # sample every N seconds
    gpu_indices: Optional[list[int]] = None  # None = all GPUs
    # Synthetic samples are useful for interactive demos and tests, but a
    # caller that will alert or export metrics can refuse any NVML fallback.
    allow_demo: bool = True


class NVMLCollector:
    """
    Async GPU telemetry collector.

    Usage:
        async with NVMLCollector(config) as collector:
            async for sample in collector.stream():
                process(sample)
    """

    # HAL protocol: vendor identity for downstream module routing
    vendor: str = "nvidia"

    def __init__(self, config: CollectorConfig):
        self.config  = config
        self._allow_demo = config.allow_demo
        self._handles: list  = []
        self._n_gpus: int    = 0
        self._demo_mode: bool = not NVML_AVAILABLE
        self._gpu_names: list[str] = []  # populated in _init_nvml
        self._caps: list = []            # DeviceCapability per slot (MIG/vGPU), set in _init_nvml
        # Per-slot failure tracking for self-healing handle reinit
        self._failure_counts: dict[int, int] = {}
        self._failure_threshold: int = 3  # consecutive misses before reinit
        # GPU_IS_LOST quarantine: a GPU that has fallen off the bus is polled
        # only every _lost_probe_every ticks (NVML calls against a lost device
        # can hang or spam) — the healthy peers keep full-rate monitoring.
        self._lost_gpus: dict[int, int] = {}   # slot → ticks since quarantine
        self._lost_probe_every: int = 60       # ≈5 min at the 5s default interval
        # Lead-time instruments that this (slot, driver, SKU) combination has
        # proven unsupported — disabled after first failure, never retried.
        self._instruments_dead: set[tuple[int, str]] = set()

    def _instrument_ok(self, idx: int, name: str) -> bool:
        return (idx, name) not in self._instruments_dead

    def _instrument_disable(self, idx: int, name: str) -> None:
        if (idx, name) not in self._instruments_dead:
            self._instruments_dead.add((idx, name))
            log.info("instrument_unsupported", gpu=idx, instrument=name,
                     note="query unsupported on this driver/SKU — disabled for this run")

    @property
    def capabilities(self) -> list:
        """HAL: per-slot DeviceCapability (MIG/vGPU mode, R_θ computability)."""
        return list(self._caps)

    @property
    def gpu_count(self) -> int:
        """HAL protocol: number of GPUs this collector is monitoring."""
        return len(self._handles) if self._handles else self._n_gpus

    @property
    def gpu_names(self) -> list[str]:
        """HAL protocol: friendly model names, indexed by slot.

        In demo mode returns placeholder Tesla T4 names so hw_profiles
        resolution still works through to the measured profile.
        """
        if self._gpu_names:
            return list(self._gpu_names)
        if self._demo_mode:
            return ["Tesla T4"] * self._n_gpus
        return ["unknown"] * self._n_gpus

    async def __aenter__(self) -> "NVMLCollector":
        await asyncio.to_thread(self._init_nvml)
        return self

    async def __aexit__(self, *_) -> None:
        if not self._demo_mode:
            await asyncio.to_thread(self._shutdown_nvml)

    def _init_nvml(self) -> None:
        if self._demo_mode:
            if not self._allow_demo:
                raise RuntimeError(
                    "NVIDIA telemetry is unavailable and demo mode was not requested. "
                    "Repair the NVIDIA driver or rerun with --demo for synthetic telemetry."
                )
            log.warning("pynvml not available — running in demo mode with synthetic data")
            self._n_gpus = 4
            return
        try:
            pynvml.nvmlInit()
        except pynvml.NVMLError as exc:
            # pynvml is installed but the NVIDIA driver / library is absent
            # (common on macOS or CPU-only Linux boxes). Production callers
            # must fail here rather than turning a driver outage into a fake,
            # healthy T4 fleet.
            if not self._allow_demo:
                raise RuntimeError(
                    "NVIDIA telemetry initialization failed and demo mode was not requested. "
                    "Repair the NVIDIA driver or rerun with --demo for synthetic telemetry."
                ) from exc
            log.warning("NVML library not found — running in demo mode with synthetic data")
            self._demo_mode = True
            self._n_gpus = 4
            return
        self._n_gpus = pynvml.nvmlDeviceGetCount()
        indices = self.config.gpu_indices or list(range(self._n_gpus))
        self._handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in indices]
        # Populate GPU names for hw_profiles resolution downstream
        names: list[str] = []
        for h in self._handles:
            try:
                name = pynvml.nvmlDeviceGetName(h)
                if isinstance(name, bytes):
                    name = name.decode("utf-8", errors="replace")
                names.append(name)
            except Exception:
                names.append("unknown")
        self._gpu_names = names

        # Probe MIG/vGPU capabilities per device so downstream knows whether R_θ
        # is computable and what it means (per-physical-die under MIG, possibly
        # unavailable under vGPU). Best-effort; never fails init.
        from .device_caps import DeviceMode, probe_capability
        self._caps = []
        for slot, h in enumerate(self._handles):
            try:
                cap = probe_capability(pynvml, h)
            except Exception:
                from .device_caps import DeviceCapability
                cap = DeviceCapability(DeviceMode.UNKNOWN, True, True, True,
                                       "capability probe failed — assuming physical")
            self._caps.append(cap)
            if cap.mode is not DeviceMode.PHYSICAL or not cap.rtheta_computable:
                log.warning("device_capability", slot=slot,
                            name=names[slot] if slot < len(names) else "?",
                            mode=cap.mode.value, rtheta_computable=cap.rtheta_computable,
                            note=cap.note)
        log.info("NVML initialized", extra={"n_gpus": len(self._handles)})

    def _shutdown_nvml(self) -> None:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass

    def _collect_one(self, idx: int, handle) -> RawSample:
        """Synchronous — called via asyncio.to_thread()."""
        t0     = time.time()
        temp   = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
        power  = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # mW → W
        util   = pynvml.nvmlDeviceGetUtilizationRates(handle)
        pstate = pynvml.nvmlDeviceGetPerformanceState(handle)

        try:
            sm_mhz  = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM)
            mem_mhz = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_MEM)
        except Exception:
            sm_mhz = mem_mhz = 0

        try:
            fan = pynvml.nvmlDeviceGetFanSpeed(handle)
        except pynvml.NVMLError:
            fan = None

        # Silicon-level health metrics — each wrapped independently so a single
        # unsupported query on older drivers doesn't drop the whole sample
        try:
            ecc_sbit = pynvml.nvmlDeviceGetTotalEccErrors(
                handle, pynvml.NVML_SINGLE_BIT_ECC, pynvml.NVML_VOLATILE_ECC
            )
        except pynvml.NVMLError:
            ecc_sbit = 0

        try:
            ecc_dbit = pynvml.nvmlDeviceGetTotalEccErrors(
                handle, pynvml.NVML_DOUBLE_BIT_ECC, pynvml.NVML_VOLATILE_ECC
            )
        except pynvml.NVMLError:
            ecc_dbit = 0

        try:
            throttle_reasons = pynvml.nvmlDeviceGetCurrentClocksThrottleReasons(handle)
        except pynvml.NVMLError:
            throttle_reasons = 0

        try:
            sm_clock_max_mhz = pynvml.nvmlDeviceGetMaxClockInfo(handle, pynvml.NVML_CLOCK_SM)
        except pynvml.NVMLError:
            sm_clock_max_mhz = 0

        # Lead-time instruments (leadtime_precursor_program 2026-07) — probed
        # once per slot; a query the driver/SKU doesn't support is disabled
        # for the rest of the run instead of raising every tick.
        mem_temp_c = None
        if self._instrument_ok(idx, "mem_temp"):
            try:
                fid = getattr(pynvml, "NVML_FI_DEV_MEMORY_TEMP", 140)
                fv = pynvml.nvmlDeviceGetFieldValues(handle, [fid])[0]
                raw_mt = fv.value.uiVal if getattr(fv, "nvmlReturn", 1) == 0 else 0
                # 0 and the int64 blank-sentinel both mean "not exposed"
                if 0 < raw_mt < 150:
                    mem_temp_c = float(raw_mt)
                else:
                    self._instrument_disable(idx, "mem_temp")
            except Exception:
                self._instrument_disable(idx, "mem_temp")

        pcie_replay = link_w = link_w_max = link_g = link_g_max = 0
        if self._instrument_ok(idx, "pcie"):
            try:
                pcie_replay = int(pynvml.nvmlDeviceGetPcieReplayCounter(handle))
                link_w      = int(pynvml.nvmlDeviceGetCurrPcieLinkWidth(handle))
                link_w_max  = int(pynvml.nvmlDeviceGetMaxPcieLinkWidth(handle))
                link_g      = int(pynvml.nvmlDeviceGetCurrPcieLinkGeneration(handle))
                link_g_max  = int(pynvml.nvmlDeviceGetMaxPcieLinkGeneration(handle))
            except pynvml.NVMLError:
                self._instrument_disable(idx, "pcie")

        nvlink_total = 0
        if self._instrument_ok(idx, "nvlink"):
            try:
                for _link in range(6):    # 6 links covers pre-Hopper; unsupported links raise and disable cleanly
                    for _ctr in (0, 1):   # CRC FLIT + CRC DATA counters
                        nvlink_total += int(
                            pynvml.nvmlDeviceGetNvLinkErrorCounter(handle, _link, _ctr))
            except pynvml.NVMLError:
                nvlink_total = 0
                self._instrument_disable(idx, "nvlink")

        # Power sanity check — B200 and other dual-die/multi-chip GPUs can
        # report per-die power via some NVML versions while T_junction reflects
        # the hotter die. If reported power is suspiciously low while the GPU
        # is clearly active, the R_θ denominator will be wrong. Log and clamp
        # to None (enrich() will mark rtheta_valid=False) rather than silently
        # emitting a bogus metric.
        power_f = float(power)
        _util_pct = float(util.gpu)
        try:
            from .hw_profiles import resolve_or_default as _rp
            _prof = _rp(self._gpu_names[idx] if idx < len(self._gpu_names) else "")
            _idle_floor = _prof.idle_floor_w if _prof else 5.0
            _tdp = _prof.tdp_w if _prof else 0.0
        except Exception:
            _idle_floor, _tdp = 5.0, 0.0

        _suspect = power_reading_suspect(power_f, _util_pct, _idle_floor, _tdp)
        if _suspect is not None:
            log.warning(
                "power_reading_suspect",
                gpu=idx,
                power_w=power_f,
                util_pct=_util_pct,
                idle_floor_w=_idle_floor,
                tdp_w=_tdp,
                reason=_suspect,
                note="power implausibly low for load — possible per-die NVML reporting on dual-die GPU; skipping R_θ for this sample",
            )
            power_f = 0.0   # drives rtheta_valid=False in enrich()

        return RawSample(
            gpu_index        = idx,
            timestamp        = time.time(),
            temp_junction    = float(temp),
            power_w          = power_f,
            util_pct         = float(util.gpu),
            mem_util_pct     = float(util.memory),
            perf_state       = int(str(pstate).replace("PerformanceState_", "").replace("P", "")),
            clock_sm_mhz     = sm_mhz,
            clock_mem_mhz    = mem_mhz,
            fan_speed_pct    = float(fan) if fan is not None else None,
            ecc_sbit         = int(ecc_sbit),
            ecc_dbit         = int(ecc_dbit),
            throttle_reasons = int(throttle_reasons),
            sm_clock_max_mhz = sm_clock_max_mhz,
            poll_latency_s   = time.time() - t0,
            mem_temp_c          = mem_temp_c,
            pcie_replay_counter = pcie_replay,
            pcie_link_width     = link_w,
            pcie_link_width_max = link_w_max,
            pcie_link_gen       = link_g,
            pcie_link_gen_max   = link_g_max,
            nvlink_errors       = nvlink_total,
        )

    def _collect_demo(self, idx: int) -> RawSample:
        """Synthetic data for development / CI without a GPU."""
        import math
        t = time.time()
        phase = (t % 300) / 300   # 5 min cycle

        if phase < 0.2:            # idle
            temp, power, util, ps = 42.0, 11.4, 0.0, 8
        elif phase < 0.5:          # load
            temp, power, util, ps = 70.0, 68.0, 97.0, 0
        elif phase < 0.6:          # transition
            temp, power, util, ps = 80.0, 31.2, 0.0, 0  # zombie-like
        else:                      # recovery
            temp = 42.0 + 20.0 * math.exp(-(phase - 0.6) * 10)
            power, util, ps = 11.4, 0.0, 8

        noise = 0.5 * math.sin(t * 7.3 + idx)
        sm_max = 1980   # T4 boost clock
        sm_cur = 1600 if ps == 0 else 300
        return RawSample(
            gpu_index        = idx,
            timestamp        = t,
            temp_junction    = temp + noise,
            power_w          = power + abs(noise) * 0.3,
            util_pct         = util,
            mem_util_pct     = util * 0.6,
            perf_state       = ps,
            clock_sm_mhz     = sm_cur,
            clock_mem_mhz    = 8000 if ps == 0 else 405,
            fan_speed_pct    = 40.0 + temp * 0.3,
            ecc_sbit         = 0,
            ecc_dbit         = 0,
            throttle_reasons = 0,
            sm_clock_max_mhz = sm_max,
        )

    async def collect_all(self) -> list[RawSample]:
        """Collect one sample from all monitored GPUs concurrently.

        Resilience: per-GPU collection failures are isolated — one bad
        handle never drops the whole tick. After a configurable run of
        consecutive failures on a single GPU, the handle is re-initialized
        (NVMLError can be transient: driver reset, brief PCIe hang, etc.).
        Re-init failures are logged but never raise — the GPU simply
        remains absent from this tick's samples.
        """
        if self._demo_mode:
            n = self.config.gpu_indices or list(range(self._n_gpus))
            return [self._collect_demo(i) for i in (n if isinstance(n, list) else range(n))]

        polled = [
            (slot, handle)
            for slot, handle in enumerate(self._handles)
            if self._should_poll(slot)
        ]
        tasks = [asyncio.to_thread(self._collect_one, slot, handle)
                 for slot, handle in polled]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        samples = []
        for (slot, _), r in zip(polled, results, strict=True):
            if isinstance(r, Exception):
                self._handle_collect_error(slot, r)
            else:
                # Successful sample — clear strike counter and any quarantine
                self._failure_counts.pop(slot, None)
                if slot in self._lost_gpus:
                    log.warning("gpu=%d recovered — leaving GPU_IS_LOST quarantine", slot)
                    del self._lost_gpus[slot]
                samples.append(r)
        return samples

    def _should_poll(self, slot: int) -> bool:
        """Healthy slots poll every tick; quarantined slots probe periodically."""
        if slot not in self._lost_gpus:
            return True
        self._lost_gpus[slot] += 1
        return self._lost_gpus[slot] % self._lost_probe_every == 0

    def _handle_collect_error(self, slot: int, exc: Exception) -> None:
        """Classify a per-GPU failure: quarantine lost devices, strike-count the rest.

        NVML_ERROR_GPU_IS_LOST means the device fell off the bus — reinit at
        tick rate would hang or spam against dead hardware, so the slot is
        quarantined and probed only every _lost_probe_every ticks while the
        healthy peers keep full-rate monitoring. Everything else keeps the
        transient strike/reinit path.
        """
        if NVML_AVAILABLE and isinstance(exc, pynvml.NVMLError) and \
                getattr(exc, "value", None) == pynvml.NVML_ERROR_GPU_IS_LOST:
            if slot not in self._lost_gpus:
                log.error(
                    "gpu=%d fell off the bus (NVML_ERROR_GPU_IS_LOST) — quarantined, "
                    "probing every %d ticks; peers unaffected",
                    slot, self._lost_probe_every,
                )
                self._lost_gpus[slot] = 0
            self._failure_counts.pop(slot, None)
            return

        # Transient-class failure — re-init the handle after N strikes
        self._failure_counts[slot] = self._failure_counts.get(slot, 0) + 1
        if self._failure_counts[slot] >= self._failure_threshold:
            log.warning(
                "collector reinit gpu=%d after %d consecutive failures: %s",
                slot, self._failure_counts[slot], exc,
            )
            self._try_reinit_handle(slot)
        else:
            log.error("collection error gpu=%d (%d/%d): %s",
                      slot, self._failure_counts[slot],
                      self._failure_threshold, exc)

    def _try_reinit_handle(self, slot: int) -> None:
        """Attempt to re-acquire a single GPU's NVML handle. Best-effort."""
        try:
            indices = self.config.gpu_indices or list(range(self._n_gpus))
            if slot < len(indices):
                new_handle = pynvml.nvmlDeviceGetHandleByIndex(indices[slot])
                self._handles[slot] = new_handle
                log.info("collector reinit gpu=%d successful", slot)
                # Reset strike counter on successful reinit so we get another
                # full window of attempts before giving up again.
                self._failure_counts.pop(slot, None)
        except Exception as exc:
            log.error("collector reinit gpu=%d failed: %s", slot, exc)
            # Don't reset counter — leave it pinned so we don't reinit-loop
            # at 5s intervals. Operator restart will be needed if persistent.

    async def stream(self):
        """Yield batches of samples on every interval tick."""
        while True:
            t0 = asyncio.get_event_loop().time()
            samples = await self.collect_all()
            for s in samples:
                yield s
            elapsed = asyncio.get_event_loop().time() - t0
            await asyncio.sleep(max(0.0, self.config.interval_sec - elapsed))
