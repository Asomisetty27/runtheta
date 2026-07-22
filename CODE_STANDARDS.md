# Code Standards — Theta

The engineering standard every new module ships against, and the bar existing
code is migrated toward. CI enforces the mechanical parts; this file records
the judgment calls and the **deliberate deviations**, so compliance never
degrades into cargo-culting.

## 1. Hardware boundary: FFI, not subprocesses

- All NVIDIA telemetry goes through **`nvidia-ml-py` (pynvml)** — which *is* a
  ctypes binding to `libnvidia-ml.so`, maintained by NVIDIA. We do **not**
  hand-roll our own ctypes/cffi layer: it would re-implement NVIDIA's binding
  with more bugs and identical FFI cost. AMD goes through `amdsmi`
  (`rocm_collector.py`), BMC via Redfish. New vendors plug in behind `hal.py`.
- **No subprocess telemetry.** Shelling out to `nvidia-smi` in a poll loop is
  banned (fork/exec cost, parse fragility).
- Blocking C calls never run on the event loop: `asyncio.to_thread()` /
  executors only (`collector.py` is the reference).
- **Deviation (documented):** we do not chase "zero-allocation, sub-millisecond
  polling." Thermal signals have multi-second time constants — E009 showed 30s
  production sampling suffices, and junction transients are unresolvable at
  any Python-achievable rate anyway. The 5s default interval leaves allocation
  overhead ~6 orders of magnitude below budget. Optimize where the physics
  says it matters.

## 2. Concurrency and bounded memory

- One asyncio event loop; blocking work in threads; per-GPU collection is
  concurrent and failure-isolated (one bad handle never drops a tick).
- **Every buffer is bounded.** `deque(maxlen=...)` or explicit window
  trimming *plus* a maxlen backstop (see `baseline.py`, `calibrate.py`).
  Under pressure we drop old telemetry, never OOM the node.

## 3. Numerics

- Fleet-scale math is **vectorized NumPy with in-place ops** — `peer.py`'s
  median polish is the reference (`np.subtract(..., out=m)`, NaN-masked
  sparse cells). Measured: 2.9× at 1024 GPUs, 5× at 4096 vs pure Python,
  bit-identical outputs, pinned by characterization tests.
- Vectorization must never change results: pin the old behavior with a test
  *before* rewriting, and benchmark honestly (including where numpy is
  slower — small-N overhead is real and acceptable at service cadence).

## 4. Fault tolerance and sandboxing

- NVML failures are **classified, not blanket-caught**:
  - `NVML_ERROR_GPU_IS_LOST` → quarantine the slot, probe every
    `_lost_probe_every` ticks, peers keep full-rate monitoring.
  - Transient errors → strike counter → single handle re-init.
  - Per-query optional metrics (fan, ECC on old drivers) degrade to `None`,
    never drop the sample.
- Every failure mode above has a **mock-NVML test**
  (`tests/test_collector_lost_gpu.py`) — no GPU required, per the repo rule.
- systemd deployment is resource-capped and sandboxed
  (`deploy/theta-monitor.service`): `CPUQuota=25%`, `CPUWeight=20`,
  `MemoryMax=512M`, `NoNewPrivileges`, `ProtectSystem=full`, etc. The agent
  may never starve the workloads it watches. **Deviation:** no
  `PrivateDevices`/`DevicePolicy` — NVML needs `/dev/nvidia*`.
- Kubernetes deployment mirrors this: non-root numeric UID, dropped
  capabilities, resource requests/limits (`deploy/helm/theta`).

## 5. Tooling gates (CI-enforced)

- **ruff**: `E, W, F, N, B, SIM, NPY, I` — config in `pyproject.toml`.
  Deviations, with reasons recorded there: `E501`, `N806` (physics notation),
  `SIM105` (explicit try/except in fault paths).
- **mypy**: gradual strictness (`mypy.ini`). The global gate is pragmatic;
  the per-module strict list (`disallow_untyped_defs`) grows monotonically —
  **every new module ships strict from day one and is added to the list.**
  Seeded with `theta/tui.py`, `theta/agent/peer.py`.
- **pytest**: no test may require GPU hardware. Driver failures, vendor
  quirks, and skewed thermal responses are simulated (mock NVML, `sim/`,
  recorded real exports). Characterization tests pin real incidents (E009);
  breaking one requires showing the new behavior is *more* correct.
- Public interfaces use `typing.Protocol` structural contracts
  (`hal.py`, `tui.FleetProvider`).

## Amendment rule

Deviations require a written reason in the config or this file. A deviation
without a reason is a bug.
