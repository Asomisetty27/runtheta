# E-MR rig tooling

Capture + analysis pipeline for the **E-MR real-silicon mini-rig** (the lead-time
experiment; design in `thermalos-vault/wiki/experiments/E-MR.md`, parts in `E-MR-build.md`).

Built **ahead of the hardware**, on purpose. The E-LT precedent: build the whole
analysis so that when the rig is assembled, the only new variable is the real
degradation. A rig that arrives to a finished pipeline gives a lead-time answer in
days; a rig that arrives to no tooling gives it in weeks. The analysis core reuses the
**shipped** detector + calibration modules (`theta.agent.*`), so the lab and the
product run one implementation and a rig-validated number maps straight onto the agent.

## What's here

| File | Job |
|------|-----|
| `throttle.py` | NVML clock-reason bitmask → thermal-throttle event (ignores power-cap; requires a sustained run). The ground-truth failure label. |
| `analyze.py` | `RunTrace` → `lead_time = t_throttle − t_anomaly`, where `t_anomaly` is the **shipped** `ChannelCUSUM`. Also a baseline+k·σ sweep (E-LT framing). `synthetic_run()` for dry-runs. |
| `calibrate_from_runs.py` | Labeled run-to-throttle runs → `FailureObservation` → `calibration` (flips UNCALIBRATED→CALIBRATED) + a survival RUL fit. This is what lights up the dormant stack. |
| `colab_capture.py` | Log the full NVML surface to CSV. Run on Colab **today** to de-risk the purchase; on the rig it's the NVML half of the log. |
| `firmware/pico_ambient.py` | MicroPython for the Pico: DS18B20 air sensors + PWM fan control/tach → CSV over USB serial. The measured-ambient half of the log (and the airflow-arm knob). |
| `serial_logger.py` | Host side: read the Pico stream (`read_serial`), and join measured ambient onto the NVML capture by time (`merge_ambient` / `build_run_trace`) → a complete `RunTrace` with a REAL T_ambient. |
| `tests/test_rig.py` | Validates the whole pipeline on synthetic run-to-throttle traces + the Pico parse/merge (no hardware). |

## Do this now (before Friday's spend)

Run the capability probe on a Colab T4 to confirm what it actually exposes — we're
*assuming* it reports the throttle bitmask and a hotspot temp, and the sensing plan
depends on it. In a Colab **Python** cell, shell commands need a `!` prefix, and the
rig code must be on the path (clone the branch until it's merged):

```python
# cell 1 (shell):
!pip -q install nvidia-ml-py
!git clone -b feat/rig-pipeline https://github.com/Asomisetty27/theta.git /content/theta

# cell 2 (optional load, background):
!cd /content && git clone https://github.com/wilicc/gpu-burn.git && cd gpu-burn && make
!cd /content/gpu-burn && nohup ./gpu_burn 200 >/content/burn.log 2>&1 &

# cell 3 (capture — run from the repo root so `rig` and `theta` import):
%cd /content/theta
!python -m rig.colab_capture --seconds 150 --out /content/t4_surface.csv
```

For a pure capability check (no repo, no load needed), the standalone probe snippet
queries each NVML field once and reports availability — capability is readable at idle.

Read the printed summary. What matters:
- **throttle-reason bitmask exposed** → we can detect the ground-truth throttle. (Expected yes.)
- **hotspot temp exposed** → the F16 hotspot-vs-average arm works from NVML alone; if not, a heatsink thermocouple substitutes (already in the BOM).
- `thermal_throttle_samples: 0` is expected on Colab (Google's cooling prevents it) — this run characterizes the *sensor surface*, not degradation.

## When the rig exists

1. `colab_capture.py` logs NVML; the Pico logs measured inlet/exhaust ambient; join by timestamp into a `RunTrace`.
2. `analyze_run(trace)` → per-run lead time (shipped CUSUM) + the k·σ sweep.
3. `calibrate_from_runs(results, component, gpu_class, path=...)` after ≥5 runs → the boundary flips CALIBRATED; `fit_survival(traces)` → the real severity→RUL curve.
4. File the numbers back into `E-MR.md` (Results/Findings) and open F-pages.

## The honest property the tooling surfaces

The shipped CUSUM is tuned conservatively (ARL0 ~6000) so healthy hardware never
false-alarms. On a **slow** arm (TIM / fan-duty) it has runway and fires well before
throttle. On a **fast acute fault** (fan yank) it can fire *late* — so the analyzer also
reports the sensitive k·σ sweep, and the rig's job is to measure which regime a real
fault falls in. The tooling reports both rather than hiding the conservative miss (this
is the E-LT fan-mode finding, reproduced in `test_fast_fault_cusum_late_but_ksigma_catches_it`).
