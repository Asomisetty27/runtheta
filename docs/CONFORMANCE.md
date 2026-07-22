# Conformance matrix — findings ensure functionality

Decision of record: 2026-07-13 alignment ("complete functionality" = every
README promise verified end-to-end in three environments). This matrix is the
implementation of Amogh's 2026-07-14 directive: *findings ensure theta's
functionality* — every promise cites the research finding that disciplines it
and the test or field run that verifies it.

**Environments**: `NV` = NVIDIA bare-metal · `AMD` = ROCm/amdsmi ·
`HL` = headless/service (systemd, CI, piped — no TTY).

**Evidence classes**: `F` = field-verified on real hardware ·
`T` = automated test (CI, no GPU) · `U` = unit-tested against vendor API
mocks · `—` = gap (listed at bottom, honestly).

| README promise | Finding(s) that discipline it | Evidence | NV | AMD | HL |
|---|---|---|---|---|---|
| `theta monitor` runs the full pipeline | — | `test_daemon_isolation`, D9 campaign (8×A100, 3 phases) | F | U | F |
| Refuses to start uncalibrated on non-T4 | F8/F13 (magnitude never transfers) | daemon pre-flight `_check_hardware_ready`; D9 field run exercised calibrate→monitor | F | U | F |
| `theta calibrate` (idle + `--ambient` bypass) | F2 (T_ref sensitivity) | `test_calibrate`; **AMD + headless fixed 0.1.12** (HAL routing, no-TTY input guard — PR #10, found by field use) | F | T | F |
| Never emits synthetic data silently for real decisions | cloud-spend incident 07-14 | `select_collector(allow_demo=False)`; `test_hal_strict` (7 tests) | T | T | T |
| Peer detection: matched power, MIN_GROUP, say-nothing below | F9 (low-P divergence), E009 | `test_peer`; blind-validated F7/F15 | F | T | T |
| Position-conditioned fleet scan (`fleet-scan`) | F17/F19/F21 (position is physics) | `test_fleet_scan` (position-masked unit recovered); E009 replay `theta demo` | F | T | T |
| Node-scope refusal under position structure | **F21 (measured false-positive modes)** | `test_peer::TestF21PositionLimitation` (pins the limitation + the fleet resolution) | T | T | T |
| Micro-throttle detection | **F21 (healthy DVFS ≠ throttle)** | evidence gate 0.1.13-dev; `test_silicon::TestMicroThrottleEvidenceGate` (incl. the exact A100 false-alarm scenario) | F | T | T |
| ECC double-bit → CRITICAL immediately | taxonomy (DBE = discrete class) | `test_silicon` EccMonitor suite | T | T | T |
| Alert channels: webhook/PagerDuty/Opsgenie/JSONL | — | `test_alerters_oncall`; JSONL field-verified (D9 alert log) | F | T | F |
| `--raw-log` per-tick telemetry | **F21 (campaigns needed side loggers)** | shipped 0.1.13-dev (PR #13); advisory-never-fatal | T | T | T |
| Prometheus / OTLP export | — | `test_metrics`, `test_otlp_exporter`; TUI RemoteProvider consumes the public surface | T | T | T |
| `theta top` (TUI: field/polish/replay) | F17/F19/F21 (the interface renders the findings) | `test_tui` (headless Pilot; demo-fleet detection proven by test) | T | T | T |
| DCGM enrichment collector | — | `test_dcgm_collector` | U | n/a | U |
| Redfish BMC ambient | F2 (real T_ref beats assumed) | `test_coolant_tref` | U | U | U |
| AMD ROCm collector | F21-era HAL audit | `test_rocm_collector` (12 tests incl. zero-default defenses) | n/a | **U — real-silicon run pending MI300X stock; harness merged** | U |
| Health API / schedulable semantics | governor invariants | `test_health_conditions`, `test_governor` | T | T | T |
| SLURM job reports (`theta report`) | — | `test_jobreport`, `test_slurm_hooks` | T | T | T |
| Steady-state gating (84%→99.8%) | F5/Stage-1 | `test_temporal_filter`, `test_signature` | T | T | T |
| Lead-time / predictive claims held conservative | hypothesis ledger (H3 open) | governor inferential tier; README §limitations states it plainly | T | T | T |

## Honest gaps (the work list this matrix generates)

1. **AMD real-silicon**: everything is `U` until the MI300X campaign runs
   (harness on main; RunPod stock watcher armed). This is the single biggest
   column gap and it is external-supply-gated, not code-gated.
2. **Redfish/DCGM on live hardware**: unit-level only; first field exercise is
   the Cal Poly B200 install (BMC ambient is in the day-1 runbook).
3. **systemd service path**: `deploy/install.sh` + unit file exist; the D9
   campaign verified headless *monitor* but not the installer itself on a
   fresh Linux host. Cheap to verify during the first install.
4. **`theta setup` wizard**: interactive by design; no automated conformance
   (manual QA only).
5. **Liquid-cooled margins product** (loop/node effects as coolant/pump/QD
   health): designed (vault: liquid_cooled_signatures_2026_07_14), not yet
   implemented — candidate next feature, needs loop telemetry to validate.

## Maintenance rule

A README promise may not be added or strengthened without a row here, and a
row may not claim `F` without a dated field run in the vault log. Findings
change → matrix rows change in the same PR (as F21 did for micro-throttle,
peer scope, and raw-log in PRs #13/#14).
