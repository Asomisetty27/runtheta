# Cal Poly AI Factory deployment runbook (DGX B200)

Status: access approved in principle by Chris Lupo (Noyce School Director,
2026-07-07: "something we can support and also we'd be interested in your
results"). Access mechanics owned by Ryan Matteson (expect ~2 weeks latency).
This runbook is the day-1-through-report plan for the first real fleet install.

## The target hardware (public record, verified 2026-07-09)

- 4x NVIDIA DGX B200 in a BasePOD, funded by the Noyce School ($3M), operational
  January 2026 (Cal Poly announcement, ucm.calpoly.edu).
- Each DGX B200 (NVIDIA DGX B200 User Guide, docs.nvidia.com/dgx):
  - 8x B200 SXM6 (Blackwell, dual-die CoWoS-L), 180 GB HBM3e each, ~1000 W TDP
  - **AIR-COOLED**: 10U chassis, 1,550 CFM, 6x 3.3 kW PSUs, operating 10-35 C
  - 2x Intel Xeon 8570 (56c), 2 TB RAM (up to 4 TB), 2x NVLink-5 switches
  - DGX OS (Ubuntu) ships with the NVIDIA driver, CUDA, **DCGM preinstalled**
- Theta profile: `b200` in `theta/agent/hw_profiles.py` — air-cooled,
  `t_ref_strategy="idle_window"`, expected R_theta ~0.060 C/W under load with a
  real idle/load gap. (The pre-2026-07-09 profile wrongly modeled B200 as
  liquid cold-plate; liquid Blackwell is the `gb200` profile.)

## Access model (what was actually approved)

Non-exclusive, **read-only, user-space** NVML/DCGM monitoring. No root, no
daemon, no BMC credentials, no scheduling footprint. Everything below works
within that envelope. Do NOT front-load asks for systemd install, Prometheus
ports, or Redfish/BMC creds — those are later asks after trust is earned.

## Day 1 (once Matteson provisions an account)

```bash
# 1. Sanity: can we see the GPUs at all?
nvidia-smi                      # 8x B200 expected
dcgmi discovery -l              # DCGM sees the node (ships with DGX OS)

# 2. User-space install (no root)
python3 -m venv ~/theta-env && source ~/theta-env/bin/activate
pip install runtheta nvidia-ml-py

# 3. Verify hardware resolution + read path
theta status                    # one-shot read: 8 GPUs, temps, power, R_theta
```

Check MIG: `nvidia-smi -L`. Delta-style MIG partitions change peer-group
semantics; `device_caps.py` degrades gracefully, but note the mode in the
report either way.

## Step 2 — baseline collection (this is E005)

The bundled classifier is T4-trained and `theta monitor` HARD-BLOCKS on
uncalibrated B200 (by design — see CLAUDE.md invariants). Calibration comes
first:

```bash
# ~2 h capture: idle tail + at least one real training workload window.
# Read-only NVML polling, writes a local CSV, nothing leaves the node.
# --coolant-c is the T_ref override; on the air-cooled DGX B200 pass the
# room/cold-aisle inlet temp if known, else omit (idle-window T_ref).
python tools/collect_b200_baseline.py --duration 7200 --out b200_node1.csv
python tools/analyze_b200_baseline.py --csv b200_node1.csv --out e005_node1_report.txt

# Live calibration (two phases: idle wait, then a load window). On an
# always-busy shared node, skip the idle wait with a known inlet temp:
theta calibrate --gpu 0 --ambient 25.0
# repeat per GPU index 0..7, or script it
```

Workload note: do not launch synthetic burn jobs on a shared cluster. Piggyback
on real scheduled jobs (any sustained training run is the load phase), or ask
whether a `dcgmi diag -r 3` pass is acceptable during a maintenance window.

## Step 3 — monitoring + first report

```bash
# Foreground, user-space, JSONL to disk, Prometheus port DISABLED (no
# listening sockets on a shared cluster without asking first):
theta monitor --log ~/theta-logs/node1.jsonl --port 0
```

- 8 GPUs per node satisfies peer detection's MIN_GROUP — the E009
  median-polish method is fully live from minute one (no warm-up).
- T_ref: `idle_window` (learned virtual ambient). If BMC inlet-air access is
  ever granted, the Redfish collector upgrades T_ref to measured chassis inlet;
  do not ask for this on day 1.
- Deliverable Lupo asked for ("interested in your results"): a per-node
  characterization report — R_theta operating range per GPU, peer spread,
  any outliers — same shape as the earlier fleet characterization reports.

## Known behaviors on B200 (from tests/sim_ai_factory.py, air model)

1. **Idle R_theta is often INVALID** (T_j_idle minus virtual ambient falls
   under MIN_DELTA_T). Expected: classification leans on load windows;
   CLEAN_IDLE may rarely be observed. Not a bug.
2. **A +23% single-GPU degradation sits BELOW the +40% absolute calibration
   threshold** — it is caught by the peer layer (z >> threshold vs 7 healthy
   node-mates) and the drift layer, not the absolute classifier. This is the
   designed division of labor; do not "fix" by tightening absolute thresholds
   (false-positive budget).
3. **Narrow dynamic range** (~0.07 C/W full span vs T4's 0.56): the
   B200 profile's `drift_min_std=0.002` handles this; T4-scale floors would
   swamp the signal.
4. Dual-die power: NVML may report per-die (~450 W each). `collector.py`
   power-sanity handles this; verify on real hardware and note in E005.

## Boundaries (standing, from the vault)

- Cal Poly hardware = **paper/research lane**. Results feed ThermalOS
  publication material, not startup marketing, per the Cal-Poly-vs-startup IP
  boundary (Q_university_ip_risk).
- Read-only; no data leaves the node until Amogh exports it deliberately.
- Patience protocol with Matteson: no nudges before ~2 weeks; one concise
  check-in cc'ing Lupo if silent after that.
