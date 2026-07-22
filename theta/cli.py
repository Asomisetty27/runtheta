"""
Theta CLI — theta <command>

Commands:
  setup       Interactive setup wizard (run this first)
  demo        Replay a real production incident through the live detectors (60s)
  status      Quick health check — shows GPU R_theta + daemon state in < 1s
  monitor     Run the monitoring agent (blocks)
  baseline    Run a baseline-only idle window scan
  calibrate   Measure hardware-specific R_theta thresholds (run once on non-T4 GPUs)
  classify    Single-snapshot classify all GPUs
  fleet-scan  Position-conditioned cross-node anomaly scan (the E009 method)
  report      Per-job R_θ report card from Prometheus/jobstats (by SLURM job ID)
  serve       Run agent + Prometheus metrics server only
  train       Retrain bundled models from Stage 1 CSV
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Optional

import typer
from rich import box
from rich.console import Console
from rich.table import Table

from . import __version__

app     = typer.Typer(
    name="theta",
    add_completion=False,
    pretty_exceptions_enable=False,
    help="GPU thermal-power forensics. Run [bold green]theta setup[/] to get started.",
)
console = Console()


# ── Saved-config helpers ──────────────────────────────────────────────────────

_CONFIG_PATH = Path.home() / ".theta" / "config.json"


def _saved_config() -> dict:
    """Load the wizard-saved config if present, else return an empty dict."""
    if not _CONFIG_PATH.exists():
        return {}
    try:
        return json.loads(_CONFIG_PATH.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def _coalesce(cli_value, config_key: str, saved: dict, default):
    """CLI value (if set) > saved config > default. Treat sentinels as 'unset'."""
    if cli_value is not None and cli_value != "__default__":
        return cli_value
    return saved.get(config_key, default)


# ── setup ─────────────────────────────────────────────────────────────────────

@app.command()
def setup():
    """Interactive setup wizard. Run this first. (~90 seconds)"""
    from .wizard import run_wizard
    run_wizard()


# ── status ────────────────────────────────────────────────────────────────────

@app.command()
def status(
    port: int = typer.Option(9102, "--port", "-p", help="Health API port (default: 9102)"),
):
    """
    Quick health check — shows all GPUs + daemon state in under a second.

    Tries the running daemon's health API first (instant). Falls back to a
    direct NVML snapshot if the daemon is not running.
    """
    import urllib.error
    import urllib.request

    # ── Try live daemon first ─────────────────────────────────────────────────
    daemon_live = False
    try:
        with urllib.request.urlopen(
            f"http://localhost:{port}/api/v1/health", timeout=1.0
        ) as resp:
            data = json.loads(resp.read())
        daemon_live = True
    except Exception:
        data = None

    t = Table(box=box.SIMPLE_HEAVY, show_header=True, expand=False)
    t.add_column("GPU",      style="bold", no_wrap=True)
    t.add_column("State",    no_wrap=True)
    t.add_column("R_θ C/W",  justify="right", no_wrap=True)
    t.add_column("Risk",     justify="right", no_wrap=True)
    t.add_column("Action",   no_wrap=True)

    _state_color = {
        "under_load": "green", "clean_idle": "blue",
        "drifting": "yellow", "critical": "red",
        "zombie_recovery": "red", "child_exit_recovery": "yellow",
        "unknown": "dim",
    }
    _rec_color = {"ok": "green", "watch": "yellow", "drain": "red", "evacuate": "red"}

    if daemon_live and data:
        for idx_str, gpu in sorted(data.get("gpus", {}).items(), key=lambda x: int(x[0])):
            state = gpu.get("state", "unknown")
            rtheta = gpu.get("rtheta")
            risk   = gpu.get("risk", 0.0)
            rec    = gpu.get("recommendation", "ok")
            color  = _state_color.get(state, "white")
            rcol   = _rec_color.get(rec, "white")
            t.add_row(
                f"GPU {idx_str}",
                f"[{color}]{state}[/{color}]",
                f"{rtheta:.4f}" if rtheta is not None else "—",
                f"{risk:.2f}",
                f"[{rcol}]{rec}[/{rcol}]",
            )
        uptime = data.get("uptime_ticks", "?")
        alerts = data.get("alerts", 0)
        console.print(
            f"[bold green]Theta v{__version__}[/bold green]  "
            f"[dim]daemon live · port {port} · "
            f"uptime {uptime} ticks · {alerts} alert(s)[/dim]"
        )
        console.print(t)
        return

    # ── Daemon not running — direct NVML snapshot ────────────────────────────
    console.print(
        f"[bold green]Theta v{__version__}[/bold green]  "
        f"[dim]daemon not running · direct NVML read[/dim]"
    )
    try:
        import pynvml as nv
        nv.nvmlInit()
        n = nv.nvmlDeviceGetCount()
        for i in range(n):
            h    = nv.nvmlDeviceGetHandleByIndex(i)
            name = nv.nvmlDeviceGetName(h)
            name = name.decode() if isinstance(name, bytes) else name
            pw   = nv.nvmlDeviceGetPowerUsage(h) / 1000.0
            util = nv.nvmlDeviceGetUtilizationRates(h).gpu
            ps   = nv.nvmlDeviceGetPerformanceState(h)
            rtheta_str = "—  (no baseline)"
            if pw > 10:
                pass  # can't compute R_theta without T_ref — show raw readings only
            t.add_row(
                f"GPU {i}  [dim]{name}[/dim]",
                f"P{ps}  util={util}%",
                rtheta_str,
                "—",
                "[dim]run theta setup[/dim]" if not _saved_config() else "[dim]run theta monitor[/dim]",
            )
        nv.nvmlShutdown()
        console.print(t)
        console.print(
            "\n[dim]R_θ requires a running daemon with a locked baseline. "
            "Start with: [bold]theta setup[/bold] then [bold]theta monitor[/bold][/dim]"
        )
    except Exception as e:
        console.print(f"[red]NVML error:[/red] {e}")
        console.print("[dim]Is the NVIDIA driver installed? Try: nvidia-smi[/dim]")


# ── health (scheduler-facing conditions) ──────────────────────────────────────

@app.command()
def health(
    port: int = typer.Option(9102, "--port", "-p", help="Health API port (default: 9102)"),
    schedulable_only: bool = typer.Option(False, "--schedulable", help="List only schedulable GPUs"),
):
    """
    Scheduler-facing health conditions — is each GPU fit to run work, and why?

    Shows the current LEVEL state (not alert events): per-GPU status, the
    `schedulable` flag a cordon/drain decision reads, and any active health
    conditions with how long they've held. Queries the running daemon's
    conditions endpoint.
    """
    import urllib.request

    try:
        with urllib.request.urlopen(
            f"http://localhost:{port}/api/v1/conditions", timeout=1.0
        ) as resp:
            data = json.loads(resp.read())
    except Exception:
        console.print(f"[yellow]No daemon on port {port}.[/] Start one: "
                      f"[bold]theta monitor[/bold]  (health API needs --health-port).")
        raise typer.Exit(1) from None

    gpus = data.get("gpus", {})
    summary = data.get("summary", {})
    _color = {"healthy": "green", "warming": "blue", "degraded": "yellow",
              "critical": "red", "unknown": "dim"}

    t = Table(box=box.SIMPLE_HEAVY, show_header=True)
    t.add_column("GPU", style="bold", no_wrap=True)
    t.add_column("Status", no_wrap=True)
    t.add_column("Schedulable", justify="center", no_wrap=True)
    t.add_column("Active conditions")
    for idx_str, gpu in sorted(gpus.items(), key=lambda x: int(x[0])):
        if schedulable_only and not gpu.get("schedulable"):
            continue
        st = gpu.get("status", "unknown")
        col = _color.get(st, "white")
        conds = ", ".join(c["name"] for c in gpu.get("conditions", [])) or "[dim]none[/dim]"
        sched = "[green]yes[/]" if gpu.get("schedulable") else "[red]no[/]"
        t.add_row(f"GPU {idx_str}", f"[{col}]{st}[/]", sched, conds)

    console.print(t)
    bs = summary.get("by_status", {})
    console.print(f"[dim]{summary.get('schedulable', 0)}/{summary.get('total', 0)} "
                  f"schedulable · " + " · ".join(f"{k}:{v}" for k, v in bs.items()) + "[/dim]")


# ── monitor ───────────────────────────────────────────────────────────────────

@app.command()
def monitor(
    interval:    Optional[float] = typer.Option(None, "--interval",   "-i",  help="Sample interval seconds [default: 5.0 or saved config]"),
    gpus:        Optional[str]   = typer.Option(None, "--gpus",       "-g",  help="GPU indices (comma-sep) or 'all' [default: saved config or all]"),
    webhook:     Optional[str]   = typer.Option(None, "--webhook",    "-w",  help="Alert webhook URL [default: saved config or none]"),
    pagerduty:   Optional[str]   = typer.Option(None, "--pagerduty",         help="PagerDuty Events API v2 routing key"),
    opsgenie:    Optional[str]   = typer.Option(None, "--opsgenie",          help="Opsgenie API integration key (GenieKey)"),
    otlp:        Optional[str]   = typer.Option(None, "--otlp",              help="OTLP/HTTP metrics endpoint (OpenTelemetry Collector)"),
    log_file:    Optional[str]   = typer.Option(None, "--log",                help="JSONL alert log file [default: saved config or none]"),
    raw_log:     Optional[str]   = typer.Option(None, "--raw-log",            help="JSONL raw-telemetry log: one RawSample per tick per GPU (forensic capture)"),
    port:        Optional[int]   = typer.Option(None, "--port",       "-p",  help="Prometheus port; 0 disables [default: saved config or 9101]"),
    demo:        bool            = typer.Option(False, "--demo",            help="Use synthetic demo telemetry (never selected automatically)"),
    quiet:       bool            = typer.Option(False, "--quiet",     "-q",  help="Suppress stdout alerts"),
    dt:          bool            = typer.Option(True,  "--dt/--nb",          help="Use Decision Tree (--dt) or Naive Bayes (--nb)"),
    sigma_warn:  float           = typer.Option(2.0,   "--sigma-warn",       help="Drift warning threshold (σ)"),
    sigma_crit:  float           = typer.Option(3.5,   "--sigma-crit",       help="Drift critical threshold (σ)"),
):
    """Run the Theta monitoring agent. Reads ~/.theta/config.json if present (CLI flags override)."""
    from .agent.daemon import AgentConfig, ThetaAgent

    saved = _saved_config()

    # Resolve each field: explicit CLI flag > saved config > hardcoded default
    interval_v   = interval if interval is not None else saved.get("interval_sec", 5.0)
    gpus_v       = gpus     if gpus     is not None else (
        ",".join(str(i) for i in saved["gpu_indices"]) if saved.get("gpu_indices") else "all"
    )
    webhook_v    = webhook  if webhook  is not None else saved.get("webhook_url")
    pagerduty_v  = pagerduty if pagerduty is not None else saved.get("pagerduty_key")
    opsgenie_v   = opsgenie  if opsgenie  is not None else saved.get("opsgenie_key")
    otlp_v       = otlp      if otlp      is not None else saved.get("otlp_endpoint")
    log_file_v   = log_file if log_file is not None else saved.get("alert_log_path")
    raw_log_v    = raw_log  if raw_log  is not None else saved.get("raw_log_path")
    port_v       = port     if port     is not None else (
        saved.get("prometheus_port", 9101) if saved.get("enable_prometheus", True) else 0
    )

    gpu_list = None if gpus_v == "all" else [int(g) for g in gpus_v.split(",")]

    cfg = AgentConfig(
        interval_sec      = interval_v,
        gpu_indices       = gpu_list,
        demo              = demo,
        webhook_url       = webhook_v,
        pagerduty_key     = pagerduty_v,
        opsgenie_key      = opsgenie_v,
        opsgenie_region   = saved.get("opsgenie_region", "us"),
        otlp_endpoint     = otlp_v,
        alert_log_path    = log_file_v,
        raw_log_path      = raw_log_v,
        prometheus_port   = port_v,
        enable_prometheus = port_v > 0,
        quiet             = quiet,
        prefer_dt         = dt,
        k_warn            = sigma_warn,
        k_critical        = sigma_crit,
        data_sharing      = saved.get("data_sharing", False),
        use_redfish       = saved.get("use_redfish", False),
        redfish_host      = saved.get("redfish_host"),
        redfish_user      = saved.get("redfish_user"),
        redfish_password  = saved.get("redfish_password"),
    )

    # rebind so the banner below uses resolved values
    interval, port = interval_v, port_v

    agent = ThetaAgent(cfg)

    console.print(
        f"[bold green]Theta v{__version__}[/bold green]  "
        f"[dim]interval={interval}s  classifier={agent._classifier.mode}  "
        f"{'prometheus:' + str(port) if port > 0 else 'no-metrics'}[/dim]"
    )

    try:
        asyncio.run(agent.run())
    except KeyboardInterrupt:
        console.print("\n[dim]Stopped.[/dim]")
    except RuntimeError as exc:
        console.print(f"[red]Cannot start monitor:[/red] {exc}")
        raise typer.Exit(1) from None


# ── baseline ──────────────────────────────────────────────────────────────────

@app.command()
def baseline(
    gpu:      int   = typer.Option(0,    "--gpu",      "-g",  help="GPU index to baseline"),
    duration: float = typer.Option(60.0, "--duration", "-d",  help="Max wait time for stable idle window (sec)"),
    manual:   Optional[float] = typer.Option(None, "--manual", "-m", help="Set T_ref manually (skip idle window)"),
):
    """Estimate or set virtual ambient temperature (T_ref) for a GPU."""
    from .agent.baseline import BaselineManager
    from .agent.collector import CollectorConfig, NVMLCollector

    bm = BaselineManager()

    if manual is not None:
        bm.set_manual(gpu, manual)
        console.print(f"[green]GPU {gpu}[/green] T_ref manually set to [bold]{manual:.1f} °C[/bold]")
        return

    console.print(f"Waiting for stable idle window on GPU {gpu} (up to {duration:.0f}s)…")

    async def _run():
        cfg = CollectorConfig(interval_sec=2.0, gpu_indices=[gpu])
        async with NVMLCollector(cfg) as c:
            deadline = asyncio.get_event_loop().time() + duration
            async for s in c.stream():
                bm.update(gpu, s.temp_junction, s.util_pct, s.perf_state, s.timestamp)
                if bm.has_baseline(gpu):
                    b = bm.get_baseline(gpu)
                    console.print(
                        f"[green]✓[/green] T_ref locked at [bold]{b.t_ref:.2f} °C[/bold]  "
                        f"σ={b.sigma:.3f}  n={b.n_samples}"
                    )
                    return
                if asyncio.get_event_loop().time() > deadline:
                    console.print("[yellow]Timeout — no stable idle window found. "
                                  "Use --manual to set T_ref explicitly.[/yellow]")
                    return
                console.print(f"  [dim]T={s.temp_junction:.1f}°C  util={s.util_pct:.0f}%  "
                               f"P-state=P{s.perf_state}[/dim]", end="\r")

    asyncio.run(_run())


# ── classify ──────────────────────────────────────────────────────────────────

@app.command()
def classify(
    gpus:   str  = typer.Option("all", "--gpus",    "-g", help="GPU indices or 'all'"),
    raw:    bool = typer.Option(False, "--raw",            help="Skip steady-state filter"),
):
    """Snapshot: classify current state of all GPUs."""
    from .agent.baseline import BaselineManager
    from .agent.classifier import StateClassifier
    from .agent.collector import CollectorConfig, NVMLCollector
    from .agent.metrics import (
        STATE_LABELS,  # noqa: F401  (used in _print_classify_table)
        enrich,
    )
    from .agent.window import SteadyStateWindow

    gpu_list = None if gpus == "all" else [int(g) for g in gpus.split(",")]

    async def _run():
        cfg  = CollectorConfig(interval_sec=1.0, gpu_indices=gpu_list)
        bm   = BaselineManager()
        win  = SteadyStateWindow(window_sec=15.0)
        clf  = StateClassifier()

        console.print("[dim]Collecting samples for steady-state window (15s)…[/dim]")

        async with NVMLCollector(cfg) as c:
            seen: dict[int, list] = {}
            async for s in c.stream():
                t_ref    = bm.get_t_ref(s.gpu_index)
                enriched = enrich(s, t_ref)
                bm.update(s.gpu_index, s.temp_junction, s.util_pct, s.perf_state, s.timestamp)

                if enriched.rtheta is None:
                    continue

                window = win.update(
                    s.gpu_index, s.timestamp,
                    enriched.rtheta, s.power_w, s.util_pct, s.perf_state
                )

                if window.is_stable or raw:
                    seen[s.gpu_index] = window

                if set(seen.keys()) == set(gpu_list or range(4)):
                    break

            _print_classify_table(seen, clf)

    asyncio.run(_run())


def _print_classify_table(windows, clf) -> None:
    from .agent.metrics import STATE_LABELS

    t = Table(box=box.SIMPLE_HEAVY, show_header=True)
    t.add_column("GPU",       style="bold")
    t.add_column("State",     style="bold")
    t.add_column("R_θ (C/W)", justify="right")
    t.add_column("σ (window)", justify="right")
    t.add_column("Conf",      justify="right")
    t.add_column("Reason")

    state_colors = {
        "under_load":          "green",
        "clean_idle":          "blue",
        "zombie_recovery":     "red",
        "child_exit_recovery": "yellow",
        "drifting":            "yellow",
        "critical":            "red",
        "unknown":             "dim",
    }

    for gpu_idx, window in sorted(windows.items()):
        state, conf = clf.classify(window)
        label  = STATE_LABELS.get(state, state.name)
        color  = state_colors.get(label, "white")
        reason = clf.explain(window).split("—")[-1].strip()[:60]
        t.add_row(
            str(gpu_idx),
            f"[{color}]{label}[/{color}]",
            f"{window.rtheta_mean:.4f}",
            f"{window.rtheta_std:.4f}",
            f"{conf:.2f}",
            f"[dim]{reason}[/dim]",
        )

    console.print(t)


# ── fleet-scan ────────────────────────────────────────────────────────────────

def _load_fleet_records(path: Path) -> list[dict]:
    """
    Normalize a fleet R_θ export into [{node, ordinal, rtheta, power, gpu}].

    Accepts two shapes:
      1. A flat list of records: [{"node","ordinal","rtheta","power"?}, ...]
      2. The analysis "results.json" shape: {"steady_bad": {"node:ord":
         {"r_mean","P_mean",...}}} — the format the E009 validation export uses,
         so `theta fleet-scan` runs on real multi-node data out of the box.
    """
    data = json.loads(path.read_text())
    records: list[dict] = []

    if isinstance(data, dict) and ("steady_bad" in data or "steady" in data):
        block = data.get("steady_bad") or data.get("steady") or {}
        for key, g in block.items():
            node, _, ordn = key.partition(":")
            records.append({
                "gpu": key, "node": node, "ordinal": int(ordn or 0),
                "rtheta": g["r_mean"], "power": g.get("P_mean", 0.0),
            })
    elif isinstance(data, list):
        for _i, r in enumerate(data):
            records.append({
                "gpu": r.get("gpu", f'{r.get("node","?")}:{r.get("ordinal","?")}'),
                "node": str(r["node"]), "ordinal": int(r["ordinal"]),
                "rtheta": float(r["rtheta"]), "power": float(r.get("power", 0.0)),
            })
    else:
        raise ValueError("unrecognized fleet export shape — expected a record "
                         "list or a results.json with a 'steady_bad' block")
    return records


@app.command()
def demo(
    html: Optional[str] = typer.Option(
        None, "--html",
        help="Also write the replay as a self-contained shareable HTML fleet "
             "health record at this path.",
    ),
):
    """
    Replay a REAL production incident through the live detectors (no GPU needed).

    Bundled telemetry from an incident on a production 64x H100 fleet at a top
    US research university (operator de-identified, numerics unchanged) runs
    through the same peer-relative R_θ code paths `theta monitor` uses live.
    Temperature saw a healthy fleet; theta flagged 3 units; 2 were independently
    confirmed (one replaced under RMA at a temperature no threshold can catch,
    one re-measured degraded by the operator's own staff months later).
    Distinct from `theta monitor --demo` (synthetic live telemetry).
    """
    from .agent.demo import load_fixture, render, report_input, run_replay
    from .agent.fleetreport import render_fleet_report

    fixture = load_fixture()
    res = run_replay(fixture)
    render(console, fixture, res)
    if html:
        out = Path(html)
        out.write_text(render_fleet_report(report_input(fixture, res)))
        console.print(f"[green]Shareable health record written:[/] {out}")


@app.command(name="fleet-scan")
def fleet_scan(
    export:    str   = typer.Argument(..., help="Path to a multi-node R_θ export (record list or results.json)"),
    z_thresh:  float = typer.Option(3.0, "--z", help="Robust-z threshold to flag a unit"),
    power_tol: float = typer.Option(0.15, "--power-tol", help="Only compare GPUs within ±this fractional power"),
    html: Optional[str] = typer.Option(
        None, "--html",
        help="Also write the scan as a self-contained shareable HTML fleet "
             "health record at this path.",
    ),
):
    """
    Position-conditioned cross-node anomaly scan — the E009 fleet method.

    Pools R_θ across nodes that share the HGX baseboard layout and runs two-way
    (node × ordinal) median polish, so per-position thermal structure is removed
    before scoring. This catches degraded units that a single-node within-node
    comparison misses (on the real E009 data: 3/3 vs 1/3). It needs MULTIPLE
    nodes — a single node cannot separate node effect from position effect; for a
    single host, the live agent's within-node peer detector is the right tool.
    """
    from .agent.peer import median_polish_z

    path = Path(export)
    if not path.exists():
        console.print(f"[red]export not found:[/] {export}")
        raise typer.Exit(2)

    records = _load_fleet_records(path)

    # Power-condition: R_θ is a curve in P, so only compare GPUs at matched load.
    # Use the median power as the reference band; drop GPUs outside ±power_tol
    # (e.g. idle nodes) so they aren't scored against a loaded cohort.
    powered = [r for r in records if r["power"] > 0]
    if powered:
        ref_p = sorted(r["power"] for r in powered)[len(powered) // 2]
        matched = [r for r in powered if abs(r["power"] - ref_p) <= power_tol * ref_p]
    else:
        matched = records  # no power data — score everything (best effort)

    nodes = {r["node"] for r in matched}
    if len(nodes) < 2:
        console.print(
            f"[yellow]fleet-scan needs ≥2 nodes; this export has {len(nodes)}.[/]\n"
            "Position-conditioning is undefined on a single node — run the live "
            "agent ([bold]theta monitor[/]) for within-node peer detection instead."
        )
        raise typer.Exit(1)

    fleet = {r["gpu"]: (r["node"], r["ordinal"], r["rtheta"]) for r in matched}
    z = median_polish_z(fleet)

    flagged = sorted(
        ((gpu, zz) for gpu, zz in z.items() if zz >= z_thresh),
        key=lambda kv: -kv[1],
    )

    console.print(
        f"[dim]scanned {len(matched)} GPUs across {len(nodes)} nodes "
        f"at matched power (~{ref_p:.0f} W); {len(records) - len(matched)} off-band excluded[/dim]"
        if powered else f"[dim]scanned {len(matched)} GPUs across {len(nodes)} nodes[/dim]"
    )

    t = Table(box=box.SIMPLE_HEAVY, show_header=True, title="Position-conditioned anomalies (median polish)")
    t.add_column("GPU", style="bold")
    t.add_column("robust-z", justify="right")
    t.add_column("R_θ (C/W)", justify="right")
    t.add_column("verdict")
    rmap = {r["gpu"]: r for r in matched}
    for gpu, zz in flagged:
        color = "red" if zz >= 8 else "yellow"
        verdict = "CRITICAL" if zz >= 8 else "anomaly"
        t.add_row(gpu, f"[{color}]{zz:+.1f}[/]", f"{rmap[gpu]['rtheta']:.4f}", f"[{color}]{verdict}[/]")

    if flagged:
        console.print(t)
        console.print(f"[bold]{len(flagged)}[/] unit(s) flagged at robust-z ≥ {z_thresh}.")
    else:
        console.print(f"[green]No units above robust-z {z_thresh}.[/] Fleet looks uniform after position correction.")

    if html:
        from .agent.fleetreport import ReportInput, ReportRow, render_fleet_report

        rows = [
            ReportRow(
                gpu=r["gpu"], node=r["node"], temp_c=None,
                power_w=r["power"] or None, rtheta=r["rtheta"],
                z=z.get(r["gpu"], 0.0),
            )
            for r in matched
        ]
        report = ReportInput(
            title="Fleet health record · position-conditioned R_θ scan",
            fleet_desc=(
                f"Snapshot scan of {len(matched)} GPUs across {len(nodes)} nodes "
                f"from {path.name}, peer-relative thermal resistance at matched load."
            ),
            rows=rows,
            ref_power=ref_p if powered else None,
            temp_threshold_c=None,
            scope_note=(
                "Position-conditioned robust-z on a single matched-load snapshot. "
                "Flags indicate peer-relative thermal-resistance outliers, i.e. "
                "likely degraded cooling paths, not imminent failure; theta detects "
                "condition, not failure dates. Verify flagged units against "
                "maintenance records and airflow/TIM inspection."
            ),
        )
        out = Path(html)
        out.write_text(render_fleet_report(report))
        console.print(f"[green]Shareable health record written:[/] {out}")


# ── report (SLURM/jobstats per-job R_θ) ───────────────────────────────────────

@app.command()
def certify(
    gpu:    int = typer.Option(0, "--gpu", help="GPU index to certify"),
    out:    Optional[str] = typer.Option(None, "--out", help="Write certificate JSON to this path"),
    agent_url: str = typer.Option("http://localhost:9102", "--agent-url",
                                  help="Running theta agent's health API (for record-backed certificates)"),
    active: bool = typer.Option(False, "--active",
                                help="Run the active probe (GEMM + memory bandwidth, ~60s of load) "
                                     "and append measured functional verification"),
    active_seconds: float = typer.Option(60.0, "--active-seconds",
                                         help="Active probe duration budget"),
    record_dir: str = typer.Option("~/.theta/record", "--record-dir",
                                   help="Device record store (hash-chained history); "
                                        "absent or empty = point-in-time certificate"),
):
    """
    Issue a verified GPU health certificate (theta-certificate/v1).

    The certificate = identity + record span + condition (with per-cause
    validation grade) + actuarial risk tier + the scope refusals, sealed with
    an integrity hash. With a running agent it is record-backed; without one
    it is issued as an explicitly-downgraded POINT-IN-TIME inspection - a
    snapshot cannot reproduce operating history, and the document says so.

      theta certify --gpu 0 --out gpu0_cert.json
    """
    import time as _time

    from . import __version__
    from .agent.certificate import build_certificate

    # ── identity (live NVML; the one hard requirement) ──────────────────────
    identity = {"gpu_index": gpu}
    try:
        import pynvml as nv
        nv.nvmlInit()
        h = nv.nvmlDeviceGetHandleByIndex(gpu)
        def _s(x):
            return x.decode() if isinstance(x, bytes) else x
        identity["name"] = _s(nv.nvmlDeviceGetName(h))
        identity["uuid"] = _s(nv.nvmlDeviceGetUUID(h))
        for key, fn in (("serial", nv.nvmlDeviceGetSerial),
                        ("vbios", nv.nvmlDeviceGetVbiosVersion)):
            try:
                identity[key] = _s(fn(h))
            except nv.NVMLError:
                identity[key] = None
        identity["driver"] = _s(nv.nvmlSystemGetDriverVersion())
    except Exception as e:  # noqa: BLE001
        console.print(f"[red]cannot read device identity (NVML):[/] {e}")
        raise typer.Exit(2) from e

    # ── agent-backed evidence (optional; absence downgrades, never fails) ───
    health = econ = incident = None
    record_span = observed_hours = None
    # Record store: the hash-chained history that upgrades a certificate from
    # point-in-time to record-backed. A broken chain is a HARD failure — a
    # tampered record must never quietly downgrade to a fresh-looking cert.
    from .agent.recordstore import RecordStore
    _rdir = Path(record_dir).expanduser()
    if _rdir.exists() and any(_rdir.glob("records-*.jsonl")):
        store = RecordStore(_rdir)
        if not store.verify_chain():
            console.print("[red]record chain verification FAILED - the record "
                          "has been modified. Refusing to certify from it.[/]")
            raise typer.Exit(3)
        _sum = store.summary()
        if _sum["span_start"] is not None:
            record_span = (_sum["span_start"], _sum["span_end"])
            observed_hours = _sum["observed_hours"]
    try:
        import httpx
        r = httpx.get(f"{agent_url}/api/v1/conditions", timeout=3.0)
        if r.status_code == 200:
            gpus = r.json().get("gpus", {})
            g = gpus.get(str(gpu))
            if g:
                health = g
    except Exception:  # noqa: BLE001 — no running agent = point-in-time cert
        pass

    probe_block = None
    if active:
        from .agent.activeprobe import grade_probe, run_active_probe
        console.print("[yellow]active probe: loading the GPU for "
                      f"~{active_seconds:.0f}s (GEMM + memory bandwidth)...[/]")

        def _sample():
            d = {}
            try:
                d["temp_c"] = nv.nvmlDeviceGetTemperature(h, nv.NVML_TEMPERATURE_GPU)
                d["power_w"] = nv.nvmlDeviceGetPowerUsage(h) / 1000.0
                d["sm_mhz"] = nv.nvmlDeviceGetClockInfo(h, nv.NVML_CLOCK_SM)
                d["throttle"] = nv.nvmlDeviceGetCurrentClocksThrottleReasons(h)
            except Exception:  # noqa: BLE001
                pass
            return d

        try:
            measured = run_active_probe(gpu_index=gpu, seconds=active_seconds,
                                        sample_fn=_sample)
            probe_block = grade_probe(measured, identity.get("name", ""))
        except RuntimeError as e:
            console.print(f"[red]active probe unavailable:[/] {e}")
            raise typer.Exit(2) from e

    cert = build_certificate(
        now=_time.time(), agent_version=__version__, identity=identity,
        record_span=record_span, observed_hours=observed_hours,
        health=health, econ=econ, incident=incident,
        active_probe=probe_block,
    )
    text = json.dumps(cert, indent=2)
    if out:
        Path(out).write_text(text)
        console.print(f"[green]certificate written:[/] {out}  "
                      f"(grade: {cert['record']['grade']})")
    else:
        console.print_json(text)


@app.command()
def report(
    jobid:    str = typer.Argument(..., help="SLURM job ID (live) or a label (export mode)"),
    prom:     Optional[str] = typer.Option(None, "--prom", help="Prometheus base URL (live mode)"),
    start:    Optional[float] = typer.Option(None, "--start", help="Range start (unix sec, live mode)"),
    end:      Optional[float] = typer.Option(None, "--end", help="Range end (unix sec, live mode)"),
    export:   Optional[str] = typer.Option(None, "--export", help="Dir of Prometheus query_range JSON (offline mode)"),
    ambient:  float = typer.Option(25.0, "--ambient", help="Assumed inlet °C (affects magnitudes only; detection is peer-relative)"),
    z_thresh: float = typer.Option(3.0, "--z", help="Robust-z to flag a unit"),
):
    """
    Per-job R_θ report card — the SLURM/jobstats integration.

    Pulls a job's GPU temp/power/util (the metrics jobstats already scrapes into
    Prometheus) and reports per-job cooling health: per-GPU R_θ, peer comparison,
    and any cooling-degraded units — flagged (act) and watch (elevated). Runs the
    validated E009 detection (peer-relative within a node; position-conditioned
    median polish across ≥2 nodes).

      theta report 6982217 --prom http://prometheus:9090 --start 1777147080 --end 1777159000
      theta report 6982217 --export /path/to/prometheus_json_dir
    """
    from .agent.jobreport import (
        build_report,
        load_exports,
        load_prometheus,
        steady_rtheta,
    )

    if export:
        d = Path(export)
        files = sorted(d.glob("*.json")) if d.is_dir() else [d]
        if not files:
            console.print(f"[red]no JSON exports found under[/] {export}")
            raise typer.Exit(2)
        aligned = load_exports(files, jobid=jobid)
    elif prom:
        if start is None or end is None:
            console.print("[red]--start and --end (unix seconds) are required in live mode[/]")
            raise typer.Exit(2)
        try:
            aligned = load_prometheus(prom, jobid, start, end)
        except Exception as e:  # network / query errors
            console.print(f"[red]Prometheus query failed:[/] {e}")
            raise typer.Exit(2) from e
    else:
        console.print("[red]provide either --export <dir> or --prom <url> --start --end[/]")
        raise typer.Exit(2)

    stats = steady_rtheta(aligned, ambient=ambient)
    rep = build_report(jobid, stats, z_thresh=z_thresh)

    console.print()
    console.print(f"[bold]Theta job report[/] · job [bold cyan]{rep.jobid}[/] · "
                  f"{len(rep.gpus)} GPUs · {len(rep.nodes)} node(s) · method=[dim]{rep.method}[/]")
    if rep.fleet_mean_r is not None:
        console.print(f"[dim]fleet mean steady-load R_θ = {rep.fleet_mean_r:.4f} C/W[/]")
    for n in rep.notes:
        console.print(f"[dim]note: {n}[/dim]")

    if not rep.gpus:
        console.print("[yellow]No steady-load samples — job may be too short or idle.[/]")
        raise typer.Exit(1)

    t = Table(box=box.SIMPLE_HEAVY, show_header=True)
    t.add_column("GPU", style="bold")
    t.add_column("R_θ (C/W)", justify="right")
    t.add_column("T̄ (°C)", justify="right")
    t.add_column("P̄ (W)", justify="right")
    t.add_column("robust-z", justify="right")
    t.add_column("status")
    for s in sorted(rep.gpus, key=lambda g: (g.node, g.ordinal)):
        z = rep.flagged.get(s.key, rep.watch.get(s.key))
        if s.key in rep.flagged:
            color, status = ("red", "FLAGGED")
        elif s.key in rep.watch:
            color, status = ("yellow", "watch")
        else:
            color, status = ("green", "ok")
        zs = f"[{color}]{z:+.1f}[/]" if z is not None else "[dim]—[/]"
        t.add_row(s.key, f"{s.r_mean:.4f}", f"{s.t_mean:.0f}",
                  f"{s.p_mean:.0f}", zs, f"[{color}]{status}[/]")
    console.print(t)

    if rep.flagged:
        console.print(f"[bold red]{len(rep.flagged)}[/] unit(s) flagged (robust-z ≥ {z_thresh}) — "
                      f"cooling-degraded relative to peers at matched power.")
    if rep.watch:
        console.print(f"[yellow]{len(rep.watch)}[/] unit(s) on watch (elevated, sub-threshold).")
    if not rep.flagged and not rep.watch:
        console.print("[green]All GPUs nominal — no peer-relative cooling anomalies.[/]")


# ── calibrate ─────────────────────────────────────────────────────────────────

@app.command()
def calibrate(
    gpu:              int            = typer.Option(0,     "--gpu",              "-g", help="GPU index to calibrate"),
    idle_wait:        float          = typer.Option(120.0, "--idle-wait",              help="Max seconds to wait for stable idle"),
    load_wait:        float          = typer.Option(120.0, "--load-wait",              help="Max seconds to wait for stable load"),
    skip_load:        bool           = typer.Option(False, "--skip-load",              help="Calibrate idle only (skip load phase)"),
    ambient:          Optional[float]= typer.Option(None,  "--ambient",           "-a", help="Known ambient/coolant temp °C — skips idle-phase wait entirely (use for always-busy DGX nodes)"),
    calibration_file: Optional[str]  = typer.Option(None,  "--calibration-file",        help="Write calibration to this path instead of ~/.theta/calibration.json (use for shared service installs)"),
):
    """
    Measure hardware-specific R_theta thresholds and save to ~/.theta/calibration.json.

    Run this once after setup on any GPU that is not a Tesla T4. The bundled
    classifiers are trained on T4 Stage 1 data — they will misclassify on
    hardware with a different thermal envelope (A100, H100, B200, etc.).

    Two phases:
      1. Idle phase   — waits for the GPU to reach stable idle automatically.
                        Skip with --ambient <temp> if the GPU is never truly idle
                        (e.g., always-running DGX nodes at an AI Factory).
      2. Load phase   — prompts you to start a workload, then locks the load R_theta.

    For shared production installs (systemd service user):
      theta calibrate --gpu 0 --calibration-file /etc/theta/calibration.json
    """
    import asyncio
    from pathlib import Path as _Path

    from .agent.baseline import BaselineManager
    from .agent.calibrate import (
        CalibrationManager,
        CalibrationResult,
        derive_thresholds,
        run_idle_phase,
        run_load_phase,
    )
    from .agent.collector import CollectorConfig
    from .agent.hal import probe_gpu_name, select_collector

    cal_file = _Path(calibration_file) if calibration_file else None
    cal_mgr = CalibrationManager(_file=cal_file)
    existing = cal_mgr.get(gpu)
    if existing:
        console.print(
            f"[dim]Existing calibration for GPU {gpu}: "
            f"load_threshold={existing.load_threshold} C/W  "
            f"idle_threshold={existing.idle_threshold} C/W  "
            f"age={existing.age_hours():.1f}h[/dim]"
        )

    # ── Get GPU name (vendor-agnostic: NVML, then amdsmi) ─────────────────────
    # The name gates hardware-profile resolution below — a pynvml-only lookup
    # on an AMD host would silently anchor the default profile instead of the
    # bundled vendor profile (e.g. mi300x).
    gpu_name = probe_gpu_name(gpu) or f"GPU {gpu}"

    console.print(f"\n[bold]theta calibrate[/bold] — {gpu_name} (GPU {gpu})\n")

    bm = BaselineManager()

    # ── Phase 1: Idle (or --ambient bypass) ───────────────────────────────────
    rtheta_idle: Optional[float] = None

    if ambient is not None:
        # Bypass idle-phase wait when the ambient temperature is known externally
        # (BMC reading, coolant inlet sensor) or when the GPU is always busy.
        # We seed T_ref from the supplied ambient and compute R_θ_idle from the
        # hardware profile's expected idle R_θ value.
        from .agent.hw_profiles import resolve_or_default
        profile = resolve_or_default(gpu_name)
        bm.set_external_ambient(gpu, ambient, source="manual_ambient_flag")

        # Derive expected idle R_θ from profile + supplied ambient.
        # For air-cooled hardware: R_θ_idle differs from R_θ_load so the profile's
        # rtheta_expected_idle gives a useful calibration anchor.
        # For liquid-cooled hardware (t_ref_strategy='coolant_inlet'): R_θ_idle ≈
        # R_θ_load — the T4-ratio idle-only scaling in derive_thresholds() would
        # produce a load_threshold BELOW the actual healthy load R_θ, causing all
        # normal load windows to be classified as DRIFTING. Use the profile's pre-
        # computed thresholds directly for liquid-cooled hardware instead.
        _strategy = getattr(profile, "t_ref_strategy", "idle_window")
        if _strategy == "coolant_inlet":
            # Liquid-cooled: R_θ_idle ≈ R_θ_load = rtheta_expected_under_load.
            # The profile thresholds are already calibrated for this regime.
            rtheta_idle = profile.rtheta_expected_under_load
        else:
            rtheta_idle = profile.rtheta_expected_idle
        console.print(
            f"[yellow]⚠[/yellow]  [bold]--ambient mode[/bold]: skipping idle-phase wait.\n"
            f"  Using supplied ambient [bold]{ambient:.1f} °C[/bold] as T_ref, "
            f"profile R_θ_idle estimate [bold]{rtheta_idle:.4f} C/W[/bold]"
            + (" [dim](liquid-cooled: R_θ_idle ≈ R_θ_load)[/dim]" if _strategy == "coolant_inlet" else "")
            + ".\n"
            "  [dim]Accuracy is lower than an observed idle window. "
            "Re-run without --ambient during a maintenance window for best results.[/dim]\n"
        )
    else:
        console.print("[bold cyan]Phase 1 — Idle[/bold cyan]")
        console.print(f"  Waiting up to {idle_wait:.0f}s for stable idle (util < 5%)…")
        console.print(
            "  [dim]Make sure no compute workloads are running.\n"
            "  On always-busy nodes (DGX, AI Factory), use: "
            "[bold]theta calibrate --ambient <inlet_temp_c>[/bold][/dim]\n"
        )

        async def _idle():
            cfg = CollectorConfig(interval_sec=2.0, gpu_indices=[gpu])
            # allow_demo=False: calibrating against synthetic samples would
            # write plausible-looking but meaningless thresholds to disk.
            async with select_collector(cfg, allow_demo=False) as c:
                return await run_idle_phase(c, bm, max_wait_sec=idle_wait)

        rtheta_idle = asyncio.run(_idle())

        if rtheta_idle is None:
            console.print(
                "[red]✗[/red] Idle phase timed out.\n"
                "  Options:\n"
                "  • Free the GPU and retry\n"
                f"  • Use [bold]theta calibrate --gpu {gpu} --ambient <inlet_temp_c>[/bold] "
                "to bypass with a known temperature\n"
                f"  • Use [bold]theta baseline --gpu {gpu} --manual <T>[/bold] to set T_ref manually"
            )
            raise typer.Exit(code=1)

        console.print(f"[green]✓[/green] Idle R_θ locked: [bold]{rtheta_idle:.4f} C/W[/bold]\n")

    # ── Phase 2: Load (optional) ──────────────────────────────────────────────
    rtheta_load: Optional[float] = None

    if not skip_load:
        console.print("[bold cyan]Phase 2 — Load[/bold cyan]")
        if sys.stdin.isatty():
            console.print("  Start a GPU compute workload now (training job, inference loop, stress test).")
            console.print("  Press [bold]Enter[/bold] when the workload is running…", end=" ")
            try:
                input()
            except EOFError:
                pass
        else:
            # Headless (piped/service/CI): input() would block on an open pipe
            # forever. Assume the workload is already running and go straight
            # to the stability wait — that's what unattended callers mean.
            console.print("  [dim]No TTY — assuming a workload is already running "
                          "(use --skip-load to calibrate idle-only).[/dim]")

        console.print(f"  Waiting up to {load_wait:.0f}s for stable load (util > 70%)…\n")

        async def _load():
            cfg = CollectorConfig(interval_sec=2.0, gpu_indices=[gpu])
            async with select_collector(cfg, allow_demo=False) as c:
                return await run_load_phase(c, bm, max_wait_sec=load_wait)

        rtheta_load = asyncio.run(_load())

        if rtheta_load is None:
            console.print(
                "[yellow]⚠[/yellow] Load phase timed out — GPU utilization did not reach 70%. "
                "Calibrating with idle phase only."
            )
        else:
            console.print(f"[green]✓[/green] Load R_θ locked: [bold]{rtheta_load:.4f} C/W[/bold]\n")

    # ── Derive thresholds + save ──────────────────────────────────────────────
    # For liquid-cooled hardware (t_ref_strategy='coolant_inlet') in --ambient mode,
    # derive_thresholds() with idle-only path applies T4-ratio scaling which produces
    # a load_threshold well below the actual healthy load R_theta (because T4 assumes
    # idle >> load, but liquid-cooled idle ≈ load). Use the profile's precomputed
    # thresholds when no load observation is available on coolant_inlet hardware.
    _use_profile_thresholds = (
        ambient is not None
        and rtheta_load is None
        and "profile" in locals()  # profile was set in the --ambient block
        and getattr(profile, "t_ref_strategy", "idle_window") == "coolant_inlet"
    )
    if _use_profile_thresholds:
        load_threshold = profile.rtheta_load_threshold
        idle_threshold = profile.rtheta_idle_threshold
        source = "profile_liquid_cooled"
        console.print(
            f"  [dim]Liquid-cooled hardware: using profile thresholds "
            f"(load={load_threshold:.3f}, idle={idle_threshold:.3f}) "
            f"instead of T4-ratio scaling.[/dim]\n"
        )
    else:
        load_threshold, idle_threshold = derive_thresholds(rtheta_idle, rtheta_load)
    source = "observed_both" if rtheta_load is not None else ("profile_liquid_cooled" if _use_profile_thresholds else "idle_only")

    result = CalibrationResult(
        gpu_index       = gpu,
        gpu_name        = gpu_name,
        rtheta_idle     = rtheta_idle,
        rtheta_load     = rtheta_load,
        load_threshold  = load_threshold,
        idle_threshold  = idle_threshold,
        calibrated_at   = time.time(),
        source          = source,
    )
    cal_mgr.set(result)

    # ── Summary table ─────────────────────────────────────────────────────────
    from rich import box
    from rich.table import Table

    t = Table(box=box.SIMPLE_HEAVY, show_header=True)
    t.add_column("Metric",    style="bold")
    t.add_column("Value",     justify="right")
    t.add_column("Notes")

    t.add_row("GPU",              gpu_name,                         "")
    t.add_row("R_θ idle",         f"{rtheta_idle:.4f} C/W",         "stable 20s window")
    t.add_row("R_θ load",
              f"{rtheta_load:.4f} C/W" if rtheta_load else "—",
              "stable 20s window" if rtheta_load else "not observed")
    t.add_row("load_threshold",   f"{load_threshold:.3f} C/W",      "R_θ ≤ this → under_load")
    t.add_row("idle_threshold",   f"{idle_threshold:.3f} C/W",      "R_θ ≥ this → idle territory")
    t.add_row("source",           source,                           "")
    t.add_row("saved to",         str(cal_mgr._file),               "")

    console.print(t)
    console.print("\n[green]Calibration complete.[/green] "
                  "Run [bold]theta monitor[/bold] to start using calibrated thresholds.")


# ── train ─────────────────────────────────────────────────────────────────────

@app.command()
def train(
    csv: str = typer.Argument(..., help="Path to ThermalOS_Measurements_Raw.csv"),
):
    """Retrain bundled classifier models from Stage 1 CSV data."""
    csv_path = Path(csv)
    if not csv_path.exists():
        console.print(f"[red]Error:[/red] CSV not found: {csv}")
        raise typer.Exit(code=1)
    from .models.train import train as do_train
    do_train(csv_path)


# ── serve ─────────────────────────────────────────────────────────────────────

@app.command()
def serve(
    port: int = typer.Option(9101, "--port", "-p", help="Prometheus metrics port"),
    interval: float = typer.Option(5.0, "--interval", "-i"),
    demo: bool = typer.Option(False, "--demo", help="Use synthetic demo telemetry (never selected automatically)"),
):
    """Run agent with Prometheus metrics export only (no stdout alerts)."""
    from .agent.daemon import AgentConfig, ThetaAgent
    cfg   = AgentConfig(interval_sec=interval, prometheus_port=port, quiet=True, demo=demo)
    agent = ThetaAgent(cfg)
    console.print(f"[green]Metrics:[/green] http://localhost:{port}/metrics")
    try:
        asyncio.run(agent.run())
    except KeyboardInterrupt:
        console.print("\n[dim]Stopped.[/dim]")
    except RuntimeError as exc:
        console.print(f"[red]Cannot start metrics exporter:[/red] {exc}")
        raise typer.Exit(1) from None


@app.command()
def top(
    url: Optional[str] = typer.Option(
        None, "--url", "-u",
        help="Attach to a running agent's metrics endpoint (host:9101). "
             "Omit to read local GPUs (demo mode without NVML).",
    ),
    k8s: bool = typer.Option(
        False, "--k8s",
        help="Attach to a Theta agent running in Kubernetes: finds the pod, "
             "port-forwards, and cleans up on exit. One command, no second terminal.",
    ),
    namespace: str = typer.Option("default", "--namespace", "-n", help="Kubernetes namespace (with --k8s)"),
    interval: float = typer.Option(2.0, "--interval", "-i", help="Refresh seconds"),
):
    """Live fleet TUI — htop for GPU thermal health (R_θ sparklines, alerts)."""
    try:
        from . import tui as theta_tui
    except ImportError:
        console.print(
            "[red]theta top needs the UI extra:[/red]  pip install 'runtheta\\[ui]'"
        )
        raise typer.Exit(1) from None
    if k8s:
        theta_tui.run_k8s(namespace=namespace, interval=interval)
    else:
        theta_tui.run(url=url, interval=interval)


# ── analyze-export (partner telemetry) ───────────────────────────────────────

@app.command(name="analyze-export")
def analyze_export(
    path: str = typer.Argument(..., help="Telemetry export: a .csv file, or a dir/file of Prometheus query_range JSON."),
    fmt: str = typer.Option("auto", "--format", "-f", help="auto | csv | prom"),
    jobid: Optional[str] = typer.Option(None, "--jobid", help="Filter to one job id/label if the export holds several."),
    label: Optional[str] = typer.Option(None, "--label", help="Display label for the report."),
    out_json: Optional[str] = typer.Option(None, "--json", help="Also write the machine-readable report here."),
):
    """
    Analyze a partner's per-GPU telemetry export end-to-end: R_θ + peer-relative
    detection + signature-matrix cause attribution. The deliverable you send back.

    Accepts a CSV (flexible columns) or Prometheus query_range JSON. Detection is
    peer-relative so the absolute ambient assumption never affects which units
    flag; cause attribution uses the fleet-relative T-vs-P decomposition.
    """
    from .agent.jobreport import load_csv, load_exports
    from .agent.partner_report import analyze

    src = Path(path)
    if not src.exists():
        console.print(f"[red]Not found:[/red] {path}")
        raise typer.Exit(1)

    kind = fmt
    if kind == "auto":
        kind = "csv" if src.is_file() and src.suffix.lower() == ".csv" else "prom"

    if kind == "csv":
        aligned = load_csv([src], jobid=jobid)
    else:
        paths = sorted(src.glob("*.json")) if src.is_dir() else [src]
        aligned = load_exports(paths, jobid=jobid)

    if not aligned:
        console.print("[yellow]No per-GPU series found in the export.[/] "
                      "Check column mapping (CSV) or that T/P/util are all present.")
        raise typer.Exit(1)

    lbl = label or jobid or src.name
    report, attributions, text = analyze(aligned, label=lbl)
    console.print(text)

    if out_json:
        payload = {
            "label": lbl,
            "method": report.method,
            "fleet_mean_r": report.fleet_mean_r,
            "n_gpus": len(report.gpus),
            "nodes": report.nodes,
            "units": [
                {
                    "key": a.key, "tier": a.tier, "robust_z": round(a.robust_z, 2),
                    "t_mean": a.stat.t_mean if a.stat else None,
                    "p_mean": a.stat.p_mean if a.stat else None,
                    "attribution": a.verdict.as_dict(),
                }
                for a in attributions
            ],
        }
        Path(out_json).write_text(json.dumps(payload, indent=2))
        console.print(f"\n[green]Wrote[/green] {out_json}")


# ── characterize (fleet characterization report) ─────────────────────────────

@app.command()
def characterize(
    path: str = typer.Argument(..., help="Telemetry export: a .csv file, or a dir/file of Prometheus query_range JSON."),
    fmt: str = typer.Option("auto", "--format", "-f", help="auto | csv | prom"),
    jobid: Optional[str] = typer.Option(None, "--jobid", help="Filter to one job id/label if the export holds several."),
    label: Optional[str] = typer.Option(None, "--label", help="Display label for the report."),
    out_html: Optional[str] = typer.Option(None, "--html", help="Output path (default: theta-characterization-<label>.html)"),
):
    """
    Generate a fleet characterization report (self-contained HTML): fleet
    summary, peer-relative findings with cause attribution, the R_θ(P)
    power-tier curve, and a per-GPU characterization table.

    Same ingestion and analysis as analyze-export — this is the shareable
    artifact form of it.
    """
    from datetime import datetime, timezone

    from .agent.characterize import characterize as run_characterize
    from .agent.jobreport import load_csv, load_exports

    src = Path(path)
    if not src.exists():
        console.print(f"[red]Not found:[/red] {path}")
        raise typer.Exit(1)

    kind = fmt
    if kind == "auto":
        kind = "csv" if src.is_file() and src.suffix.lower() == ".csv" else "prom"

    if kind == "csv":
        aligned = load_csv([src], jobid=jobid)
    else:
        paths = sorted(src.glob("*.json")) if src.is_dir() else [src]
        aligned = load_exports(paths, jobid=jobid)

    if not aligned:
        console.print("[yellow]No per-GPU series found in the export.[/] "
                      "Check column mapping (CSV) or that T/P/util are all present.")
        raise typer.Exit(1)

    lbl = label or jobid or src.name
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    report, attributions, doc = run_characterize(aligned, label=lbl, generated_at=generated)

    dest = Path(out_html or f"theta-characterization-{lbl.replace('/', '_').replace(' ', '_')}.html")
    dest.write_text(doc)
    console.print(
        f"[green]Wrote[/green] {dest}   "
        f"[dim]{len(report.gpus)} GPUs · {len(report.nodes)} node(s) · "
        f"{len(report.flagged)} flagged · {len(report.watch)} watch[/dim]"
    )


# ── incidents ───────────────────────────────────────────────────────────────

@app.command()
def incidents(
    show: Optional[str] = typer.Argument(None, help="Incident id to show in full; omit to list."),
    pending: bool = typer.Option(False, "--pending", help="Only resolved incidents awaiting a label."),
    path: Optional[str] = typer.Option(None, "--store", help="Incident store path (default ~/.theta/incidents.jsonl)."),
):
    """List tracked GPU incidents, or show one in full. The flywheel ledger."""
    from .agent.incident_store import IncidentStore

    store = IncidentStore(path)

    if show:
        inc = store.get(show)
        if inc is None:
            console.print(f"[red]No incident with id[/red] {show}")
            raise typer.Exit(code=1)
        console.print_json(json.dumps(inc.to_dict()))
        return

    rows = store.unlabeled_resolved() if pending else store.all()
    acc = store.accuracy()
    if acc["rate"] is None:
        console.print("[dim]Measured cause accuracy: not earned yet (0 labeled incidents).[/dim]")
    else:
        console.print(
            f"[bold]Measured cause accuracy:[/bold] {acc['correct']}/{acc['labeled']} "
            f"= {acc['rate']:.0%} [dim](labeled incidents only)[/dim]"
        )
    if not rows:
        console.print("[dim]No incidents.[/dim]")
        return

    table = Table(box=box.SIMPLE)
    for col in ("id", "gpu", "node", "cause", "tier", "stage", "label"):
        table.add_column(col)
    for inc in rows:
        table.add_row(
            inc.id, str(inc.gpu_index), inc.node, inc.cause,
            inc.effective_tier, inc.stage, inc.confirmed_label or "—",
        )
    console.print(table)


# ── label ─────────────────────────────────────────────────────────────────────

@app.command()
def label(
    incident_id: str = typer.Argument(..., help="Incident id (from `theta incidents`)."),
    cause: str = typer.Argument(..., help="Confirmed cause, e.g. tim_degradation, airflow_blockage, fabric_link."),
    notes: Optional[str] = typer.Option(None, "--notes", "-n", help="What the inspection/repair found."),
    path: Optional[str] = typer.Option(None, "--store", help="Incident store path."),
):
    """
    Attach an operator-confirmed ground-truth label to an incident.

    This is the only action that elevates an incident to CONFIRMED_CAUSE — it is
    how Theta's accuracy becomes a measured number instead of a physics prior.
    """
    from .agent.incident_store import IncidentStore

    store = IncidentStore(path)
    inc = store.label(incident_id, cause, now=time.time(), notes=notes)
    if inc is None:
        console.print(f"[red]No incident with id[/red] {incident_id}")
        raise typer.Exit(code=1)

    verdict = inc.prediction_was_correct
    mark = "[green]✓ matched Theta's prediction[/green]" if verdict else "[yellow]≠ differed from Theta's prediction[/yellow]"
    console.print(f"Labeled {incident_id}: predicted [bold]{inc.cause}[/bold], confirmed [bold]{cause}[/bold] — {mark}")


@app.command()
def ask(
    question: str = typer.Argument(..., help="Operator question, e.g. 'is any GPU degrading?'"),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Override the agent model."),
    show_calls: bool = typer.Option(False, "--show-calls", help="Print the tool-call audit trail."),
):
    """
    Ask the LLM operator-reasoning agent about fleet health (natural language).

    The agent reasons over Theta's live READ-ONLY tools (fleet status, per-GPU
    prognosis/details/risk) and explains what is degrading and how confident it
    is. It is advisory: it cannot drain a node or fire an alert -- those stay with
    the deterministic governor. Needs the optional dep: pip install 'runtheta[agent]'
    and ANTHROPIC_API_KEY. See theta/agent/operator.py for the design.
    """
    from .agent.operator import OperatorAgent

    agent = OperatorAgent(model=model) if model else OperatorAgent()
    ok, reason = agent.available()
    if not ok:
        console.print(f"[yellow]{reason}[/yellow]")
        raise typer.Exit(code=1)

    with console.status("[dim]Theta agent reasoning over live fleet state...[/dim]"):
        answer = agent.ask(question)

    console.print(answer.text)
    if answer.stopped_reason == "max_turns":
        console.print("[dim](turn budget reached — answer may be partial)[/dim]")
    if show_calls and answer.tool_calls:
        console.print("\n[dim]tool calls:[/dim]")
        for c in answer.tool_calls:
            console.print(f"[dim]  {c.name}({c.args})[/dim]")


@app.command()
def chat(
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Override the agent model."),
):
    """
    Interactive multi-turn session with the operator agent (Memory pattern, Ch 8).

    Same read-only, advisory agent as `theta ask`, but it keeps conversation
    context so you can ask follow-ups ("what about GPU 5?", "why?"). Type 'exit'
    or Ctrl-D to quit. Needs pip install 'runtheta[agent]' and ANTHROPIC_API_KEY.
    """
    from .agent.operator import OperatorAgent, OperatorSession

    agent = OperatorAgent(model=model) if model else OperatorAgent()
    ok, reason = agent.available()
    if not ok:
        console.print(f"[yellow]{reason}[/yellow]")
        raise typer.Exit(code=1)

    session = OperatorSession()
    console.print("[dim]Theta operator agent. Ask about fleet health. 'exit' to quit.[/dim]")
    while True:
        try:
            q = typer.prompt("\nyou")
        except (EOFError, typer.Abort):
            break
        if q.strip().lower() in {"exit", "quit"}:
            break
        with console.status("[dim]reasoning...[/dim]"):
            answer = agent.ask(q, session=session)
        console.print(f"\n[bold]theta[/bold] {answer.text}")


if __name__ == "__main__":
    app()
