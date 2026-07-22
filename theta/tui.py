"""theta top — live fleet TUI.

An htop for GPU thermal health: one card per GPU with junction temperature,
power, utilization, effective thermal resistance (R_theta) and its recent
history as a sparkline, plus a rolling alert feed.

Two attach modes:

  theta top                     # local GPUs via the collector (demo mode
                                # with synthetic telemetry when NVML is absent)
  theta top --url host:9101     # attach to any running agent's Prometheus
                                # endpoint — including one running inside
                                # Kubernetes via `kubectl port-forward`

The remote mode deliberately consumes the agent's public metrics surface
(the same one Prometheus scrapes) rather than a private API: if the TUI can
render it, any dashboard can.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Protocol

from rich.text import Text

try:
    from textual.app import App, ComposeResult
    from textual.containers import Horizontal
    from textual.reactive import reactive
    from textual.widgets import Footer, Header, RichLog, Static
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "theta top requires the UI extra:  pip install 'runtheta[ui]'"
    ) from e

T_REF_FALLBACK = 25.0   # °C — assumed ambient when no baseline is available
MIN_POWER_W = 15.0      # R_theta is unstable as P→0 (F9); gate the display

# ── data model ────────────────────────────────────────────────────────────────


@dataclass
class GpuReading:
    """One GPU's current state, source-agnostic (local collector or remote scrape)."""
    index: int
    name: str = "GPU"
    temp_c: float = 0.0
    power_w: float = 0.0
    util_pct: float = 0.0
    rtheta: Optional[float] = None
    drift_sigma: Optional[float] = None
    clock_eff: Optional[float] = None
    schedulable: Optional[bool] = None
    ecc_dbit: int = 0
    node: str = ""                   # set in fleet (multi-node) mode

    @property
    def uid(self) -> str:
        """Stable widget-id-safe identity, unique across nodes."""
        if not self.node:
            return f"gpu{self.index}"
        slug = "".join(ch if ch.isalnum() else "-" for ch in self.node)
        return f"gpu-{slug}-{self.index}"

    @property
    def label(self) -> str:
        return f"{self.node}:GPU {self.index}" if self.node else f"GPU {self.index}"


@dataclass
class FleetSample:
    gpus: List[GpuReading]
    source: str                      # "demo" | "nvml" | url
    alerts: List[str] = field(default_factory=list)


class FleetProvider(Protocol):
    """Structural contract every telemetry source fulfils (local, remote, fake)."""

    async def sample(self) -> FleetSample: ...


class LocalProvider:
    """Reads GPUs through the agent's own collector (demo mode off-GPU)."""

    def __init__(self) -> None:
        from .agent.collector import (  # lazy: pulls pynvml
            CollectorConfig,
            NVMLCollector,
        )
        self._collector = NVMLCollector(CollectorConfig())
        self._started = False

    async def sample(self) -> FleetSample:
        if not self._started:
            await self._collector.__aenter__()   # binds NVML (or enters demo mode)
            self._started = True
        self._names = self._collector.gpu_names
        raws = await self._collector.collect_all()
        gpus = []
        for r in raws:
            rtheta = None
            if r.power_w >= MIN_POWER_W:
                rtheta = (r.temp_junction - T_REF_FALLBACK) / r.power_w
            eff = (r.clock_sm_mhz / r.sm_clock_max_mhz) if r.sm_clock_max_mhz else None
            gpus.append(GpuReading(
                index=r.gpu_index,
                name=self._names[r.gpu_index] if r.gpu_index < len(self._names) else "GPU",
                temp_c=r.temp_junction, power_w=r.power_w, util_pct=r.util_pct,
                rtheta=rtheta, clock_eff=eff, ecc_dbit=r.ecc_dbit,
            ))
        source = "demo" if getattr(self._collector, "_demo_mode", False) else "nvml"
        return FleetSample(gpus=gpus, source=source)


class RemoteProvider:
    """Scrapes a running agent's Prometheus endpoint and re-derives per-GPU state."""

    _SERIES = {
        "theta_gpu_temperature_celsius": "temp_c",
        "theta_gpu_power_watts": "power_w",
        "theta_gpu_rtheta_cwatt": "rtheta",
        "theta_gpu_drift_sigma": "drift_sigma",
        "theta_gpu_clock_efficiency_ratio": "clock_eff",
    }

    def __init__(self, url: str) -> None:
        if "://" not in url:
            url = f"http://{url}"
        self.url = url.rstrip("/") + ("" if url.endswith("/metrics") else "/metrics")

    async def sample(self) -> FleetSample:
        import httpx
        async with httpx.AsyncClient(timeout=5.0) as client:
            text = (await client.get(self.url)).text

        gpus: Dict[int, GpuReading] = {}

        def gpu(idx: int) -> GpuReading:
            return gpus.setdefault(idx, GpuReading(index=idx))

        for line in text.splitlines():
            if not line.startswith("theta_gpu_"):
                continue
            metric, _, rest = line.partition("{")
            labels, _, value = rest.partition("} ")
            idx = _label(labels, "gpu_index")
            if idx is None:
                continue
            try:
                v = float(value)
            except ValueError:
                continue
            if metric in self._SERIES:
                setattr(gpu(idx), self._SERIES[metric], v)
            elif metric == "theta_gpu_utilization_ratio":
                gpu(idx).util_pct = v * 100.0
            elif metric == "theta_gpu_schedulable":
                gpu(idx).schedulable = bool(v)
            elif metric == "theta_gpu_ecc_dbit_total":
                gpu(idx).ecc_dbit = int(v)

        return FleetSample(gpus=[gpus[i] for i in sorted(gpus)], source=self.url)


def _label(labels: str, key: str) -> Optional[int]:
    marker = f'{key}="'
    start = labels.find(marker)
    if start < 0:
        return None
    end = labels.find('"', start + len(marker))
    try:
        return int(labels[start + len(marker):end])
    except ValueError:
        return None


# ── alert derivation (display-level; the agent's detector is the authority) ──


def derive_alerts(g: GpuReading) -> List[str]:
    alerts = []
    who = g.label
    if g.ecc_dbit > 0:
        alerts.append(f"[b red]CRITICAL[/] {who}: {g.ecc_dbit} uncorrectable ECC errors")
    if g.schedulable is False:
        alerts.append(f"[b red]CRITICAL[/] {who}: marked unfit for new work")
    if g.temp_c >= 85:
        alerts.append(f"[b yellow]WARN[/] {who}: junction {g.temp_c:.0f}°C")
    if g.drift_sigma is not None and g.drift_sigma >= 3:
        alerts.append(f"[b yellow]WARN[/] {who}: R_θ drift +{g.drift_sigma:.1f}σ from baseline")
    if g.clock_eff is not None and g.clock_eff < 0.85 and g.util_pct > 90:
        alerts.append(f"[b yellow]WARN[/] {who}: micro-throttle, SM at {g.clock_eff:.0%} of boost under load")
    return alerts


# ── widgets ───────────────────────────────────────────────────────────────────

from .tuifx import (  # noqa: E402  (kept beside the widgets they serve)
    DemoFleet,
    HeatSource,
    PolishView,
    ThermalField,
    heat_color,
    hgx_positions,
    polish,
    polish_rows,
    sparkline,
)


def temp_color(t: float) -> str:
    return "green" if t < 70 else ("yellow" if t < 82 else "red")


Frame = Dict[str, List[GpuReading]]   # node → readings, one instant


def _to_frame(sample: FleetSample) -> Frame:
    frame: Frame = {}
    for g in sample.gpus:
        frame.setdefault(g.node or "local", []).append(g)
    for readings in frame.values():
        readings.sort(key=lambda g: g.index)
    return frame


class DemoFleetProvider:
    """4-node demo fleet driven by tuifx.DemoFleet — measured D9 physics plus
    a unit that slowly degrades, so the interface demonstrates a detection."""

    def __init__(self) -> None:
        self._fleet = DemoFleet()

    async def sample(self) -> FleetSample:
        self._fleet.tick(1.0)
        snap = self._fleet.sample()
        gpus: List[GpuReading] = []
        for (node, slot), m in snap.items():
            gpus.append(GpuReading(
                index=slot, name="A100-SXM4-80GB (demo)", node=node,
                temp_c=m["temp_c"], power_w=m["power_w"], util_pct=m["util_pct"],
                rtheta=m["rtheta"],
            ))
        return FleetSample(gpus=gpus, source="demo")


class FieldPanel(Static):
    """The node as a physical thermal object — braille heat field, GPUs at
    their real slot positions (F17/F19/F21: position IS physics)."""

    DEFAULT_CSS = """
    FieldPanel { border: round $surface-lighten-2; padding: 0 1; height: 100%; }
    FieldPanel.flagged { border: round red; }
    """

    def show(self, node: str, readings: List[GpuReading],
             flagged: Dict[int, bool]) -> None:
        w = max(24, (self.size.width or 48) - 4)
        h = max(6, (self.size.height or 12) - 3)
        pos = hgx_positions(len(readings))
        sources = [
            HeatSource(x=pos[i][0], y=pos[i][1], temp_c=g.temp_c,
                       label=str(g.index), power_w=g.power_w,
                       flagged=flagged.get(g.index, False))
            for i, g in enumerate(readings)
        ]
        field = ThermalField(w, h)
        rows = field.render(sources, airflow="")
        body = Text()
        for r in rows:
            body.append_text(r)
            body.append("\n")
        watts = sum(g.power_w for g in readings)
        body.append(f"→ airflow    {node}   {len(readings)} GPUs   {watts:,.0f} W",
                    style="dim")
        self.set_class(any(flagged.values()), "flagged")
        self.border_title = f"thermal field · {node}"
        self.update(body)


class PolishPanel(Static):
    """Live two-way median polish — the E009 fleet method AS the interface."""

    DEFAULT_CSS = """
    PolishPanel { border: round $surface-lighten-2; padding: 0 1; height: 100%; }
    """

    def show(self, frame: Frame) -> Optional[PolishView]:
        cells: Dict[tuple, float] = {}
        for node, readings in frame.items():
            for g in readings:
                if g.rtheta is not None:
                    cells[(node, g.index)] = g.rtheta
        self.border_title = "median polish · residual z"
        if len(frame) < 2:
            self.update(Text(
                "single node — board-position structure is unresolvable at\n"
                "node scope (F21). Attach a fleet (--k8s / demo fleet [f])\n"
                "for position-conditioned residuals.\n\n"
                "The naive within-node view can flag a healthy hot-half GPU\n"
                "at z>8; theta refuses to pretend otherwise.", style="dim"))
            return None
        view = polish(cells)
        body = Text()
        for r in polish_rows(view):
            body.append_text(r)
            body.append("\n")
        body.append(f"grand μ {view.grand*1000:.1f} mC/W   "
                    "cells = residual z after node+slot effects removed",
                    style="dim")
        self.update(body)
        return view


class TrajectoryPanel(Static):
    """Per-GPU R_theta trajectories: braille strips with the baseline band."""

    DEFAULT_CSS = """
    TrajectoryPanel { border: round $surface-lighten-2; padding: 0 1; height: 100%; }
    """

    def show(self, node: str, readings: List[GpuReading],
             history: Dict[str, "Deque[float]"],
             flagged: Dict[int, bool]) -> None:
        self.border_title = f"R_θ trajectories · {node}"
        width = max(16, ((self.size.width or 80) - 14))
        body = Text()
        for g in readings:
            h = history.get(g.uid)
            vals = list(h) if h else []
            r_now = f"{g.rtheta*1000:5.1f}" if g.rtheta is not None else "  n/a"
            hot = flagged.get(g.index, False)
            color = "#ff5555" if hot else heat_color(
                min(1.0, max(0.0, (g.temp_c - 30) / 62)))
            label = f"{g.index} {r_now} "
            if vals:
                lo, hi = min(vals), max(vals)
                pad = (hi - lo) * 0.15 + 1e-6
                strips = sparkline(vals, width, height=1,
                                   lo=lo - pad, hi=hi + pad, color=color)
                body.append(label, style="bold" if hot else "")
                body.append_text(strips[0])
            else:
                body.append(label + "…", style="dim")
            body.append("\n")
        body.append("mC/W · window = buffered history · [f]leet demo  [space] pause  [←/→] scrub",
                    style="dim")
        self.update(body)


class FleetBar(Static):
    DEFAULT_CSS = "FleetBar { height: 1; padding: 0 2; background: $surface; }"

    def update_fleet(self, s: FleetSample, paused: bool, scrub: str = "") -> None:
        worst = max((g.temp_c for g in s.gpus), default=0.0)
        c = temp_color(worst)
        state = "[b yellow]PAUSED[/]" if paused else "[b green]LIVE[/]"
        if scrub:
            state = f"[b magenta]REPLAY {scrub}[/]"
        source = "[b yellow]DEMO — synthetic telemetry (D9-measured physics)[/]" \
            if s.source == "demo" else f"source [b]{s.source}[/]"
        self.update(
            f"{state}  {source}  ·  {len(s.gpus)} GPUs  ·  "
            f"worst junction [{c}]{worst:.1f}°C[/]  ·  "
            f"{time.strftime('%H:%M:%S')}"
        )


class Timeline(Static):
    """Scrub bar: buffered frames with alert marks; ←/→ moves the cursor."""

    DEFAULT_CSS = "Timeline { height: 1; padding: 0 2; background: $surface-darken-1; }"

    def show(self, n_frames: int, pos: Optional[int], marks: set) -> None:
        width = max(10, (self.size.width or 80) - 12)
        if n_frames <= 1:
            self.update(Text("timeline —", style="dim"))
            return
        cursor = (pos if pos is not None else n_frames - 1)
        t = Text("⏱ ", style="dim")
        for i in range(width):
            fi = int(i * (n_frames - 1) / max(1, width - 1))
            ch, style = "·", "dim"
            if any(abs(m - fi) <= max(1, n_frames // width) for m in marks):
                ch, style = "▲", "red"
            if fi == cursor or (pos is None and i == width - 1):
                ch, style = "█", ("magenta" if pos is not None else "green")
            t.append(ch, style=style)
        t.append("  live" if pos is None else f"  −{n_frames - 1 - cursor}s·frames",
                 style="dim")
        self.update(t)


# ── app ───────────────────────────────────────────────────────────────────────


class ThetaTopApp(App):
    """Fleet thermal forensics console.

    Not another gauge grid: the node is rendered as a physical thermal object
    (braille heat field, GPUs at their true slot positions), the E009 median
    polish runs live as the centerpiece (F21: position-conditioned fleet
    scoring is the correct scope), every R_theta trajectory is a braille strip
    with its baseline, and the whole session is scrubbable — ←/→ replays the
    buffered history, alert frames marked on the timeline.
    """

    TITLE = "theta top"
    SUB_TITLE = "GPU thermal-power forensics — live"
    CSS = """
    #upper { height: 14; layout: horizontal; }
    #field-panel { width: 58%; }
    #polish-panel { width: 42%; }
    #trajectories { height: 1fr; min-height: 6; }
    #alerts { height: 6; border-top: heavy $surface-lighten-2;
              background: $surface-darken-1; padding: 0 1; }
    """
    BINDINGS = [
        ("q", "quit", "Quit"),
        ("space", "toggle_pause", "Pause"),
        ("p", "toggle_pause", "Pause"),
        ("left", "scrub(-1)", "Back"),
        ("right", "scrub(1)", "Fwd"),
        ("n", "next_node", "Node"),
        ("f", "demo_fleet", "Demo fleet"),
    ]

    paused = reactive(False)

    def __init__(self, provider: Optional[FleetProvider] = None, interval: float = 2.0) -> None:
        super().__init__()
        self.provider: FleetProvider = provider or LocalProvider()
        self.interval = interval
        self._seen_alerts: Deque[str] = deque(maxlen=200)
        self._frames: Deque[FleetSample] = deque(maxlen=600)
        self._marks: set = set()          # frame indices that carried alerts
        self._scrub_pos: Optional[int] = None
        self._node_idx = 0
        self._history: Dict[str, Deque[float]] = {}

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield FleetBar()
        with Horizontal(id="upper"):
            yield FieldPanel(id="field-panel")
            yield PolishPanel(id="polish-panel")
        yield TrajectoryPanel(id="trajectories")
        yield RichLog(id="alerts", markup=True, wrap=True)
        yield Timeline()
        yield Footer()

    async def on_mount(self) -> None:
        self.query_one("#alerts", RichLog).write(
            "[dim]alert feed — display-level triage; the agent's detector pipeline is authoritative[/]"
        )
        await self.refresh_fleet()
        self.set_interval(self.interval, self.refresh_fleet)

    # ── data plumbing ────────────────────────────────────────────────────────

    async def refresh_fleet(self) -> None:
        if self.paused:
            return
        try:
            sample = await self.provider.sample()
        except Exception as e:  # keep the UI alive through scrape failures
            self.query_one("#alerts", RichLog).write(f"[red]sample failed:[/] {e}")
            return
        self._frames.append(sample)
        for g in sample.gpus:
            if g.rtheta is not None:
                self._history.setdefault(g.uid, deque(maxlen=360)).append(g.rtheta)
        if self._scrub_pos is None:
            self._render(sample, live=True)

    def _flagged(self, sample: FleetSample) -> Dict[str, Dict[int, bool]]:
        """Fleet-scope flags from the live polish (multi-node), else display
        triage only — never fake node-scope peer verdicts (F21)."""
        frame = _to_frame(sample)
        flags: Dict[str, Dict[int, bool]] = {n: {} for n in frame}
        if len(frame) >= 2:
            cells = {(n, g.index): g.rtheta for n, rs in frame.items()
                     for g in rs if g.rtheta is not None}
            if len(cells) >= 8:
                view = polish(cells)
                for (n, slot), z in view.residual_z.items():
                    if abs(z) >= 4.0:
                        flags[n][slot] = True
        return flags

    def _render(self, sample: FleetSample, live: bool) -> None:
        frame = _to_frame(sample)
        nodes = sorted(frame)
        if not nodes:
            return
        self._node_idx %= len(nodes)
        node = nodes[self._node_idx]
        flags = self._flagged(sample)

        self.query_one(FieldPanel).show(node, frame[node], flags.get(node, {}))
        self.query_one(PolishPanel).show(frame)
        self.query_one(TrajectoryPanel).show(
            node, frame[node], self._history, flags.get(node, {}))
        pos = self._scrub_pos
        scrub = "" if (live or pos is None) else f"frame {pos + 1}/{len(self._frames)}"
        self.query_one(FleetBar).update_fleet(sample, self.paused, scrub)
        self.query_one(Timeline).show(len(self._frames), self._scrub_pos, self._marks)

        if live:
            log = self.query_one("#alerts", RichLog)
            fired = False
            for note in sample.alerts:
                if note not in self._seen_alerts:
                    self._seen_alerts.append(note)
                    log.write(f"{time.strftime('%H:%M:%S')}  {note}")
                    fired = True
            for g in sample.gpus:
                for alert in derive_alerts(g):
                    if alert not in self._seen_alerts:
                        self._seen_alerts.append(alert)
                        log.write(f"{time.strftime('%H:%M:%S')}  {alert}")
                        fired = True
            for n, d in flags.items():
                for slot, on in d.items():
                    if on:
                        key = f"polish:{n}:{slot}"
                        if key not in self._seen_alerts:
                            self._seen_alerts.append(key)
                            log.write(
                                f"{time.strftime('%H:%M:%S')}  [b red]FLEET[/] "
                                f"{n}:GPU {slot} — polish residual |z| ≥ 4 "
                                f"(position-conditioned, E009 method)")
                            fired = True
            if fired:
                self._marks.add(len(self._frames) - 1)

    # ── actions ──────────────────────────────────────────────────────────────

    def action_toggle_pause(self) -> None:
        self.paused = not self.paused

    def action_scrub(self, delta: int) -> None:
        if not self._frames:
            return
        if self._scrub_pos is None:
            self._scrub_pos = len(self._frames) - 1
        self._scrub_pos += delta
        if self._scrub_pos >= len(self._frames) - 1:
            self._scrub_pos = None          # walked back to live
            self._render(self._frames[-1], live=True)
            return
        self._scrub_pos = max(0, self._scrub_pos)
        self._render(self._frames[self._scrub_pos], live=False)

    def action_next_node(self) -> None:
        self._node_idx += 1
        if self._frames:
            self._render(self._frames[self._scrub_pos if self._scrub_pos is not None
                                      else -1],
                         live=self._scrub_pos is None)

    def action_demo_fleet(self) -> None:
        """Switch to the D9-seeded demo fleet (screenshots, demos, no GPU)."""
        self.provider = DemoFleetProvider()
        self._frames.clear()
        self._history.clear()
        self._marks.clear()
        self._scrub_pos = None


class MultiNodeProvider:
    """Merges every node's agent into one fleet view.

    One RemoteProvider per (node, endpoint); nodes are sampled concurrently
    and a node that fails to scrape is reported in the alert feed while the
    rest of the fleet keeps rendering — one dead tunnel must never blank the
    console.
    """

    def __init__(self, endpoints: List[tuple[str, str]]) -> None:
        # endpoints: [(node_name, url)]
        self._providers = [(node, RemoteProvider(url)) for node, url in endpoints]

    async def sample(self) -> FleetSample:
        import asyncio

        results = await asyncio.gather(
            *(p.sample() for _, p in self._providers), return_exceptions=True
        )
        gpus: List[GpuReading] = []
        alerts: List[str] = []
        for (node, _), res in zip(self._providers, results, strict=True):
            if isinstance(res, BaseException):
                alerts.append(f"[red]node {node}: scrape failed[/] — {res}")
                continue
            for g in res.gpus:
                g.node = node
                gpus.append(g)
        return FleetSample(
            gpus=gpus,
            source=f"k8s fleet · {len(self._providers)} nodes",
            alerts=alerts,
        )


def run(url: Optional[str] = None, interval: float = 2.0) -> None:
    provider = RemoteProvider(url) if url else LocalProvider()
    ThetaTopApp(provider=provider, interval=interval).run()


def _free_port() -> int:
    import socket
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _wait_port(port: int, timeout_s: float = 5.0) -> bool:
    import socket
    for _ in range(int(timeout_s / 0.1)):
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.1):
                return True
        except OSError:
            time.sleep(0.1)
    return False


def run_k8s(namespace: str = "default", selector: str = "app.kubernetes.io/component=agent",
            interval: float = 2.0) -> None:
    """Zero-config fleet attach for Kubernetes.

    Discovers EVERY running Theta agent pod, opens one `kubectl port-forward`
    per pod, and renders the whole fleet in a single console — cards are
    labeled by node. All tunnels are torn down on exit. One command, no
    second terminal, no per-node juggling.
    """
    import json as _json
    import shutil
    import subprocess
    import sys

    if shutil.which("kubectl") is None:
        sys.exit("theta top --k8s requires kubectl on PATH")

    out = subprocess.run(
        ["kubectl", "get", "pods", "-n", namespace, "-l", selector,
         "--field-selector", "status.phase=Running", "-o", "json"],
        capture_output=True, text=True,
    ).stdout
    pods = [
        (item["metadata"]["name"], item["spec"].get("nodeName", "?"))
        for item in (_json.loads(out).get("items", []) if out else [])
    ]
    if not pods:
        sys.exit(f"no running Theta agent pods found (namespace={namespace!r}, selector={selector!r})\n"
                 "install one:  helm install theta deploy/helm/theta")

    forwards: List = []
    endpoints: List[tuple[str, str]] = []
    try:
        for pod, node in pods:
            port = _free_port()
            forwards.append(subprocess.Popen(
                ["kubectl", "port-forward", "-n", namespace, f"pod/{pod}", f"{port}:9101"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            ))
            if not _wait_port(port):
                sys.exit(f"port-forward to {pod} never became ready")
            endpoints.append((node, f"127.0.0.1:{port}"))

        provider = MultiNodeProvider(endpoints) if len(endpoints) > 1 \
            else RemoteProvider(endpoints[0][1])
        ThetaTopApp(provider=provider, interval=interval).run()
    finally:
        for fwd in forwards:
            fwd.terminate()
