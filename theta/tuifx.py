"""
Rendering core for the theta TUI — the parts no terminal GPU monitor has.

BrailleCanvas
  A 2D sub-cell drawing surface: each terminal cell is a 2x4 braille dot
  matrix (U+2800 block), giving 8x the plotting resolution of character
  graphics, with an independent 24-bit foreground color per cell. Used for
  the thermal field and the R_theta trajectory strips.

ThermalField
  Renders a node as a physical object: GPUs are heat sources placed at their
  real slot positions (airflow order — F17/F19/F21: position IS physics), and
  the space between them is a continuous scalar field (inverse-distance
  interpolation), colored on a perceptual heat ramp. The result reads like a
  CFD cross-section, live in the terminal.

polish_table
  Tukey two-way median polish (node x slot) rendered as a live table: margin
  effects (node effect, slot effect) and residual robust-z per cell — the
  E009 fleet method AS the interface, not behind it.

DemoFleet
  A physically-plausible animated fleet seeded from the measured D9 campaign
  values (8x A100 HGX: cool board half 72-80 mC/W, hot half 104-110 mC/W at
  matched power; idle floors 36/43 C) plus one unit slowly drifting toward
  degradation — so the no-hardware demo shows real physics and a detection
  actually happening.

Everything here is pure computation + Rich renderables: no Textual imports,
no GPU imports, fully testable in CI.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from rich.style import Style
from rich.text import Text

# ── color ramps ───────────────────────────────────────────────────────────────

# Perceptual heat ramp (deep blue → cyan → green → yellow → orange → red → white)
_HEAT_STOPS: List[Tuple[float, Tuple[int, int, int]]] = [
    (0.00, (13, 17, 38)),
    (0.15, (28, 60, 120)),
    (0.35, (32, 144, 140)),
    (0.55, (76, 190, 82)),
    (0.70, (222, 205, 58)),
    (0.85, (240, 128, 48)),
    (0.95, (235, 62, 54)),
    (1.00, (255, 244, 235)),
]

# Diverging ramp for residual z (blue = below peers, gray = nominal, red = above)
_DIV_STOPS: List[Tuple[float, Tuple[int, int, int]]] = [
    (0.00, (58, 110, 235)),
    (0.50, (60, 64, 72)),
    (1.00, (235, 62, 54)),
]


def _lerp(a: Tuple[int, int, int], b: Tuple[int, int, int], t: float) -> Tuple[int, int, int]:
    return (
        int(a[0] + (b[0] - a[0]) * t),
        int(a[1] + (b[1] - a[1]) * t),
        int(a[2] + (b[2] - a[2]) * t),
    )


def ramp(stops: List[Tuple[float, Tuple[int, int, int]]], t: float) -> Tuple[int, int, int]:
    t = max(0.0, min(1.0, t))
    for (t0, c0), (t1, c1) in zip(stops, stops[1:], strict=False):
        if t <= t1:
            span = (t1 - t0) or 1e-9
            return _lerp(c0, c1, (t - t0) / span)
    return stops[-1][1]


def heat_color(t: float) -> str:
    r, g, b = ramp(_HEAT_STOPS, t)
    return f"#{r:02x}{g:02x}{b:02x}"


def diverging_color(z: float, z_span: float = 6.0) -> str:
    r, g, b = ramp(_DIV_STOPS, 0.5 + max(-1.0, min(1.0, z / z_span)) / 2.0)
    return f"#{r:02x}{g:02x}{b:02x}"


# ── braille canvas ────────────────────────────────────────────────────────────

# Braille dot bit layout within one cell (col, row) → bit
_DOT_BITS = {
    (0, 0): 0x01, (0, 1): 0x02, (0, 2): 0x04, (0, 3): 0x40,
    (1, 0): 0x08, (1, 1): 0x10, (1, 2): 0x20, (1, 3): 0x80,
}


class BrailleCanvas:
    """A width x height CELL canvas addressed in (2*width) x (4*height) dots.

    Each cell carries a color; setting a dot ORs its bit and (optionally)
    updates the cell color — last writer wins, which is the right semantics
    for plotting bright signal over dim background.
    """

    def __init__(self, width: int, height: int) -> None:
        self.width = max(1, width)
        self.height = max(1, height)
        self._bits: List[int] = [0] * (self.width * self.height)
        self._color: List[Optional[str]] = [None] * (self.width * self.height)

    @property
    def dot_width(self) -> int:
        return self.width * 2

    @property
    def dot_height(self) -> int:
        return self.height * 4

    def set_dot(self, x: int, y: int, color: Optional[str] = None) -> None:
        if not (0 <= x < self.dot_width and 0 <= y < self.dot_height):
            return
        cell = (y // 4) * self.width + (x // 2)
        self._bits[cell] |= _DOT_BITS[(x % 2, y % 4)]
        if color is not None:
            self._color[cell] = color

    def fill_cell(self, cx: int, cy: int, color: str, bits: int = 0xFF) -> None:
        if not (0 <= cx < self.width and 0 <= cy < self.height):
            return
        cell = cy * self.width + cx
        self._bits[cell] |= bits
        self._color[cell] = color

    def line(self, x0: int, y0: int, x1: int, y1: int, color: Optional[str] = None) -> None:
        """Bresenham in dot space."""
        dx, dy = abs(x1 - x0), -abs(y1 - y0)
        sx, sy = (1 if x0 < x1 else -1), (1 if y0 < y1 else -1)
        err = dx + dy
        while True:
            self.set_dot(x0, y0, color)
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 >= dy:
                err += dy
                x0 += sx
            if e2 <= dx:
                err += dx
                y0 += sy

    def rows(self, default_style: str = "") -> List[Text]:
        out: List[Text] = []
        for cy in range(self.height):
            t = Text()
            for cx in range(self.width):
                cell = cy * self.width + cx
                ch = chr(0x2800 + self._bits[cell]) if self._bits[cell] else " "
                style = self._color[cell] or default_style
                t.append(ch, style=style)
            out.append(t)
        return out


def sparkline(
    values: Sequence[float],
    width: int,
    height: int = 2,
    lo: Optional[float] = None,
    hi: Optional[float] = None,
    color: str = "#4cbe52",
    band: Optional[Tuple[float, float]] = None,
    band_color: str = "#2a2f3a",
) -> List[Text]:
    """Braille time-series strip with an optional baseline band behind it."""
    canvas = BrailleCanvas(width, height)
    vals = list(values)[-canvas.dot_width:]
    if not vals:
        return canvas.rows()
    vlo = lo if lo is not None else min(vals)
    vhi = hi if hi is not None else max(vals)
    if vhi - vlo < 1e-9:
        vhi = vlo + 1e-9

    def ydot(v: float) -> int:
        frac = (v - vlo) / (vhi - vlo)
        return int(round((1.0 - max(0.0, min(1.0, frac))) * (canvas.dot_height - 1)))

    if band is not None:
        by0, by1 = sorted((ydot(band[0]), ydot(band[1])))
        for x in range(canvas.dot_width):
            for y in range(by0, by1 + 1):
                canvas.set_dot(x, y, band_color)

    x0 = canvas.dot_width - len(vals)
    prev: Optional[Tuple[int, int]] = None
    for i, v in enumerate(vals):
        pt = (x0 + i, ydot(v))
        if prev is not None:
            canvas.line(prev[0], prev[1], pt[0], pt[1], color)
        else:
            canvas.set_dot(pt[0], pt[1], color)
        prev = pt
    return canvas.rows()


# ── the thermal field ─────────────────────────────────────────────────────────

@dataclass
class HeatSource:
    """One GPU as a physical heat source in board space (0..1 x 0..1)."""
    x: float
    y: float
    temp_c: float
    label: str
    power_w: float = 0.0
    flagged: bool = False


def hgx_positions(n: int) -> List[Tuple[float, float]]:
    """Physical slot layout: two board halves (F21's measured cool/hot split),
    airflow left→right, slots front-to-back within each half."""
    pos: List[Tuple[float, float]] = []
    half = max(1, n // 2)
    for i in range(n):
        board = 0 if i < half else 1
        k = i if board == 0 else i - half
        x = 0.14 + 0.72 * (k / max(1, half - 1)) if half > 1 else 0.5
        y = 0.28 if board == 0 else 0.72
        pos.append((x, y))
    return pos


class ThermalField:
    """Continuous temperature field over the board from point heat sources."""

    def __init__(self, width: int, height: int, t_floor: float = 30.0, t_ceil: float = 92.0) -> None:
        self.width = width
        self.height = height
        self.t_floor = t_floor
        self.t_ceil = t_ceil

    def render(self, sources: List[HeatSource], airflow: str = "→ airflow") -> List[Text]:
        canvas = BrailleCanvas(self.width, self.height)
        span = (self.t_ceil - self.t_floor) or 1.0

        # Inverse-distance-squared interpolation of the temperature field.
        for cy in range(self.height):
            for cx in range(self.width):
                fx = (cx + 0.5) / self.width
                fy = (cy + 0.5) / self.height
                num = 0.0
                den = 0.0
                for s in sources:
                    d2 = (fx - s.x) ** 2 + ((fy - s.y) * 0.9) ** 2 + 1e-4
                    w = 1.0 / d2
                    num += w * s.temp_c
                    den += w
                t = (num / den - self.t_floor) / span if den else 0.0
                t = max(0.0, min(1.0, t))
                # dot density encodes intensity too — cool zones render sparse,
                # hot zones dense, so the field reads even without color.
                bits = 0x00
                if t > 0.10:
                    bits |= 0x02 | 0x10
                if t > 0.30:
                    bits |= 0x04 | 0x20
                if t > 0.50:
                    bits |= 0x01 | 0x08
                if t > 0.70:
                    bits |= 0x40 | 0x80
                canvas.fill_cell(cx, cy, heat_color(t), bits or 0x02)

        # Character overlay: source markers + labels stamped over the field.
        overlay: Dict[Tuple[int, int], Tuple[str, Style]] = {}
        for s in sources:
            cx = min(self.width - 1, int(s.x * self.width))
            cy = min(self.height - 1, int(s.y * self.height))
            frac = max(0.0, min(1.0, (s.temp_c - self.t_floor) / span))
            marker = "◉" if s.flagged else "●"
            style = Style(color="#ff5555", bold=True) if s.flagged \
                else Style(color=heat_color(frac), bold=True)
            label = f"{marker}{s.label}"
            start = max(0, min(self.width - len(label), cx - 1))
            for i, ch in enumerate(label):
                overlay[(cy, start + i)] = (ch, style)

        rows: List[Text] = []
        for cy in range(self.height):
            t = Text()
            for cx in range(self.width):
                ov = overlay.get((cy, cx))
                if ov is not None:
                    t.append(ov[0], style=ov[1])
                else:
                    cell = cy * canvas.width + cx
                    bits = canvas._bits[cell]
                    ch = chr(0x2800 + bits) if bits else " "
                    t.append(ch, style=canvas._color[cell] or "")
            rows.append(t)

        footer = Text(airflow, style="dim")
        rows.append(footer)
        return rows


# ── median polish as a live table ─────────────────────────────────────────────

@dataclass
class PolishView:
    nodes: List[str]
    slots: List[int]
    grand: float
    node_effect: Dict[str, float]
    slot_effect: Dict[int, float]
    residual_z: Dict[Tuple[str, int], float]


def polish(fleet: Dict[Tuple[str, int], float], iterations: int = 12,
           rel_floor: float = 0.04) -> PolishView:
    """Pure-python Tukey two-way median polish (no numpy — TUI stays light).

    fleet: {(node, slot): rtheta} at comparable load. Returns margin effects
    and per-cell residual robust-z. Mirrors peer.median_polish_z semantics.
    """
    nodes = sorted({n for n, _ in fleet})
    slots = sorted({s for _, s in fleet})
    resid = dict(fleet)
    grand = _median(list(resid.values()))
    for k in resid:
        resid[k] -= grand
    node_eff = {n: 0.0 for n in nodes}
    slot_eff = {s: 0.0 for s in slots}

    for _ in range(iterations):
        for n in nodes:
            row = [resid[(n, s)] for s in slots if (n, s) in resid]
            if not row:
                continue
            m = _median(row)
            node_eff[n] += m
            for s in slots:
                if (n, s) in resid:
                    resid[(n, s)] -= m
        for s in slots:
            col = [resid[(n, s)] for n in nodes if (n, s) in resid]
            if not col:
                continue
            m = _median(col)
            slot_eff[s] += m
            for n in nodes:
                if (n, s) in resid:
                    resid[(n, s)] -= m

    vals = list(resid.values())
    rmed = _median(vals)
    mad = _median([abs(v - rmed) for v in vals]) or 1e-9
    sigma = max(1.4826 * mad, rel_floor * abs(grand) if grand else 1e-9)
    z = {k: v / sigma for k, v in resid.items()}
    return PolishView(nodes=nodes, slots=slots, grand=grand,
                      node_effect=node_eff, slot_effect=slot_eff, residual_z=z)


def _median(xs: List[float]) -> float:
    ys = sorted(xs)
    n = len(ys)
    if n == 0:
        return 0.0
    mid = n // 2
    return ys[mid] if n % 2 else (ys[mid - 1] + ys[mid]) / 2.0


def polish_rows(view: PolishView, cell_w: int = 6) -> List[Text]:
    """Render a PolishView: slots across, nodes down, margins showing effects,
    cells colored by residual z on the diverging ramp."""
    rows: List[Text] = []
    head = Text("node".ljust(9), style="bold dim")
    for s in view.slots:
        head.append(f"s{s}".center(cell_w), style="bold dim")
    head.append("  node-fx", style="bold dim")
    rows.append(head)

    for n in view.nodes:
        line = Text(n[:8].ljust(9), style="bold")
        for s in view.slots:
            z = view.residual_z.get((n, s))
            if z is None:
                line.append("·".center(cell_w), style="dim")
            else:
                mag = abs(z)
                txt = f"{z:+.1f}".center(cell_w)
                style = Style(color=diverging_color(z), bold=mag >= 3.0,
                              reverse=mag >= 4.0)
                line.append(txt, style=style)
        fx = view.node_effect[n] * 1000
        line.append(f"  {fx:+5.1f}", style="cyan dim")
        rows.append(line)

    foot = Text("slot-fx  ", style="bold dim")
    for s in view.slots:
        fx = view.slot_effect[s] * 1000
        foot.append(f"{fx:+.0f}".center(cell_w), style="magenta dim")
    foot.append("  mC/W", style="dim")
    rows.append(foot)
    return rows


# ── demo fleet (D9-seeded physics, with a story) ──────────────────────────────

class DemoFleet:
    """Animated 4-node x 8-GPU fleet whose numbers come from measured reality:
    D9's board-half split (cool 72-80, hot 104-110 mC/W at ~400 W) and idle
    floors (36/43 C). Node n3 slot 4 slowly degrades — the demo shows the
    polish matrix catching what the naive view cannot, live.
    """

    NODES = ("node-a", "node-b", "node-c", "node-d")
    SLOTS = tuple(range(8))
    # measured healthy R_theta by slot (C/W), D9 campaign, matched ~400 W
    BASE_R = (0.0725, 0.0757, 0.0798, 0.0789, 0.1042, 0.1090, 0.1084, 0.1102)

    def __init__(self) -> None:
        self.t = 0.0
        self.victim = ("node-c", 4)

    def tick(self, dt: float = 1.0) -> None:
        self.t += dt

    def degradation(self) -> float:
        """Victim's extra R_theta, ramping in over ~3 minutes of demo time."""
        return min(0.035, max(0.0, (self.t - 20.0) * 0.0003))

    def sample(self) -> Dict[Tuple[str, int], Dict[str, float]]:
        out: Dict[Tuple[str, int], Dict[str, float]] = {}
        for ni, node in enumerate(self.NODES):
            for s in self.SLOTS:
                r = self.BASE_R[s]
                r *= 1.0 + 0.015 * math.sin(0.13 * self.t + ni * 1.7 + s * 0.9)
                r += 0.0008 * ((ni * 3 + s) % 5 - 2)
                if (node, s) == self.victim:
                    r += self.degradation()
                power = 396.0 + 10.0 * math.sin(0.21 * self.t + s)
                temp = 36.0 + r * power
                util = 97.0 + 2.0 * math.sin(0.4 * self.t + s * 2.1)
                out[(node, s)] = {
                    "rtheta": r, "power_w": power, "temp_c": temp, "util_pct": util,
                }
        return out
