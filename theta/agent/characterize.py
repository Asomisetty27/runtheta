"""Fleet characterization report — the shareable HTML artifact.

`theta characterize` turns one telemetry export into the deliverable a
validation engineer hands over: fleet summary, peer-relative findings with
cause attribution, the fleet R_θ(P) power-tier curve, and a per-GPU
characterization table — one self-contained HTML file, no external assets.

All analysis is reused from the validated pipeline (`jobreport` for
ingestion/steady-load stats, `partner_report` for detection + α/β signature
attribution); this module only bins the R_θ(P) curve and renders.
"""

from __future__ import annotations

import html
import statistics
from dataclasses import dataclass

from .jobreport import JobReport
from .partner_report import UnitAttribution, analyze

# E009-style fleet power tiers (W). R_θ is a curve in P — characterization
# reports the curve, never a single number.
POWER_TIERS: tuple[tuple[float, float], ...] = (
    (100.0, 250.0), (250.0, 400.0), (400.0, 550.0), (550.0, 700.0), (700.0, 1500.0),
)
DEFAULT_AMBIENT_C = 25.0
MIN_TIER_SAMPLES = 30


@dataclass(frozen=True)
class TierStat:
    lo_w: float
    hi_w: float
    median_r: float
    n: int

    @property
    def label(self) -> str:
        return f"{self.lo_w:.0f}–{self.hi_w:.0f} W"

    @property
    def mid_w(self) -> float:
        return (self.lo_w + self.hi_w) / 2.0


def power_tier_curve(aligned: dict, ambient: float = DEFAULT_AMBIENT_C) -> list[TierStat]:
    """Pool every sample across the fleet and bin R_θ by power tier.

    Absolute values carry the T_ref assumption (stated in the report footer);
    the curve's *shape* is what characterization consumes — E009 showed H100
    R_θ falls 0.120→0.0585 across 120→700 W, so tier-blind thresholds are
    meaningless.
    """
    buckets: dict[tuple[float, float], list[float]] = {t: [] for t in POWER_TIERS}
    for d in aligned.values():
        for temp, power in zip(d["T"], d["P"], strict=True):
            if power <= 0:
                continue
            for lo, hi in POWER_TIERS:
                if lo <= power < hi:
                    buckets[(lo, hi)].append((temp - ambient) / power)
                    break
    return [
        TierStat(lo, hi, statistics.median(vals), len(vals))
        for (lo, hi), vals in buckets.items()
        if len(vals) >= MIN_TIER_SAMPLES
    ]


def characterize(
    aligned: dict, label: str, generated_at: str,
) -> tuple[JobReport, list[UnitAttribution], str]:
    """End-to-end: aligned series → (report, attributions, HTML document)."""
    report, attributions, _text = analyze(aligned, label=label)
    tiers = power_tier_curve(aligned)
    doc = _render_html(report, attributions, tiers, label, generated_at)
    return report, attributions, doc


# ── rendering (stdlib only, self-contained output) ───────────────────────────

_CSS = """
body{background:#0b0f0e;color:#e8ecea;font:14px/1.5 -apple-system,'Segoe UI',sans-serif;
     max-width:960px;margin:2rem auto;padding:0 1rem}
h1{font-size:1.4rem;margin:0}  h2{font-size:1.05rem;margin:2rem 0 .6rem;color:#7ee0c0}
.sub{color:#8a938f;font-size:.85rem}
.cards{display:flex;gap:.8rem;flex-wrap:wrap;margin:1.2rem 0}
.card{background:#141a18;border:1px solid #233029;border-radius:8px;padding:.7rem 1.1rem;min-width:120px}
.card b{display:block;font-size:1.3rem}  .card span{color:#8a938f;font-size:.75rem}
table{border-collapse:collapse;width:100%;font-size:.85rem}
th,td{padding:.35rem .6rem;text-align:left;border-bottom:1px solid #233029}
th{color:#8a938f;font-weight:600;text-transform:uppercase;font-size:.7rem;letter-spacing:.06em}
.flag{color:#ff6b6b;font-weight:700}  .watch{color:#e8b23a;font-weight:700}  .ok{color:#35c792}
.mono{font-family:ui-monospace,monospace}
footer{color:#8a938f;font-size:.75rem;margin:2.5rem 0 1rem;border-top:1px solid #233029;padding-top:1rem}
"""


def _esc(s: object) -> str:
    return html.escape(str(s))


def _svg_curve(tiers: list[TierStat]) -> str:
    """Inline SVG polyline of the fleet R_θ(P) curve."""
    if len(tiers) < 2:
        return ""
    w, h, pad = 640, 180, 42
    xs = [t.mid_w for t in tiers]
    ys = [t.median_r for t in tiers]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    yspan = (y1 - y0) or 1e-9

    def px(x: float) -> float:
        return pad + (x - x0) / (x1 - x0) * (w - 2 * pad)

    def py(y: float) -> float:
        return h - pad - (y - y0) / yspan * (h - 2 * pad)

    pts = " ".join(f"{px(x):.1f},{py(y):.1f}" for x, y in zip(xs, ys, strict=True))
    dots = "".join(
        f'<circle cx="{px(x):.1f}" cy="{py(y):.1f}" r="3.5" fill="#7ee0c0"/>'
        f'<text x="{px(x):.1f}" y="{py(y) - 9:.1f}" fill="#8a938f" font-size="10" '
        f'text-anchor="middle">{y:.4f}</text>'
        for x, y in zip(xs, ys, strict=True)
    )
    xlabels = "".join(
        f'<text x="{px(t.mid_w):.1f}" y="{h - 14}" fill="#8a938f" font-size="10" '
        f'text-anchor="middle">{t.label}</text>'
        for t in tiers
    )
    return (
        f'<svg viewBox="0 0 {w} {h}" role="img" style="width:100%;max-width:{w}px">'
        f'<polyline points="{pts}" fill="none" stroke="#7ee0c0" stroke-width="2"/>'
        f"{dots}{xlabels}"
        f'<text x="{pad}" y="16" fill="#8a938f" font-size="10">fleet median R_θ (°C/W) by power tier</text>'
        "</svg>"
    )


def _findings_rows(attributions: list[UnitAttribution]) -> str:
    if not attributions:
        return '<tr><td colspan="5" class="ok">No peer-relative outliers detected.</td></tr>'
    rows = []
    for a in attributions:
        v = a.verdict
        cause = v.headline_cause.name if v.headline_cause else "—"
        precision = "exact" if v.identifiable and v.discriminated else (
            "lean" if v.top else "insufficient")
        supporting = "; ".join(v.top.supporting) if v.top and v.top.supporting else ""
        rows.append(
            f'<tr><td class="mono">{_esc(a.key)}</td>'
            f'<td class="{"flag" if a.tier == "FLAG" else "watch"}">{a.tier}</td>'
            f'<td class="mono">{a.robust_z:+.2f}σ</td>'
            f'<td>{_esc(cause)} <span class="sub">({precision})</span></td>'
            f"<td>{_esc(supporting)}</td></tr>"
        )
    return "".join(rows)


def _pergpu_rows(report: JobReport) -> str:
    z_by_key = {**report.watch, **report.flagged}
    rows = []
    for s in sorted(report.gpus, key=lambda g: -(z_by_key.get(g.key, 0.0))):
        z = z_by_key.get(s.key)
        cv = (s.r_std / s.r_mean * 100.0) if s.r_mean else 0.0
        status = ('<span class="flag">FLAG</span>' if s.key in report.flagged else
                  '<span class="watch">WATCH</span>' if s.key in report.watch else
                  '<span class="ok">nominal</span>')
        rows.append(
            f'<tr><td class="mono">{_esc(s.key)}</td><td>{s.n}</td>'
            f"<td>{s.p_mean:.0f} W</td><td>{s.t_mean:.1f} °C</td>"
            f'<td class="mono">{s.r_mean:.4f}</td><td>{cv:.1f}%</td>'
            f'<td class="mono">{f"{z:+.2f}σ" if z is not None else "—"}</td>'
            f"<td>{status}</td></tr>"
        )
    return "".join(rows)


def _render_html(
    report: JobReport,
    attributions: list[UnitAttribution],
    tiers: list[TierStat],
    label: str,
    generated_at: str,
) -> str:
    tier_rows = "".join(
        f'<tr><td>{_esc(t.label)}</td><td class="mono">{t.median_r:.4f}</td><td>{t.n}</td></tr>'
        for t in tiers
    )
    fleet_r = f"{report.fleet_mean_r:.4f}" if report.fleet_mean_r else "—"
    notes = "".join(f"<li>{_esc(n)}</li>" for n in report.notes)
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>Theta characterization — {_esc(label)}</title>
<style>{_CSS}</style></head><body>
<h1>Theta fleet characterization</h1>
<div class="sub">{_esc(label)} · generated {_esc(generated_at)} · detection: {_esc(report.method)}</div>

<div class="cards">
<div class="card"><b>{len(report.gpus)}</b><span>GPUs characterized</span></div>
<div class="card"><b>{len(report.nodes)}</b><span>nodes</span></div>
<div class="card"><b class="mono">{fleet_r}</b><span>fleet mean R_θ (°C/W)</span></div>
<div class="card"><b class="flag">{len(report.flagged)}</b><span>flagged</span></div>
<div class="card"><b class="watch">{len(report.watch)}</b><span>watch</span></div>
</div>

<h2>Findings — peer-relative outliers with cause attribution</h2>
<table><tr><th>GPU</th><th>tier</th><th>robust z</th><th>attribution</th><th>supporting evidence</th></tr>
{_findings_rows(attributions)}</table>

<h2>Fleet R_θ(P) power-tier curve</h2>
{_svg_curve(tiers)}
<table><tr><th>power tier</th><th>median R_θ (°C/W)</th><th>samples</th></tr>{tier_rows}</table>

<h2>Per-GPU characterization</h2>
<table><tr><th>GPU</th><th>samples</th><th>P̄</th><th>T̄</th><th>R̄_θ</th><th>CV</th><th>robust z</th><th>status</th></tr>
{_pergpu_rows(report)}</table>

{f"<h2>Notes</h2><ul>{notes}</ul>" if notes else ""}
<footer>Detection is peer-relative (T_ref cancels); α/β attribution is fleet-relative —
both are invariant to the absolute ambient assumption. Absolute R_θ magnitudes assume
T_ref = {DEFAULT_AMBIENT_C:.0f} °C. Generated by <span class="mono">theta characterize</span>
(github.com/Asomisetty27/theta).</footer>
</body></html>"""
