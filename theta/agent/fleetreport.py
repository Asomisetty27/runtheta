"""
Self-contained HTML fleet health report — the shareable artifact.

One HTML file, zero external assets, that an operator can attach to a ticket,
drop in Slack, or forward upward: per-GPU peer-relative R_θ standings for a
fleet snapshot, with the flags front and center and the honest scope stated.

Produced by:
  * ``theta fleet-scan <export> --html report.html``  — your fleet, this method
  * ``theta demo --html report.html``                 — the bundled real-incident replay

Honesty invariants (same as `theta demo` — do not "improve" these away):
  * every number in the report comes from the caller's computed rows;
  * the scope/limits paragraph ALWAYS renders — a health record that hides
    its own error bars is marketing, and this is not that;
  * ground-truth badges only appear when the caller passes them, and the badge
    text states HOW each unit was confirmed (RMA replacement vs field
    re-measurement) — the two kinds are never merged into one claim.
"""
from __future__ import annotations

import html as _html
from dataclasses import dataclass, field
from datetime import datetime, timezone

from .. import __version__

Z_FLAG = 3.0
Z_CRIT = 8.0


@dataclass
class ReportRow:
    gpu: str
    node: str
    temp_c: float | None
    power_w: float | None
    rtheta: float
    z: float


@dataclass
class ReportInput:
    title: str
    fleet_desc: str                 # one line: what this snapshot is
    rows: list[ReportRow]
    ref_power: float | None         # matched-load band center (W), if power data
    temp_threshold_c: float | None  # absolute alert threshold used for the temp view
    scope_note: str                 # the honest-scope paragraph (ALWAYS rendered)
    badges: dict[str, tuple[str, str]] = field(default_factory=dict)   # gpu -> (label, note)
    unconfirmed: dict[str, str] = field(default_factory=dict)          # gpu -> note


def _esc(s: object) -> str:
    return _html.escape(str(s))


def _verdict(z: float) -> tuple[str, str]:
    if z >= Z_CRIT:
        return "CRITICAL", "crit"
    if z >= Z_FLAG:
        return "anomaly", "warn"
    return "", "ok"


def render_fleet_report(inp: ReportInput) -> str:
    """Render the report as one self-contained HTML document."""
    rows = sorted(inp.rows, key=lambda r: -r.z)
    flagged = [r for r in rows if r.z >= Z_FLAG]
    nodes = {r.node for r in rows}
    temp_alerts = (
        sum(1 for r in rows if r.temp_c is not None and inp.temp_threshold_c is not None
            and r.temp_c >= inp.temp_threshold_c)
        if inp.temp_threshold_c is not None else None
    )
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    chips = [
        (f"{len(rows)}", "GPUs scanned"),
        (f"{len(nodes)}", "nodes"),
        (f"{len(flagged)}", f"flagged at z ≥ {Z_FLAG:g}"),
    ]
    if inp.ref_power is not None:
        chips.append((f"~{inp.ref_power:.0f} W", "matched load"))
    if temp_alerts is not None:
        chips.append((f"{temp_alerts}", f"over {inp.temp_threshold_c:.0f}°C threshold"))

    def row_html(r: ReportRow) -> str:
        verdict, cls = _verdict(r.z)
        badge = ""
        if r.gpu in inp.badges:
            lbl, note = inp.badges[r.gpu]
            badge = f'<span class="badge confirm" title="{_esc(note)}">{_esc(lbl)}</span>'
        elif r.gpu in inp.unconfirmed:
            badge = f'<span class="badge open" title="{_esc(inp.unconfirmed[r.gpu])}">unconfirmed</span>'
        t = f"{r.temp_c:.1f}" if r.temp_c is not None else "·"
        p = f"{r.power_w:.0f}" if r.power_w is not None else "·"
        return (
            f'<tr class="{cls}"><td class="mono">{_esc(r.gpu)}</td>'
            f'<td class="num">{t}</td><td class="num">{p}</td>'
            f'<td class="num">{r.rtheta:.4f}</td><td class="num z">{r.z:+.1f}</td>'
            f'<td>{verdict} {badge}</td></tr>'
        )

    flag_rows = "\n".join(row_html(r) for r in flagged) or (
        '<tr><td colspan="6" class="quiet">No units above the flag threshold. '
        "Fleet looks uniform after position correction.</td></tr>"
    )
    all_rows = "\n".join(row_html(r) for r in rows)
    chip_html = "\n".join(
        f'<div class="chip"><div class="v">{_esc(v)}</div><div class="l">{_esc(lbl)}</div></div>'
        for v, lbl in chips
    )

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_esc(inp.title)}</title>
<style>
  :root {{ --bg:#0a0a0e; --s1:#111117; --border:#23232d; --text:#e8e8f0; --muted:#9a9aa8;
           --faint:#61616e; --gold:#d4af37; --green:#50c878; --amber:#e0a53c; --red:#e05c4f; }}
  * {{ box-sizing:border-box; margin:0; }}
  body {{ background:var(--bg); color:var(--text); font:14px/1.6 -apple-system,'Segoe UI',Inter,sans-serif; padding:40px 16px; }}
  .wrap {{ max-width:880px; margin:0 auto; }}
  .mono, td.num, .z {{ font-family:ui-monospace,'SF Mono',Menlo,Consolas,monospace; }}
  header {{ display:flex; justify-content:space-between; align-items:baseline; gap:12px; flex-wrap:wrap; margin-bottom:6px; }}
  h1 {{ font-size:21px; font-weight:600; letter-spacing:-.02em; }}
  .brand {{ font-family:ui-monospace,Menlo,monospace; font-size:11px; color:var(--gold); }}
  .desc {{ color:var(--muted); font-size:13px; margin-bottom:22px; }}
  .chips {{ display:flex; gap:2px; flex-wrap:wrap; border-radius:6px; overflow:hidden; margin-bottom:26px; }}
  .chip {{ background:var(--s1); border:1px solid var(--border); padding:12px 18px; min-width:120px; }}
  .chip .v {{ font-size:20px; font-weight:600; color:var(--gold); font-variant-numeric:tabular-nums; }}
  .chip .l {{ font-size:10px; letter-spacing:.08em; text-transform:uppercase; color:var(--muted); margin-top:2px; }}
  h2 {{ font-size:12px; letter-spacing:.14em; text-transform:uppercase; color:var(--muted); margin:26px 0 10px; }}
  table {{ width:100%; border-collapse:collapse; background:var(--s1); border:1px solid var(--border); border-radius:6px; overflow:hidden; }}
  th {{ text-align:left; font-size:10px; letter-spacing:.1em; text-transform:uppercase; color:var(--faint); padding:8px 12px; border-bottom:1px solid var(--border); }}
  td {{ padding:7px 12px; border-bottom:1px solid var(--border); font-size:13px; }}
  td.num {{ text-align:right; font-variant-numeric:tabular-nums; }}
  tr:last-child td {{ border-bottom:none; }}
  tr.crit td.z, tr.crit td:last-child {{ color:var(--red); font-weight:600; }}
  tr.warn td.z, tr.warn td:last-child {{ color:var(--amber); }}
  .badge {{ font-size:10px; padding:1px 7px; border-radius:3px; border:1px solid; margin-left:6px; }}
  .badge.confirm {{ color:var(--green); border-color:var(--green); }}
  .badge.open {{ color:var(--muted); border-color:var(--border); }}
  .quiet {{ color:var(--muted); }}
  details {{ margin-top:10px; }}
  summary {{ cursor:pointer; color:var(--muted); font-size:12px; }}
  .scope {{ margin-top:26px; border:1px solid var(--border); border-left:3px solid var(--gold); border-radius:4px;
            padding:14px 16px; color:var(--muted); font-size:12.5px; }}
  .scope b {{ color:var(--text); }}
  footer {{ margin-top:26px; padding-top:14px; border-top:1px solid var(--border);
            display:flex; justify-content:space-between; gap:12px; flex-wrap:wrap;
            font-size:11px; color:var(--faint); }}
  footer a {{ color:var(--gold); text-decoration:none; }}
</style>
</head>
<body><div class="wrap">
  <header><h1>{_esc(inp.title)}</h1><div class="brand">θ fleet health record</div></header>
  <p class="desc">{_esc(inp.fleet_desc)}</p>
  <div class="chips">{chip_html}</div>

  <h2>Flagged units · peer-relative R_θ · position-conditioned</h2>
  <table>
    <thead><tr><th>GPU</th><th style="text-align:right">T (°C)</th><th style="text-align:right">P (W)</th>
    <th style="text-align:right">R_θ (C/W)</th><th style="text-align:right">robust-z</th><th>verdict</th></tr></thead>
    <tbody>{flag_rows}</tbody>
  </table>

  <details><summary>Full fleet standings ({len(rows)} GPUs, sorted by robust-z)</summary>
  <table style="margin-top:10px">
    <thead><tr><th>GPU</th><th style="text-align:right">T (°C)</th><th style="text-align:right">P (W)</th>
    <th style="text-align:right">R_θ (C/W)</th><th style="text-align:right">robust-z</th><th>verdict</th></tr></thead>
    <tbody>{all_rows}</tbody>
  </table></details>

  <div class="scope"><b>Scope.</b> {_esc(inp.scope_note)}</div>

  <footer>
    <span>Generated by <a href="https://runtheta.com">theta</a> v{_esc(__version__)} · {generated}</span>
    <span><span class="mono">pip install runtheta</span> · <a href="https://pypi.org/project/runtheta/">PyPI</a></span>
  </footer>
</div></body></html>
"""
