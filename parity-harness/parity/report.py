"""Collects per-cell results and renders the conformance matrix.

Three outputs share this data: report.md (plain matrix), report.html (the
human-review artifact), and report-junit.xml (pytest-native, CI). Tests append
rows via `record(...)`, matrix cells via `record_matrix(...)`, and benchmark
rows via `record_bench(...)`. A pytest_sessionfinish hook renders everything.
"""

from __future__ import annotations

import html as _html
from dataclasses import dataclass, field
from pathlib import Path

PASS = "PASS"
FAIL = "FAIL"
GAP = "GAP"  # native diverges from omlx; recorded as a baseline gap, not a hard fail

_AXES = [
    "1. Schema conformance",
    "2. Semantic agreement",
    "3. Model-architecture matrix",
    "4. Error semantics",
    "5. Streaming framing",
    "6. Audio",
    "7. Endpoint smoke",
    "8. Prefix cache",
]


@dataclass
class Row:
    axis: str
    cell: str
    verdict: str
    note: str


@dataclass
class MatrixCell:
    model_id: str
    family: str
    family_label: str
    tier: str
    native_ok: bool
    omlx_ok: bool
    native_note: str
    omlx_note: str
    milestone: str


@dataclass
class BenchRow:
    server: str
    prompt_pp: float | None  # prompt tokens/sec
    gen_tg: float | None  # generation tokens/sec
    ttft_ms: float | None  # time to first token, ms


@dataclass
class Meta:
    generated_at: str = ""
    native_version: str = "unknown"
    omlx_version: str = "unknown"
    model_count: int = 0
    tier: str = "smoke"


@dataclass
class Report:
    rows: list[Row] = field(default_factory=list)
    matrix: list[MatrixCell] = field(default_factory=list)
    bench: list[BenchRow] = field(default_factory=list)
    meta: Meta = field(default_factory=Meta)

    def record(self, axis: str, cell: str, verdict: str, note: str = "") -> None:
        self.rows.append(Row(axis=axis, cell=cell, verdict=verdict, note=note))

    def record_matrix(self, cell: MatrixCell) -> None:
        self.matrix.append(cell)

    def record_bench(self, row: BenchRow) -> None:
        self.bench.append(row)

    # --- counts --------------------------------------------------------------

    def axis_counts(self, axis: str) -> tuple[int, int, int]:
        rows = [r for r in self.rows if r.axis == axis]
        p = sum(1 for r in rows if r.verdict == PASS)
        g = sum(1 for r in rows if r.verdict == GAP)
        f = sum(1 for r in rows if r.verdict == FAIL)
        return p, g, f

    def totals(self) -> tuple[int, int, int]:
        p = sum(1 for r in self.rows if r.verdict == PASS)
        g = sum(1 for r in self.rows if r.verdict == GAP)
        f = sum(1 for r in self.rows if r.verdict == FAIL)
        return p, g, f

    # --- markdown ------------------------------------------------------------

    def _axis_table_md(self, axis: str) -> str:
        rows = [r for r in self.rows if r.axis == axis]
        if not rows:
            return "_no cells recorded_\n"
        out = ["| Cell | Verdict | Note |", "| --- | --- | --- |"]
        for row in rows:
            note = row.note.replace("|", "\\|")
            out.append(f"| {row.cell} | {row.verdict} | {note} |")
        return "\n".join(out) + "\n"

    def _summary_md(self) -> str:
        out = ["| Axis | PASS | GAP | FAIL |", "| --- | ---: | ---: | ---: |"]
        for axis in _AXES:
            p, g, f = self.axis_counts(axis)
            out.append(f"| {axis} | {p} | {g} | {f} |")
        return "\n".join(out) + "\n"

    def render_md(self) -> str:
        m = self.meta
        parts = [
            "# MLXServe Native ⇄ omlx Parity Conformance Matrix",
            "",
            f"Generated: {m.generated_at}  ",
            f"native `mlxserve-http` ({m.native_version}) vs `omlx` ({m.omlx_version}) · "
            f"tier=**{m.tier}** · {m.model_count} models",
            "",
            "Legend: **PASS** = native matches omlx · **GAP** = native diverges "
            "(recorded, not a hard fail — the baseline distance) · **FAIL** = "
            "harness-level failure (server unreachable / crash).",
            "",
            "## Summary",
            "",
            self._summary_md(),
        ]
        for axis in _AXES:
            parts += ["", f"## {axis}", "", self._axis_table_md(axis)]
        return "\n".join(parts) + "\n"

    def write_report(self, path: Path) -> None:
        path.write_text(self.render_md())

    # --- html ----------------------------------------------------------------

    def write_html(self, path: Path) -> None:
        path.write_text(render_html(self))


# ---------------------------------------------------------------------------
# HTML rendering (self-contained; inline CSS/JS; light/dark aware)
# ---------------------------------------------------------------------------

def _esc(text: str) -> str:
    return _html.escape(str(text))


_VERDICT_CLASS = {PASS: "v-pass", GAP: "v-gap", FAIL: "v-fail"}


def _matrix_section_html(rep: Report) -> str:
    if not rep.matrix:
        return "<p class='muted'>No matrix cells recorded.</p>"
    rows = []
    for cell in rep.matrix:
        nat = (
            "<span class='pill ok'>🟢 pass</span>"
            if cell.native_ok
            else "<span class='pill bad'>🔴 fault</span>"
        )
        oml = (
            "<span class='pill ok'>🟢 pass</span>"
            if cell.omlx_ok
            else "<span class='pill bad'>🔴 fault</span>"
        )
        gapnote = ""
        if not cell.native_ok:
            fix = f"<span class='ms'>{_esc(cell.milestone or 'triage')}</span>"
            gapnote = f"<div class='reason'>{_esc(cell.native_note)} {fix}</div>"
        rows.append(
            f"<tr>"
            f"<td class='model'><code>{_esc(cell.model_id)}</code></td>"
            f"<td><span class='fam fam-{_esc(cell.family)}'>{_esc(cell.family_label)}</span></td>"
            f"<td class='tier'>{_esc(cell.tier)}</td>"
            f"<td class='c'>{nat}{gapnote}</td>"
            f"<td class='c'>{oml}</td>"
            f"</tr>"
        )
    return (
        "<div class='scroll'><table class='matrix'>"
        "<thead><tr><th>Model</th><th>Architecture family</th><th>Tier</th>"
        "<th>native MLXServe</th><th>omlx</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table></div>"
    )


def _axis_section_html(rep: Report, axis: str) -> str:
    rows = [r for r in rep.rows if r.axis == axis]
    if not rows:
        return "<p class='muted'>No cells recorded.</p>"
    body = []
    for row in rows:
        cls = _VERDICT_CLASS.get(row.verdict, "")
        body.append(
            f"<tr class='{cls}'>"
            f"<td>{_esc(row.cell)}</td>"
            f"<td class='verdict'><span class='badge {cls}'>{_esc(row.verdict)}</span></td>"
            f"<td class='note'>{_esc(row.note)}</td>"
            f"</tr>"
        )
    return (
        "<div class='scroll'><table class='axis'>"
        "<thead><tr><th>Cell</th><th>Verdict</th><th>Diff / note</th></tr></thead>"
        f"<tbody>{''.join(body)}</tbody></table></div>"
    )


def _bench_section_html(rep: Report) -> str:
    if not rep.bench:
        return "<p class='muted'>Benchmark not captured in this pass.</p>"

    def bar(value: float | None, vmax: float, unit: str) -> str:
        if value is None or vmax <= 0:
            return "<span class='muted'>n/a</span>"
        pct = max(2, min(100, round(100 * value / vmax)))
        return (
            f"<div class='barwrap'><div class='bar' style='width:{pct}%'></div>"
            f"<span class='barval'>{value:,.1f} {unit}</span></div>"
        )

    pp_max = max((b.prompt_pp or 0) for b in rep.bench) or 1
    tg_max = max((b.gen_tg or 0) for b in rep.bench) or 1
    ttft_max = max((b.ttft_ms or 0) for b in rep.bench) or 1
    rows = []
    for b in rep.bench:
        rows.append(
            f"<tr><td>{_esc(b.server)}</td>"
            f"<td>{bar(b.prompt_pp, pp_max, 'tok/s')}</td>"
            f"<td>{bar(b.gen_tg, tg_max, 'tok/s')}</td>"
            f"<td>{bar(b.ttft_ms, ttft_max, 'ms')}</td></tr>"
        )
    return (
        "<div class='scroll'><table class='bench'>"
        "<thead><tr><th>Server</th><th>Prefill (PP)</th><th>Decode (TG)</th>"
        "<th>TTFT</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table></div>"
    )


def _gap_list_html(rep: Report) -> str:
    # Native matrix faults grouped by milestone.
    groups: dict[str, list[MatrixCell]] = {}
    for cell in rep.matrix:
        if not cell.native_ok:
            key = cell.milestone or "triage / HTTP surface"
            groups.setdefault(key, []).append(cell)
    # Conformance/error/streaming GAP rows grouped under HTTP-surface milestone.
    surface_gaps = [
        r
        for r in rep.rows
        if r.verdict == GAP and not r.axis.startswith("3.")
    ]
    parts = []
    for milestone, cells in sorted(groups.items()):
        items = "".join(
            f"<li><code>{_esc(c.model_id)}</code> "
            f"<span class='fam fam-{_esc(c.family)}'>{_esc(c.family_label)}</span> — "
            f"{_esc(c.native_note)}</li>"
            for c in cells
        )
        parts.append(
            f"<div class='gapgroup'><h4>🔴 {_esc(milestone)}</h4><ul>{items}</ul></div>"
        )
    if surface_gaps:
        items = "".join(
            f"<li><strong>{_esc(r.axis.split('. ',1)[-1])}:</strong> "
            f"{_esc(r.cell)} — {_esc(r.note)}</li>"
            for r in surface_gaps
        )
        parts.append(
            "<div class='gapgroup'><h4>🟡 M6 HTTP surface (error envelope, "
            "usage timing, SSE framing)</h4>"
            f"<ul>{items}</ul></div>"
        )
    if not parts:
        return "<p class='muted'>No gaps — full parity.</p>"
    return "".join(parts)


def render_html(rep: Report) -> str:
    m = rep.meta
    p, g, f = rep.totals()
    total = p + g + f
    green_pct = round(100 * p / total) if total else 0
    matrix_green = sum(1 for c in rep.matrix if c.native_ok)
    matrix_total = len(rep.matrix)

    axis_cards = []
    for axis in _AXES:
        ap, ag, af = rep.axis_counts(axis)
        atot = ap + ag + af
        rate = round(100 * ap / atot) if atot else 0
        axis_cards.append(
            f"<div class='axiscard'><div class='axname'>{_esc(axis)}</div>"
            f"<div class='axrate'>{rate}%</div>"
            f"<div class='axsub'>{ap} pass · {ag} gap · {af} fail</div></div>"
        )

    axis_sections = "".join(
        f"<section class='axis-section'><h3>{_esc(axis)}</h3>{_axis_section_html(rep, axis)}</section>"
        for axis in _AXES
        if not axis.startswith("3.")
    )

    return _HTML_TEMPLATE.format(
        generated=_esc(m.generated_at),
        native_version=_esc(m.native_version),
        omlx_version=_esc(m.omlx_version),
        model_count=m.model_count,
        tier=_esc(m.tier),
        green=p,
        total=total,
        green_pct=green_pct,
        gap=g,
        fail=f,
        matrix_green=matrix_green,
        matrix_total=matrix_total,
        axis_cards="".join(axis_cards),
        matrix_section=_matrix_section_html(rep),
        axis_sections=axis_sections,
        bench_section=_bench_section_html(rep),
        gap_list=_gap_list_html(rep),
    )


_HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>MLXServe Native ⇄ omlx Parity</title>
<style>
  :root {{
    --bg:#f7f7f8; --panel:#ffffff; --ink:#1a1a1f; --muted:#6b7280;
    --line:#e5e7eb; --accent:#4f46e5; --ok:#15803d; --okbg:#dcfce7;
    --bad:#b91c1c; --badbg:#fee2e2; --warn:#a16207; --warnbg:#fef9c3;
    --code:#f3f4f6;
  }}
  @media (prefers-color-scheme: dark) {{
    :root {{
      --bg:#0e0f13; --panel:#17181d; --ink:#e7e7ea; --muted:#9aa0ac;
      --line:#2a2c33; --accent:#8b87ff; --ok:#4ade80; --okbg:#0f2e1c;
      --bad:#f87171; --badbg:#3a1414; --warn:#fbbf24; --warnbg:#332a08;
      --code:#1f2128;
    }}
  }}
  * {{ box-sizing:border-box; }}
  body {{
    margin:0; background:var(--bg); color:var(--ink);
    font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
  }}
  .wrap {{ max-width:1080px; margin:0 auto; padding:32px 20px 80px; }}
  header h1 {{ font-size:26px; margin:0 0 6px; letter-spacing:-0.02em; }}
  header .sub {{ color:var(--muted); font-size:13.5px; }}
  header .sub code {{ background:var(--code); padding:1px 6px; border-radius:5px; }}
  .banner {{
    display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr));
    gap:14px; margin:24px 0 8px;
  }}
  .stat {{
    background:var(--panel); border:1px solid var(--line); border-radius:12px;
    padding:16px 18px;
  }}
  .stat .big {{ font-size:30px; font-weight:700; letter-spacing:-0.02em; }}
  .stat .lbl {{ color:var(--muted); font-size:12.5px; text-transform:uppercase; letter-spacing:0.04em; }}
  .stat.hero .big {{ color:var(--accent); }}
  .axisgrid {{
    display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr));
    gap:12px; margin:18px 0 6px;
  }}
  .axiscard {{
    background:var(--panel); border:1px solid var(--line); border-radius:10px; padding:12px 14px;
  }}
  .axiscard .axname {{ font-size:12px; color:var(--muted); min-height:2.4em; }}
  .axiscard .axrate {{ font-size:22px; font-weight:700; }}
  .axiscard .axsub {{ font-size:11.5px; color:var(--muted); }}
  h2 {{ font-size:18px; margin:38px 0 4px; letter-spacing:-0.01em; }}
  h2 .hint {{ font-size:13px; color:var(--muted); font-weight:400; }}
  h3 {{ font-size:15px; margin:22px 0 8px; }}
  .scroll {{ overflow-x:auto; -webkit-overflow-scrolling:touch; border:1px solid var(--line); border-radius:10px; }}
  table {{ border-collapse:collapse; width:100%; font-size:13.5px; background:var(--panel); }}
  th, td {{ text-align:left; padding:9px 12px; border-bottom:1px solid var(--line); vertical-align:top; }}
  th {{ background:var(--code); font-size:12px; text-transform:uppercase; letter-spacing:0.03em; color:var(--muted); position:sticky; top:0; }}
  tbody tr:last-child td {{ border-bottom:none; }}
  code {{ font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:12.5px; }}
  .matrix td.c {{ white-space:nowrap; }}
  .pill {{ font-weight:600; font-size:13px; }}
  .pill.ok {{ color:var(--ok); }}
  .pill.bad {{ color:var(--bad); }}
  .reason {{ margin-top:5px; color:var(--muted); font-size:12px; white-space:normal; max-width:340px; }}
  .ms {{ display:inline-block; background:var(--warnbg); color:var(--warn); border-radius:5px; padding:0 6px; font-size:11px; font-weight:600; }}
  .fam {{ display:inline-block; border-radius:6px; padding:2px 8px; font-size:12px; font-weight:600; border:1px solid var(--line); }}
  .fam-dense {{ background:var(--okbg); color:var(--ok); }}
  .fam-moe {{ background:#e0e7ff; color:#4338ca; }}
  .fam-sliding-window {{ background:var(--warnbg); color:var(--warn); }}
  .fam-mamba-hybrid {{ background:var(--badbg); color:var(--bad); }}
  .fam-vlm {{ background:#f3e8ff; color:#7e22ce; }}
  @media (prefers-color-scheme: dark) {{
    .fam-moe {{ background:#1e1b4b; color:#a5b4fc; }}
    .fam-vlm {{ background:#2a1140; color:#d8b4fe; }}
  }}
  .badge {{ font-weight:700; font-size:11.5px; padding:2px 8px; border-radius:6px; }}
  .badge.v-pass {{ background:var(--okbg); color:var(--ok); }}
  .badge.v-gap {{ background:var(--warnbg); color:var(--warn); }}
  .badge.v-fail {{ background:var(--badbg); color:var(--bad); }}
  tr.v-gap td.note {{ color:var(--ink); }}
  td.note {{ color:var(--muted); font-size:12.5px; white-space:normal; }}
  .verdict {{ white-space:nowrap; }}
  .muted {{ color:var(--muted); }}
  .barwrap {{ position:relative; background:var(--code); border-radius:6px; height:22px; min-width:150px; }}
  .bar {{ position:absolute; left:0; top:0; bottom:0; background:var(--accent); border-radius:6px; opacity:0.85; }}
  .barval {{ position:relative; padding:0 8px; line-height:22px; font-size:12px; font-variant-numeric:tabular-nums; }}
  .gapgroup {{ background:var(--panel); border:1px solid var(--line); border-radius:10px; padding:6px 16px; margin:12px 0; }}
  .gapgroup h4 {{ margin:10px 0 6px; font-size:14px; }}
  .gapgroup ul {{ margin:0 0 10px; padding-left:20px; }}
  .gapgroup li {{ margin:4px 0; font-size:13px; }}
  footer {{ margin-top:48px; color:var(--muted); font-size:12px; text-align:center; }}
</style>
</head>
<body>
<div class="wrap">
  <header>
    <h1>MLXServe Native ⇄ omlx — Parity Report</h1>
    <div class="sub">
      Generated {generated} · native <code>mlxserve-http</code> ({native_version})
      vs <code>omlx</code> ({omlx_version}) · tier <strong>{tier}</strong> · {model_count} models
    </div>
  </header>

  <div class="banner">
    <div class="stat hero"><div class="big">{green}/{total}</div><div class="lbl">cells at parity ({green_pct}%)</div></div>
    <div class="stat"><div class="big">{matrix_green}/{matrix_total}</div><div class="lbl">native architecture cells green</div></div>
    <div class="stat"><div class="big">{gap}</div><div class="lbl">recorded gaps</div></div>
    <div class="stat"><div class="big">{fail}</div><div class="lbl">harness failures</div></div>
  </div>

  <div class="axisgrid">{axis_cards}</div>

  <h2>Model-architecture matrix <span class="hint">— the centerpiece: does native serve each family at all?</span></h2>
  {matrix_section}

  <h2>Conformance by axis <span class="hint">— schema · semantic · error · streaming</span></h2>
  {axis_sections}

  <h2>Benchmark <span class="hint">— native vs omlx throughput / latency</span></h2>
  {bench_section}

  <h2>Gap list <span class="hint">— every red, grouped by the milestone that closes it</span></h2>
  {gap_list}

  <footer>Differential parity harness · native MLXServe vs Python omlx · regenerated every run</footer>
</div>
</body>
</html>
"""


# One shared instance per pytest session.
REPORT = Report()
