"""Self-contained HTML report: every available question, grouped by stage."""
from __future__ import annotations

import html
import math
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from .questions import QUESTIONS, PLANNED, STAGES, answer, available

PALETTE = ["#0f6b74", "#c2672c", "#5b5fa8", "#3f8f4e", "#b8456b", "#8a7a2c", "#4f7fb3", "#7a5c99"]


def _e(x: Any) -> str:
    return html.escape("" if x is None else str(x))


# ---------------------------------------------------------------- charts --
def _bar(spec: Dict[str, Any]) -> str:
    labels, values = spec["labels"][:15], [float(v) for v in spec["values"][:15]]
    if not values:
        return ""
    w, row, left = 640, 24, 210
    h = row * len(values) + 20
    lo, hi = min(0.0, min(values)), max(0.0, max(values))
    span = (hi - lo) or 1.0
    x0 = left + (0 - lo) / span * (w - left - 60)
    out = [f'<svg viewBox="0 0 {w} {h}" class="chart" role="img" aria-label="{_e(spec.get("title"))}">']
    for k, (lab, v) in enumerate(zip(labels, values)):
        y = 10 + k * row
        x1 = left + (v - lo) / span * (w - left - 60)
        out.append(f'<text x="{left - 8}" y="{y + 15}" text-anchor="end" class="lab">{_e(str(lab)[:32])}</text>')
        out.append(f'<rect x="{min(x0, x1):.1f}" y="{y + 3}" width="{max(1, abs(x1 - x0)):.1f}" height="{row - 8}" '
                   f'rx="2" fill="{PALETTE[0] if v >= 0 else PALETTE[1]}"/>')
        out.append(f'<text x="{max(x0, x1) + 6:.1f}" y="{y + 15}" class="val">{v:g}</text>')
    out.append(f'<line x1="{x0:.1f}" x2="{x0:.1f}" y1="6" y2="{h - 6}" class="axis"/></svg>')
    return "".join(out)


def _line(spec: Dict[str, Any]) -> str:
    xs = [float(x) for x in spec["x"]]
    series = list(spec["series"].items())[:8]
    if len(xs) < 2 or not series:
        return ""
    w, h, l, r, t, b = 640, 260, 48, 16, 14, 34
    ys = [float(v) for _, s in series for v in s]
    lo, hi = min(0.0, min(ys)), max(ys) or 1.0
    X = lambda x: l + (x - xs[0]) / ((xs[-1] - xs[0]) or 1) * (w - l - r)
    Y = lambda y: t + (1 - (y - lo) / ((hi - lo) or 1)) * (h - t - b)
    out = [f'<svg viewBox="0 0 {w} {h}" class="chart" role="img" aria-label="{_e(spec.get("title"))}">']
    for k in range(5):
        yv = lo + (hi - lo) * k / 4
        out.append(f'<line x1="{l}" x2="{w - r}" y1="{Y(yv):.1f}" y2="{Y(yv):.1f}" class="grid"/>'
                   f'<text x="{l - 6}" y="{Y(yv) + 4:.1f}" text-anchor="end" class="val">{yv:.3g}</text>')
    step = max(1, len(xs) // 8)
    for x in xs[::step]:
        out.append(f'<text x="{X(x):.1f}" y="{h - 12}" text-anchor="middle" class="val">{x:g}</text>')
    for k, (name, s) in enumerate(series):
        pts = " ".join(f"{X(x):.1f},{Y(float(v)):.1f}" for x, v in zip(xs, s))
        out.append(f'<polyline points="{pts}" fill="none" stroke="{PALETTE[k % 8]}" stroke-width="2"/>')
    out.append("</svg>")
    legend = "".join(f'<span><i style="background:{PALETTE[k % 8]}"></i>{_e(n)}</span>'
                     for k, (n, _) in enumerate(series)) if len(series) > 1 else ""
    return "".join(out) + (f'<div class="legend">{legend}</div>' if legend else "")


def _heat(spec: Dict[str, Any]) -> str:
    rows, cols, M = spec["rows"][:12], spec["cols"][:12], spec["matrix"]
    vals = [float(v) for r in M[:12] for v in r[:12]]
    if not vals:
        return ""
    lo, hi = min(vals), max(vals)
    cells = []
    for i, rn in enumerate(rows):
        tds = []
        for j in range(len(cols)):
            v = float(M[i][j])
            a = 0.08 + 0.85 * ((v - lo) / ((hi - lo) or 1))
            tds.append(f'<td style="background:rgba(15,107,116,{a:.2f});color:{"#fff" if a > 0.55 else "inherit"}">{v:g}</td>')
        cells.append(f"<tr><th>{_e(str(rn)[:28])}</th>{''.join(tds)}</tr>")
    head = "".join(f"<th>{_e(str(c)[:14])}</th>" for c in cols)
    return f'<div class="tw"><table class="heat"><tr><th></th>{head}</tr>{"".join(cells)}</table></div>'


def _chart(spec: Optional[Dict[str, Any]]) -> str:
    if not spec:
        return ""
    body = {"bar": _bar, "line": _line, "heatmap": _heat}[spec["type"]](spec)
    return f'<figure><figcaption>{_e(spec.get("title"))}</figcaption>{body}</figure>' if body else ""


def _table(name: str, df: pd.DataFrame, max_rows: int = 20) -> str:
    d = df.head(max_rows)
    head = "".join(f"<th>{_e(c)}</th>" for c in d.columns)
    body = []
    for _, r in d.iterrows():
        tds = []
        for v in r:
            if isinstance(v, float):
                txt = "" if math.isnan(v) else f"{v:.4g}"
                tds.append(f'<td class="n">{txt}</td>')
            elif isinstance(v, (int,)) and not isinstance(v, bool):
                tds.append(f'<td class="n">{v}</td>')
            else:
                tds.append(f"<td>{_e(v)}</td>")
        body.append(f"<tr>{''.join(tds)}</tr>")
    more = f'<p class="more">Showing {len(d)} of {len(df)} rows. Full table: export or res.answer().</p>' \
        if len(df) > max_rows else ""
    return f'<details open><summary>{_e(name)}</summary><div class="tw"><table>' \
           f'<tr>{head}</tr>{"".join(body)}</table></div>{more}</details>'


CSS = """
:root{--bg:#f6f8f9;--fg:#16232a;--mut:#5a6a72;--line:#dce3e7;--card:#fff;--acc:#0f6b74;--soft:#e3f0f1}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#0f171b;--fg:#e3ebee;--mut:#9aabb3;--line:#26363f;--card:#16222a;--acc:#5fbcc4;--soft:#173238;color-scheme:dark}}
:root[data-theme="dark"]{--bg:#0f171b;--fg:#e3ebee;--mut:#9aabb3;--line:#26363f;--card:#16222a;--acc:#5fbcc4;--soft:#173238;color-scheme:dark}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:16px/1.55 "Segoe UI",system-ui,-apple-system,sans-serif}
.wrap{max-width:1100px;margin:0 auto;padding-block:32px;padding-left:max(16px,env(safe-area-inset-left,0px));padding-right:max(16px,env(safe-area-inset-right,0px))}
h1{font-size:2rem;margin:0 0 4px}h2{font-size:1.35rem;margin:40px 0 12px;padding-top:12px;border-top:2px solid var(--acc)}
h3{font-size:1.1rem;margin:0}.mut{color:var(--mut)}
nav{display:grid;grid-template-columns:repeat(auto-fill,minmax(250px,1fr));gap:4px 20px;margin:20px 0;font-size:.9rem}
nav a{color:var(--fg);text-decoration:none}nav a:hover{color:var(--acc)}nav b{color:var(--acc);font-family:ui-monospace,Menlo,monospace;font-size:.8rem;margin-right:6px}
.q{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:18px 20px;margin:14px 0;display:flex;flex-direction:column;gap:10px}
.qh{display:flex;gap:10px;align-items:baseline;flex-wrap:wrap}.id{font-family:ui-monospace,Menlo,monospace;color:var(--acc);font-weight:600}
.pill{font-size:.72rem;text-transform:uppercase;letter-spacing:.05em;padding:2px 8px;border-radius:99px;background:var(--soft);color:var(--acc)}
.meta{font-size:.85rem;color:var(--mut)}ul.f{margin:0;padding-left:20px}ul.f li{margin:3px 0}
figure{margin:0}figcaption{font-size:.85rem;color:var(--mut);margin-bottom:4px}
.chart{width:100%;max-width:680px;height:auto}.chart text{font-size:11px;fill:var(--mut)}.chart .lab{fill:var(--fg)}
.chart .axis{stroke:var(--mut)}.chart .grid{stroke:var(--line)}
.legend{display:flex;flex-wrap:wrap;gap:12px;font-size:.8rem;color:var(--mut)}.legend i{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:5px}
details summary{cursor:pointer;font-size:.9rem;font-weight:600;margin:4px 0}
.tw{overflow-x:auto;max-width:100%}table{border-collapse:collapse;font-size:.82rem;width:100%}
th,td{border-bottom:1px solid var(--line);padding:5px 8px;text-align:left;vertical-align:top}th{color:var(--mut);font-weight:600}
td.n{text-align:right;font-variant-numeric:tabular-nums}table.heat td{text-align:center;min-width:46px}
.more{font-size:.8rem;color:var(--mut);margin:4px 0}.na{opacity:.7}
footer{margin-top:40px;font-size:.85rem;color:var(--mut)}
"""


def build_report(res, path: Union[str, Path], title: Optional[str] = None,
                 qids: Optional[List[str]] = None) -> Path:
    path = Path(path).expanduser()
    title = title or "S3 analysis of the field"
    n = len(res.papers)
    yr = res.papers["year"].dropna()
    span = f"{int(yr.min())}–{int(yr.max())}" if len(yr) else "n/a"
    nav, body = [], []
    stage_open = None
    for q in QUESTIONS:
        if qids and q.qid not in qids:
            continue
        if q.stage != stage_open:
            stage_open = q.stage
            body.append(f'<h2 id="s{q.stage}">{q.stage}. {_e(STAGES[q.stage])}</h2>')
        ok, why = available(res, q)
        nav.append(f'<a href="#{q.qid}"><b>{q.qid}</b>{_e(q.text)}</a>')
        head = (f'<div class="qh"><span class="id">{q.qid}</span><h3>{_e(q.text)}</h3>'
                f'<span class="pill">{q.priority}</span></div>'
                f'<div class="meta">{_e(q.why)} · Method: {_e(q.method)}</div>')
        if not ok:
            body.append(f'<section class="q na" id="{q.qid}">{head}<p class="mut">Not available: {_e(why)}</p></section>')
            continue
        try:
            a = answer(res, q.qid)
        except Exception as ex:  # keep the report going
            body.append(f'<section class="q na" id="{q.qid}">{head}<p class="mut">Could not compute: {_e(ex)}</p></section>')
            continue
        findings = "".join(f"<li>{_e(f)}</li>" for f in a.findings)
        tables = "".join(_table(k, v) for k, v in a.tables.items())
        body.append(f'<section class="q" id="{q.qid}">{head}<ul class="f">{findings}</ul>'
                    f'{_chart(a.chart)}{tables}</section>')
    thr = f"{res.best['threshold']:.2f}" if res.best else "n/a"
    planned = "".join(f"<li>{_e(p)} <span class='mut'>({_e(s)})</span></li>" for p, s in PLANNED)
    doc = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>{_e(title)}</title><style>{CSS}</style></head>
<body><div class="wrap">
<h1>{_e(title)}</h1>
<p class="mut">{n} papers · {span} · best S3 threshold {thr} ·
{int(res.papers['community'].notna().sum())} clustered papers in {len(res.communities())} communities · generated {date.today()} with s3-bibliometric</p>
<nav>{''.join(nav)}</nav>
{''.join(body)}
<h2>Planned analyses</h2><ul>{planned}</ul>
<footer>S3 (Semantic Similarity Score) bibliometric method · github.com/HamidBekamiri/s3-bibliometric</footer>
</div></body></html>"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(doc, encoding="utf-8")
    return path
