"""Self-contained HTML report: every available question, grouped by stage."""
from __future__ import annotations

import html
import json
import math
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from .questions import QUESTIONS, PLANNED, STAGES, answer, available

# One colour per community (community 0 = largest); grey for communities outside the top 10.
PALETTE = ["#0f6b74", "#c2672c", "#5b5fa8", "#3f8f4e", "#b8456b", "#8a7a2c", "#4f7fb3", "#7a5c99",
           "#2f9c95", "#d0893a", "#a33f3f", "#3d6e91", "#7e8c2a", "#9a4f8b", "#4a8a6f"]
GREY = "#9aa7ad"


def _e(x: Any) -> str:
    return html.escape("" if x is None else str(x))


def _col(c) -> str:
    try:
        c = int(c)
    except (TypeError, ValueError):
        return GREY
    return PALETTE[c] if 0 <= c < len(PALETTE) else GREY


# ---------------------------------------------------------------- charts --
def _bar(spec: Dict[str, Any]) -> str:
    n = int(spec.get("max", 15))
    labels, values = spec["labels"][:n], [float(v) for v in spec["values"][:n]]
    if not values:
        return ""
    hl = spec.get("highlight")
    cids = spec.get("color_ids")
    w, row, left = 640, 22 if len(values) > 15 else 24, 210
    h = row * len(values) + 20
    lo, hi = min(0.0, min(values)), max(0.0, max(values))
    span = (hi - lo) or 1.0
    x0 = left + (0 - lo) / span * (w - left - 60)
    out = [f'<svg viewBox="0 0 {w} {h}" class="chart" role="img" aria-label="{_e(spec.get("title"))}">']
    for k, (lab, v) in enumerate(zip(labels, values)):
        y = 10 + k * row
        x1 = left + (v - lo) / span * (w - left - 60)
        if hl is not None:
            fill = _col(cids[k]) if (cids and k < hl) else (PALETTE[0] if k < hl else GREY)
        else:
            fill = PALETTE[0] if v >= 0 else PALETTE[1]
        out.append(f'<text x="{left - 8}" y="{y + 14}" text-anchor="end" class="lab">{_e(str(lab)[:32])}</text>')
        out.append(f'<rect x="{min(x0, x1):.1f}" y="{y + 3}" width="{max(1, abs(x1 - x0)):.1f}" height="{row - 8}" '
                   f'rx="2" fill="{fill}"/>')
        out.append(f'<text x="{max(x0, x1) + 6:.1f}" y="{y + 14}" class="val">{v:g}</text>')
        if hl is not None and k == hl - 1 and k < len(values) - 1:
            out.append(f'<line x1="{left - 200}" x2="{w}" y1="{y + row + 0.5}" y2="{y + row + 0.5}" class="cut"/>'
                       f'<text x="{w - 4}" y="{y + row - 3}" text-anchor="end" class="val">elbow: top {hl}</text>')
    out.append(f'<line x1="{x0:.1f}" x2="{x0:.1f}" y1="6" y2="{h - 6}" class="axis"/></svg>')
    return "".join(out)


def _axes(xs, lo, hi, w, h, l, r, t, b):
    X = lambda x: l + (x - xs[0]) / ((xs[-1] - xs[0]) or 1) * (w - l - r)
    Y = lambda y: t + (1 - (y - lo) / ((hi - lo) or 1)) * (h - t - b)
    out = []
    for k in range(5):
        yv = lo + (hi - lo) * k / 4
        out.append(f'<line x1="{l}" x2="{w - r}" y1="{Y(yv):.1f}" y2="{Y(yv):.1f}" class="grid"/>'
                   f'<text x="{l - 6}" y="{Y(yv) + 4:.1f}" text-anchor="end" class="val">{yv:.3g}</text>')
    step = max(1, len(xs) // 8)
    for x in xs[::step]:
        out.append(f'<text x="{X(x):.1f}" y="{h - 12}" text-anchor="middle" class="val">{x:g}</text>')
    return X, Y, out


def _legend(items) -> str:
    return '<div class="legend">' + "".join(
        f'<span><i style="background:{c}"></i>{_e(n)}</span>' for n, c in items) + "</div>"


def _line(spec: Dict[str, Any]) -> str:
    xs = [float(x) for x in spec["x"]]
    series = list(spec["series"].items())[:10]
    if len(xs) < 2 or not series:
        return ""
    w, h, l, r, t, b = 640, 260, 48, 16, 18, 34
    ys = [float(v) for _, s in series for v in s]
    lo, hi = min(0.0, min(ys)), max(ys) or 1.0
    X, Y, out = _axes(xs, lo, hi, w, h, l, r, t, b)
    out.insert(0, f'<svg viewBox="0 0 {w} {h}" class="chart" role="img" aria-label="{_e(spec.get("title"))}">')
    cids = spec.get("color_ids")
    colors = [_col(cids[k]) if cids else PALETTE[k % len(PALETTE)] for k in range(len(series))]
    for k, (name, s) in enumerate(series):
        pts = " ".join(f"{X(x):.1f},{Y(float(v)):.1f}" for x, v in zip(xs, s))
        out.append(f'<polyline points="{pts}" fill="none" stroke="{colors[k]}" stroke-width="2"/>')
        if len(series) == 1:
            out += [f'<circle cx="{X(x):.1f}" cy="{Y(float(v)):.1f}" r="2.6" fill="{colors[k]}"/>' for x, v in zip(xs, s)]
    if spec.get("mark") is not None:
        mx = X(float(spec["mark"]))
        out.append(f'<line x1="{mx:.1f}" x2="{mx:.1f}" y1="{t}" y2="{h - b}" class="cut"/>'
                   f'<text x="{mx + 5:.1f}" y="{t + 10}" class="lab">{_e(spec.get("mark_label", ""))}</text>')
    out.append("</svg>")
    return "".join(out) + (_legend([(n, colors[k]) for k, (n, _) in enumerate(series)]) if len(series) > 1 else "")


def _stack(spec: Dict[str, Any]) -> str:
    xs = [float(x) for x in spec["x"]]
    series = list(spec["series"].items())[:10]
    if len(xs) < 2 or not series:
        return ""
    w, h, l, r, t, b = 640, 280, 48, 16, 14, 34
    tot = [sum(float(s[i]) for _, s in series) for i in range(len(xs))]
    X, Y, out = _axes(xs, 0.0, max(tot) or 1.0, w, h, l, r, t, b)
    out.insert(0, f'<svg viewBox="0 0 {w} {h}" class="chart" role="img" aria-label="{_e(spec.get("title"))}">')
    cids = spec.get("color_ids")
    base = [0.0] * len(xs)
    items = []
    for k, (name, s) in enumerate(series):
        top = [base[i] + float(s[i]) for i in range(len(xs))]
        poly = [f"{X(x):.1f},{Y(v):.1f}" for x, v in zip(xs, top)] + \
               [f"{X(x):.1f},{Y(v):.1f}" for x, v in zip(reversed(xs), reversed(base))]
        c = _col(cids[k]) if cids else PALETTE[k % len(PALETTE)]
        out.append(f'<polygon points="{" ".join(poly)}" fill="{c}" fill-opacity=".8" stroke="{c}" stroke-width=".5"/>')
        items.append((name, c))
        base = top
    out.append("</svg>")
    return "".join(out) + _legend(items)


def _heat(spec: Dict[str, Any]) -> str:
    rows, cols, M = spec["rows"][:20], spec["cols"][:12], spec["matrix"]
    vals = [float(v) for r in M[:20] for v in r[:12]]
    if not vals:
        return ""
    lo, hi = min(vals), max(vals)
    cells = []
    for i, rn in enumerate(rows):
        tds = []
        for j in range(len(cols)):
            v = float(M[i][j])
            a = 0.06 + 0.85 * ((v - lo) / ((hi - lo) or 1))
            tds.append(f'<td style="background:rgba(15,107,116,{a:.2f});color:{"#fff" if a > 0.55 else "inherit"}">{v:g}</td>')
        cells.append(f"<tr><th>{_e(str(rn)[:30])}</th>{''.join(tds)}</tr>")
    head = "".join(f'<th title="{_e(c)}">{_e(str(c)[:16])}</th>' for c in cols)
    return f'<div class="tw"><table class="heat"><tr><th></th>{head}</tr>{"".join(cells)}</table></div>'


def _sankey(spec: Dict[str, Any]) -> str:
    names, cids, links = spec["names"], spec["color_ids"], spec["links"]
    k = len(names)
    out_tot = [sum(v for i, j, v in links if i == a) for a in range(k)]
    in_tot = [sum(v for i, j, v in links if j == a) for a in range(k)]
    total = max(sum(out_tot), sum(in_tot)) or 1
    w, gap, top, H = 640, 6, 24, 400
    sc = (H - top * 2 - gap * (k - 1)) / total
    xl, xr, nw = 190, 450, 12
    def stack(tots):
        y, pos = top, []
        for v in tots:
            pos.append(y)
            y += max(v * sc, 1) + gap
        return pos
    yl, yr = stack(out_tot), stack(in_tot)
    offl, offr = [0.0] * k, [0.0] * k
    paths = []
    for i, j, v in sorted(links, key=lambda x: (x[0], x[1])):
        th = v * sc
        y0 = yl[i] + offl[i] + th / 2
        y1 = yr[j] + offr[j] + th / 2
        offl[i] += th
        offr[j] += th
        xm = (xl + nw + xr) / 2
        paths.append(f'<path d="M{xl + nw},{y0:.1f} C{xm},{y0:.1f} {xm},{y1:.1f} {xr},{y1:.1f}" fill="none" '
                     f'stroke="{_col(cids[i])}" stroke-opacity=".42" stroke-width="{max(th, .6):.1f}">'
                     f'<title>{_e(names[i])} → {_e(names[j])}: {v}</title></path>')
    nodes = []
    for a in range(k):
        c = _col(cids[a])
        if out_tot[a]:
            nodes.append(f'<rect x="{xl}" y="{yl[a]:.1f}" width="{nw}" height="{max(out_tot[a] * sc, 1):.1f}" fill="{c}"/>'
                         f'<text x="{xl - 6}" y="{yl[a] + max(out_tot[a] * sc, 1) / 2 + 4:.1f}" text-anchor="end" class="lab">'
                         f'{_e(str(names[a])[:28])} ({out_tot[a]})</text>')
        if in_tot[a]:
            nodes.append(f'<rect x="{xr}" y="{yr[a]:.1f}" width="{nw}" height="{max(in_tot[a] * sc, 1):.1f}" fill="{c}"/>'
                         f'<text x="{xr + nw + 6}" y="{yr[a] + max(in_tot[a] * sc, 1) / 2 + 4:.1f}" class="lab">'
                         f'{_e(str(names[a])[:28])} ({in_tot[a]})</text>')
    hh = max(max(yl[-1] + out_tot[-1] * sc, yr[-1] + in_tot[-1] * sc) + top, 60)
    return (f'<svg viewBox="0 0 {w} {hh:.0f}" class="chart" role="img" aria-label="{_e(spec.get("title"))}">'
            f'<text x="{xl + nw / 2}" y="12" text-anchor="middle" class="val">citing</text>'
            f'<text x="{xr + nw / 2}" y="12" text-anchor="middle" class="val">cited</text>'
            + "".join(paths) + "".join(nodes) + "</svg>")


_NET_N = [0]


def _network(spec: Dict[str, Any]) -> str:
    _NET_N[0] += 1
    nid = f"s3net{_NET_N[0]}"
    data = json.dumps({"nodes": spec["nodes"], "edges": spec["edges"], "palette": PALETTE, "grey": GREY},
                      separators=(",", ":")).replace("</", "<\\/")
    legend = "".join(f'<button type="button" data-c="{g["id"]}"><i style="background:{_col(g["id"])}"></i>'
                     f'{_e(g["name"])} <span class="mut">({g["n"]})</span></button>' for g in spec["legend"])
    return (f'<div class="net"><canvas id="{nid}" aria-label="{_e(spec.get("title"))}"></canvas>'
            f'<div class="net-tip" id="{nid}-tip" hidden></div>'
            f'<div class="net-help">Scroll or use the buttons to zoom, drag to move, hover a dot for the paper. '
            f'<button type="button" data-z="1.4">+</button><button type="button" data-z="0.7">−</button>'
            f'<button type="button" data-z="0">Reset</button></div></div>'
            f'<div class="legend net-legend" id="{nid}-legend">{legend}</div>'
            f'<script type="application/json" id="{nid}-data">{data}</script>'
            f'<script>s3Net("{nid}")</script>')


NET_JS = r"""
<script>
function s3Net(id){
  const cv=document.getElementById(id), tip=document.getElementById(id+'-tip');
  const d=JSON.parse(document.getElementById(id+'-data').textContent), ctx=cv.getContext('2d');
  const N=d.nodes, E=d.edges; let W=0,H=0,dpr=1,scale=1,ox=0,oy=0,hl=null,hover=-1,drag=null,P=[];
  const xs=N.map(n=>n[0]), ys=N.map(n=>n[1]);
  const cx=(Math.min(...xs)+Math.max(...xs))/2, cy=(Math.min(...ys)+Math.max(...ys))/2;
  const sx=(Math.max(...xs)-Math.min(...xs))||1, sy=(Math.max(...ys)-Math.min(...ys))||1;
  const col=c=>(c>=0&&c<d.palette.length)?d.palette[c]:d.grey;
  function size(){dpr=window.devicePixelRatio||1;W=cv.clientWidth;H=cv.clientHeight;cv.width=W*dpr;cv.height=H*dpr;draw();}
  function place(){const s=Math.min(W/sx,H/sy)*0.92*scale;P=N.map(n=>[(n[0]-cx)*s+W/2+ox,(n[1]-cy)*s+H/2+oy]);}
  function draw(){
    if(!W)return; place(); ctx.setTransform(dpr,0,0,dpr,0,0); ctx.clearRect(0,0,W,H); ctx.lineWidth=0.6;
    for(const [a,b] of E){const ca=N[a][2],cb=N[b][2],on=hl===null||ca===hl||cb===hl;
      ctx.globalAlpha=on?(ca===cb?0.28:0.16):0.04; ctx.strokeStyle=ca===cb?col(ca):d.grey;
      ctx.beginPath();ctx.moveTo(P[a][0],P[a][1]);ctx.lineTo(P[b][0],P[b][1]);ctx.stroke();}
    for(let i=0;i<N.length;i++){const n=N[i],on=hl===null||n[2]===hl;
      ctx.globalAlpha=on?0.92:0.1;ctx.fillStyle=col(n[2]);ctx.beginPath();
      ctx.arc(P[i][0],P[i][1],n[3]*(i===hover?1.8:1)*Math.sqrt(scale),0,6.2832);ctx.fill();}
    ctx.globalAlpha=1;}
  function zoom(f,mx,my){if(f===0){scale=1;ox=0;oy=0;draw();return;}
    mx=mx??W/2;my=my??H/2;const ns=Math.min(40,Math.max(0.5,scale*f));
    ox=mx-W/2-(mx-W/2-ox)*ns/scale;oy=my-H/2-(my-H/2-oy)*ns/scale;scale=ns;draw();}
  function near(mx,my){let best=-1,bd=100;for(let i=0;i<P.length;i++){const dx=P[i][0]-mx,dy=P[i][1]-my,dd=dx*dx+dy*dy;
    if(dd<bd&&(hl===null||N[i][2]===hl)){bd=dd;best=i;}}return best;}
  function show(i,mx,my){hover=i;draw();if(i<0){tip.hidden=true;return;}const n=N[i];
    tip.textContent='';const b=document.createElement('b');b.textContent=n[4];tip.appendChild(b);
    tip.appendChild(document.createTextNode(' '+(n[5]??'')+' · '+n[6]+' citations · '+n[7]));
    tip.hidden=false;const r=tip.getBoundingClientRect();
    tip.style.left=Math.min(mx+12,W-r.width-4)+'px';tip.style.top=Math.min(my+12,H-r.height-4)+'px';}
  cv.addEventListener('wheel',e=>{e.preventDefault();zoom(e.deltaY<0?1.15:1/1.15,e.offsetX,e.offsetY);},{passive:false});
  cv.addEventListener('pointerdown',e=>{drag=[e.offsetX,e.offsetY,ox,oy];cv.setPointerCapture(e.pointerId);});
  cv.addEventListener('pointermove',e=>{if(drag&&e.buttons){ox=drag[2]+e.offsetX-drag[0];oy=drag[3]+e.offsetY-drag[1];draw();return;}
    show(near(e.offsetX,e.offsetY),e.offsetX,e.offsetY);});
  cv.addEventListener('pointerup',e=>{if(drag&&Math.abs(e.offsetX-drag[0])+Math.abs(e.offsetY-drag[1])<4)show(near(e.offsetX,e.offsetY),e.offsetX,e.offsetY);drag=null;});
  cv.addEventListener('pointerleave',()=>{if(!drag)show(-1);});
  cv.parentElement.querySelectorAll('button[data-z]').forEach(b=>b.addEventListener('click',()=>zoom(+b.dataset.z)));
  document.querySelectorAll('#'+id+'-legend button').forEach(b=>b.addEventListener('click',()=>{
    const c=+b.dataset.c;hl=hl===c?null:c;document.querySelectorAll('#'+id+'-legend button').forEach(x=>
    x.classList.toggle('off',hl!==null&&+x.dataset.c!==hl));draw();}));
  new ResizeObserver(size).observe(cv); size();
}
</script>
"""

_KT_N = [0]


def _ktree(spec: Dict[str, Any]) -> str:
    _KT_N[0] += 1
    kid = f"s3kt{_KT_N[0]}"
    data = json.dumps({k: spec[k] for k in ("windows", "communities", "names", "nodes", "links")} |
                      {"palette": PALETTE, "grey": GREY}, separators=(",", ":")).replace("</", "<\\/")
    cit = spec.get("has_citations", True)
    mode = "both" if cit else "sim"
    btn = lambda m, t: (f'<button type="button" data-mode="{m}"{" disabled" if (m != "sim" and not cit) else ""}'
                        f'{" class=on" if m == mode else ""}>{t}</button>')
    legend = "".join(f'<button type="button" data-c="{c}"><i style="background:{_col(c)}"></i>'
                     f'{_e(spec["names"][c] if c in spec["names"] else spec["names"].get(str(c), c))}</button>'
                     for c in spec["communities"])
    return (f'<div class="kt-ctl" id="{kid}-ctl">Links: {btn("cit", "Citations")}{btn("sim", "Similarity")}'
            f'{btn("both", "Both")} <label>Minimum links <input type="number" min="1" value="2" '
            f'id="{kid}-min"></label></div>'
            f'<div class="tw kt"><svg id="{kid}" class="ktree" role="img" aria-label="{_e(spec.get("title"))}"></svg></div>'
            f'<div class="legend net-legend" id="{kid}-legend">{legend}</div>'
            f'<script type="application/json" id="{kid}-data">{data}</script>'
            f'<script>s3KTree("{kid}","{mode}")</script>')


KT_JS = r"""
<script>
function s3KTree(id, mode){
  const d=JSON.parse(document.getElementById(id+'-data').textContent), svg=document.getElementById(id);
  const NS='http://www.w3.org/2000/svg', col=c=>(c>=0&&c<d.palette.length)?d.palette[c]:d.grey;
  const name=c=>d.names[c]??d.names[String(c)]??('C'+c);
  const colW=210, rowH=46, nw=30, left=20, top=40;
  const rows={}; d.communities.forEach((c,k)=>rows[c]=k);
  const maxN=Math.max(...d.nodes.map(n=>n[2]));
  const nh=n=>Math.max(3,Math.min(rowH*0.82,n*(rowH*0.82)/maxN));
  const X=w=>left+w*colW, Y=c=>top+rows[c]*rowH+rowH/2;
  let hl=null, minw=2;
  const W=left+d.windows.length*colW+40, H=top+d.communities.length*rowH+10;
  svg.setAttribute('viewBox',`0 0 ${W} ${H}`); svg.style.width=W+'px'; svg.style.height=H+'px';
  const el=(t,a,p)=>{const e=document.createElementNS(NS,t);for(const k in a)e.setAttribute(k,a[k]);(p||svg).appendChild(e);return e;};
  function draw(){
    svg.textContent='';
    d.windows.forEach((w,k)=>{const t=el('text',{x:X(k)+nw/2,y:18,'text-anchor':'middle',class:'lab'});t.textContent=w;});
    const wt=l=>mode==='cit'?l[4]:mode==='sim'?l[5]:l[4]+l[5];
    const L=d.links.filter(l=>wt(l)>=minw); const mx=Math.max(1,...L.map(wt));
    const g=el('g',{});
    for(const l of L){const [a,wa,b,wb]=l, v=wt(l), on=hl===null||a===hl||b===hl;
      const x0=X(wa)+nw, y0=Y(a), x1=X(wb), y1=Y(b), xm=(x0+x1)/2;
      const p=el('path',{d:`M${x0},${y0} C${xm},${y0} ${xm},${y1} ${x1},${y1}`,fill:'none',
        stroke:hl!==null&&on?col(a):'var(--kt-link)','stroke-width':(1+10*v/mx).toFixed(1),
        'stroke-opacity':on?(hl===null?0.6:0.8):0.06},g);
      const t=el('title',{},p); t.textContent=`${name(a)} ${d.windows[wa]} → ${name(b)} ${d.windows[wb]}: ${l[4]} citations, ${l[5]} similarity links`;}
    for(const [c,w,n] of d.nodes){const on=hl===null||c===hl, h=nh(n), x=X(w), y=Y(c)-h/2;
      const r=el('rect',{x,y,width:nw,height:h,fill:col(c),stroke:'var(--line)','fill-opacity':on?1:0.2});
      const t=el('title',{},r); t.textContent=`${name(c)} · ${d.windows[w]}: ${n} papers`;
      const last=w===d.windows.length-1;
      const lab=el('text',{x:last?x-6:x+nw+6,y:Y(c)+4,'text-anchor':last?'end':'start',class:'lab','opacity':on?1:0.25});
      lab.textContent=`${c} (N=${n})`;}
  }
  document.querySelectorAll('#'+id+'-ctl button').forEach(b=>b.addEventListener('click',()=>{
    mode=b.dataset.mode;document.querySelectorAll('#'+id+'-ctl button').forEach(x=>x.classList.toggle('on',x===b));draw();}));
  document.getElementById(id+'-min').addEventListener('input',e=>{minw=Math.max(1,+e.target.value||1);draw();});
  document.querySelectorAll('#'+id+'-legend button').forEach(b=>b.addEventListener('click',()=>{
    const c=+b.dataset.c;hl=hl===c?null:c;document.querySelectorAll('#'+id+'-legend button').forEach(x=>
      x.classList.toggle('off',hl!==null&&+x.dataset.c!==hl));draw();}));
  draw();
  const box=svg.parentElement; box.scrollLeft=box.scrollWidth;
}
</script>
"""


def _chart(spec: Optional[Dict[str, Any]]) -> str:
    if not spec:
        return ""
    fn = {"bar": _bar, "line": _line, "heatmap": _heat, "stack": _stack, "sankey": _sankey,
          "network": _network, "ktree": _ktree}.get(spec["type"])
    body = fn(spec) if fn else ""
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
.chart .cut{stroke:var(--fg);stroke-dasharray:4 3;opacity:.6}
.net{position:relative;height:min(70vh,620px);border:1px solid var(--line);border-radius:8px;background:var(--bg);overflow:hidden}
.net canvas{width:100%;height:100%;display:block;cursor:grab;touch-action:none}
.net-tip{position:absolute;max-width:320px;padding:6px 9px;font-size:.8rem;line-height:1.35;background:var(--card);border:1px solid var(--line);border-radius:6px;box-shadow:0 2px 8px rgba(0,0,0,.15);pointer-events:none}
.net-help{position:absolute;left:8px;bottom:8px;font-size:.75rem;color:var(--mut);display:flex;gap:6px;align-items:center;flex-wrap:wrap}
.net-help button,.net-legend button{font:inherit;color:inherit;background:var(--card);border:1px solid var(--line);border-radius:5px;padding:2px 8px;cursor:pointer}
.net-legend{margin-top:8px}.net-legend button{display:inline-flex;align-items:center;gap:5px;font-size:.8rem}
.net-legend button.off{opacity:.35}
.kt{border:1px solid var(--line);border-radius:8px;background:var(--bg)}
.ktree .lab{font-size:12px;fill:var(--fg)}
:root{--kt-link:#98a4aa}
.kt-ctl{display:flex;gap:6px;align-items:center;flex-wrap:wrap;font-size:.85rem;color:var(--mut);margin-bottom:6px}
.kt-ctl button{font:inherit;color:inherit;background:var(--card);border:1px solid var(--line);border-radius:5px;padding:2px 10px;cursor:pointer}
.kt-ctl button.on{background:var(--acc);border-color:var(--acc);color:var(--card)}
.kt-ctl button:disabled{opacity:.4;cursor:not-allowed}
.kt-ctl input{width:4em;font:inherit;color:inherit;background:var(--card);border:1px solid var(--line);border-radius:5px;padding:1px 4px;margin-left:4px}
.kt-ctl button:focus-visible,.net-help button:focus-visible,.net-legend button:focus-visible{outline:2px solid var(--acc);outline-offset:1px}
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
                    f'{_chart(a.chart)}{"".join(_chart(c) for c in a.charts)}{tables}</section>')
    thr = f"{res.best['threshold']:.2f}" if res.best else "n/a"
    planned = "".join(f"<li>{_e(p)} <span class='mut'>({_e(s)})</span></li>" for p, s in PLANNED)
    doc = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>{_e(title)}</title><style>{CSS}</style></head>
<body>{NET_JS}{KT_JS}<div class="wrap">
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
