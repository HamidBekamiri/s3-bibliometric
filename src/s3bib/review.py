"""The literature-review report: the S3 results organised the way a review is written.

  Brief            the field in one page: numbers, themes, what to read first, where it is going, gaps
  1 Scope          records in and out (PRISMA-style counts), papers that do not belong
  2 Themes         one card per theme: what it is about, papers to read first, foundations,
                   growth, links to other themes (+ authors, countries, journals, funders at level 2)
  3 How themes connect, 4 Evolution, 5 Who does the work (level 2), 6 Research agenda
  7 Methods appendix  a methods paragraph to adapt, a glossary, and the validation questions

Every question of the chosen level appears once, with a "For your review" note. Downloads:
reading lists (RIS for Zotero/EndNote/Mendeley, BibTeX), the themes table, the PRISMA counts
and the methods paragraph.
"""
from __future__ import annotations

import re
import urllib.parse
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from .questions import (LEVELS, QUESTIONS, S3_PAPER, _authors, _centroid_sim, _clip, _display_name,
                        _first_author, _int, _main_comms, _split, answer, available, included,
                        lifecycle, report_level)
from .report import _col, _e, meta_line, page, q_section

# What each answer means for the person writing the review.
NOTES = {
    "Q1": "Report the threshold and SII in your methods section; they show the map is not arbitrary.",
    "Q2": "Quote the silhouette and modularity when you describe how distinct the themes are.",
    "Q3": "Check these papers: drop off-topic ones from your corpus, or discuss them as a niche.",
    "Q4": "Use this to show readers that the themes agree with classic citation-based maps.",
    "Q5": "These communities are the themes of your review; the map shows their size and neighbours.",
    "Q6": "Use the terms and keywords to name each theme and to write its one-sentence description.",
    "Q7": "Start reading here: the most central papers define each theme.",
    "Q8": "Cite the foundations when you introduce each theme: local = what this field builds on.",
    "Q9": "Neighbouring themes can be discussed together; distant ones need a bridge in your text.",
    "Q10": "Shows which themes are mature (cite themselves) and which are knowledge sources for others.",
    "Q11": "Similar themes that ignore each other are gaps you can point out in your discussion.",
    "Q12": "Bridging papers connect themes; they are good examples when you link sections.",
    "Q13": "Tell readers which themes are emerging, growing or declining, and since when.",
    "Q14": "Rising concepts are candidates for your future-research section.",
    "Q15": "The leading authors per theme; useful for the who's who of the field.",
    "Q16": "Authors who work across themes carry ideas between them.",
    "Q17": "Similar themes with few shared authors are collaboration opportunities.",
    "Q18": "Describe where each theme is studied and how that changed.",
    "Q19": "Where each theme is published; helps you choose outlets and find missing literature.",
    "Q20": "Who funds which themes.",
    "Q21": "Which themes integrate others; useful when you discuss interdisciplinarity.",
    "Q22": "Whether integrative work is cited more.",
    "Q23": "Whether the field as a whole is becoming more interdisciplinary.",
    "Q24": "Supports the claim that content similarity anticipates citations.",
    "Q25": "Gaps likely to close soon: a forecast for your future-research section.",
    "Q26": "Concepts likely to enter a theme next.",
    "Q27": "Authors likely to move into a theme.",
    "Q28": "Likely next collaborations.",
    "Q29": "Institutions and funders likely to enter a theme.",
    "Q30": "Where new or outlying papers fit.",
    "Q31": "Which recent papers are likely to become highly cited.",
    "Q32": "Who fits a topic: experts and partners.",
    "Q33": "Which themes are likely to grow or decline next.",
    "Q34": "Relations likely missing from the knowledge graph.",
    "Q35": "Shows whether citation and co-authorship ties tell the same story as content; a robustness check.",
    "Q36": "Whether ties stay within journals, countries or funders; mention strong homophily as a bias.",
    "Q37": "Which themes are knowledge sources and which build on others.",
    "Q38": "Evidence that the themes are coherent: report it when reviewers ask why you used S3.",
}

SECTIONS = [
    ("scope", "1. Scope: what is in the corpus",
     "The records that went in, what was removed, and papers that do not belong to the field.", ["Q3"]),
    ("themes", "2. Themes",
     "One card per theme: what it is about, what to read first, how it grows and how it links to "
     "the other themes. Each card is the material for one subsection of your review.", ["Q5", "Q6", "Q7", "Q8"]),
    ("connect", "3. How themes connect",
     "Closeness in content, citation flows and the papers that bridge themes.", ["Q9", "Q10", "Q37", "Q12", "Q21"]),
    ("evolution", "4. Evolution", "How the themes and their concepts change over time.", ["Q13", "Q14", "Q23"]),
    ("actors", "5. Who does the work",
     "Authors, countries and institutions, journals and funders behind the themes.",
     ["Q15", "Q16", "Q18", "Q19", "Q20"]),
    ("agenda", "6. Research agenda",
     "Material for your discussion and future-research sections.",
     ["Q11", "Q17", "Q22", "Q25", "Q26", "Q27", "Q28", "Q29", "Q33"]),
    ("methods", "7. Methods appendix",
     "A methods paragraph to adapt, a glossary, and the analyses that show the map can be trusted.",
     ["Q1", "Q2", "Q38", "Q4", "Q35", "Q36", "Q24", "Q30", "Q31", "Q32", "Q34"]),
]

GLOSSARY = [
    (1, "S3", "Semantic Similarity Score: cosine similarity of two papers' title-and-abstract embeddings."),
    (1, "Theme (community)", "A group of papers more similar to each other than to the rest, found by the Leiden algorithm."),
    (1, "SII", "Sensitivity Inclusion Index = silhouette × retained papers; the threshold with the highest SII is used."),
    (1, "Silhouette", "How well separated the themes are (−1 to 1; higher is better)."),
    (1, "c-TF-IDF", "Terms that are frequent in one theme and rare in the others."),
    (1, "Eigenvector centrality", "A paper is central if it is similar to other central papers of its theme."),
    (1, "Local / global citations", "Citations from papers in this corpus / citations from all of Scopus."),
    (1, "QI", "Quality Index: citations within a theme per paper; higher = a more self-contained, mature theme."),
    (1, "MI", "Maturity Index: a theme's share of the field's in-corpus citations (within + inward), in %."),
    (1, "TCI", "Topic Coherence Index = 1 − Jensen-Shannon divergence between a paper's words and its theme's words."),
    (1, "Momentum", "A theme's share of papers in the last 3 years relative to the field's (above 1.2 = growing)."),
    (1, "Knowledge gap", "Two themes close in content that rarely cite each other."),
    (1, "Outlier", "A paper with no similarity link above the threshold."),
    (2, "Bibliographic coupling", "Two papers cite the same references."),
    (2, "Co-citation", "Two papers are cited together by a third."),
    (2, "Significant ties", "Coupling, co-citation or co-authorship ties that are more frequent than chance."),
    (2, "Assortativity", "Whether ties stay within groups such as countries (+1) or cross them (−1)."),
    (2, "PageRank", "Importance in the citation network: being cited by well-cited papers counts more."),
    (3, "ROC AUC", "How well a prediction ranks true future links above false ones (0.5 = chance, 1 = perfect)."),
]


# ------------------------------------------------------------------ helpers --
def _themes(res) -> List[int]:
    return _main_comms(res, res.config.top_n_communities)


def _tab(answers, qid, name) -> pd.DataFrame:
    a = answers.get(qid)
    if a is None or isinstance(a, Exception):
        return pd.DataFrame()
    return a.tables.get(name, pd.DataFrame())


def _paper(r) -> str:
    return f"{_clip(r['title'], 140)} ({_int(r['year']) or 'n.d.'})"


def reading_list(res, per_theme: int = 10) -> pd.DataFrame:
    """Per theme: its most central papers, then its most locally cited papers not yet listed."""
    P = res.papers
    rows = []
    for c in _themes(res):
        sub = P[(P["community"] == c).fillna(False)]
        core = sub.sort_values("eigen_centrality", ascending=False).head(per_theme)
        extra = sub.drop(core.index, errors="ignore")
        if "local_citations" in sub:
            extra = extra.sort_values("local_citations", ascending=False).head(3)
            extra = extra[extra["local_citations"] > 0]
        else:
            extra = extra.head(0)
        for why, part in (("central", core), ("foundation", extra)):
            for i, r in part.iterrows():
                rows.append({"theme": _display_name(res, c), "community": int(c), "why": why, "paper": i,
                             "title": r["title"], "authors": r.get("authors", ""), "year": _int(r["year"]),
                             "source": r.get("source", ""), "doi": r.get("doi", ""), "abstract": r.get("abstract", ""),
                             "cited_by": int(r["cited_by"])})
    return pd.DataFrame(rows)


def _s(v) -> str:
    return "" if v is None or (isinstance(v, float) and np.isnan(v)) or str(v).lower() == "nan" else str(v).strip()


def to_ris(rl: pd.DataFrame) -> str:
    out = []
    for r in rl.itertuples(index=False):
        out.append("TY  - JOUR")
        out.append(f"TI  - {_s(r.title)}")
        for a in _authors(r.authors):
            out.append(f"AU  - {a}")
        if r.year:
            out.append(f"PY  - {r.year}")
        if _s(r.source):
            out.append(f"JO  - {_s(r.source)}")
        if _s(r.doi):
            out.append(f"DO  - {_s(r.doi)}")
        if _s(r.abstract):
            out.append(f"AB  - {_s(r.abstract)}")
        out.append(f"KW  - S3 theme: {r.theme}")
        out.append(f"N1  - S3 theme: {r.theme} ({r.why}); cited by {r.cited_by}")
        out.append("ER  - ")
        out.append("")
    return "\n".join(out)


def to_bibtex(rl: pd.DataFrame) -> str:
    out, seen = [], {}
    esc = lambda x: _s(x).replace("{", "").replace("}", "")
    for r in rl.itertuples(index=False):
        first = re.sub(r"[^A-Za-z]", "", (_authors(r.authors) or ["anon"])[0].split(" ")[0]).lower() or "anon"
        word = next((w for w in re.findall(r"[A-Za-z]{4,}", _s(r.title)).__iter__()), "paper").lower()
        key = f"{first}{r.year or ''}{word}"
        seen[key] = seen.get(key, 0) + 1
        if seen[key] > 1:
            key += chr(ord("a") + seen[key] - 1)
        fields = {"title": esc(r.title), "author": " and ".join(esc(a) for a in _authors(r.authors)),
                  "year": r.year or "", "journal": esc(r.source), "doi": esc(r.doi),
                  "keywords": f"S3 theme: {esc(r.theme)}", "note": f"S3 theme: {esc(r.theme)} ({r.why})"}
        body = ",\n".join(f"  {k} = {{{v}}}" for k, v in fields.items() if v != "")
        out.append(f"@article{{{key},\n{body}\n}}\n")
    return "\n".join(out)


def prisma(res) -> pd.DataFrame:
    """Records in and out, PRISMA-style."""
    n = len(res.papers)
    rem = res.removed if isinstance(res.removed, pd.DataFrame) else pd.DataFrame()
    reasons = rem["reason"].value_counts() if len(rem) and "reason" in rem else pd.Series(dtype=int)
    rows = [("Records loaded", n + int(reasons.sum()))]
    for reason, k in reasons.items():
        rows.append((f"Removed: {reason}", -int(k)))
    rows.append(("Papers analysed", n))
    rows.append(("With references", int(res.papers["references"].notna().sum()) if "references" in res.papers else 0))
    rows.append(("In-corpus citation links found", len(res.citation_pairs)))
    clustered = int(res.papers["community"].notna().sum())
    rows.append(("In a theme (community)", clustered))
    themes = _themes(res)
    rows.append((f"In the {len(themes)} main themes", int(res.papers["community"].isin(themes).sum())))
    rows.append(("Outliers: no link above the threshold", int(res.papers["is_outlier"].sum())))
    return pd.DataFrame(rows, columns=["step", "papers"])


def themes_table(res, answers) -> pd.DataFrame:
    comms = _themes(res)
    life, _, _ = lifecycle(res, comms)
    life = life.set_index("community") if len(life) else life
    topics = _tab(answers, "Q6", "Topic profile per community")
    topics = topics.set_index("community") if len(topics) else topics
    metr = _tab(answers, "Q10", "Community citation metrics (Table 1)")
    metr = metr.set_index("community") if len(metr) else metr
    P = res.papers
    rows = []
    for c in comms:
        sub = P[(P["community"] == c).fillna(False)]
        yrs = sub["year"].dropna()
        top = sub.sort_values("eigen_centrality", ascending=False).head(1)
        rows.append({
            "theme": _display_name(res, c), "community": int(c), "papers": len(sub),
            "years": f"{int(yrs.min())}–{int(yrs.max())}" if len(yrs) else "",
            "status": life.at[c, "status"] if len(life) and c in life.index else "",
            "momentum": life.at[c, "relative momentum"] if len(life) and c in life.index else np.nan,
            "key terms": topics.at[c, "top terms (c-TF-IDF)"] if len(topics) and c in topics.index else "",
            "QI": round(float(metr.at[c, "qi"]), 2) if len(metr) and c in metr.index else np.nan,
            "MI (%)": round(float(metr.at[c, "mi"]), 1) if len(metr) and c in metr.index else np.nan,
            "most central paper": _paper(top.iloc[0]) if len(top) else ""})
    return pd.DataFrame(rows)


def methods_text(res, level: int) -> str:
    from . import __version__
    from .hybrid import map_label
    cfg = res.config
    n = len(res.papers)
    rem = res.removed if isinstance(res.removed, pd.DataFrame) else pd.DataFrame()
    reasons = rem["reason"].value_counts() if len(rem) and "reason" in rem else pd.Series(dtype=int)
    dup = int(sum(k for r, k in reasons.items() if "duplicate" in r))
    other = {r: int(k) for r, k in reasons.items() if "duplicate" not in r}
    yrs = res.papers["year"].dropna()
    span = f" published {int(yrs.min())}–{int(yrs.max())}" if len(yrs) else ""
    b = res.best or {}
    lo, hi = min(cfg.thresholds), max(cfg.thresholds)
    clustered = int(res.papers["community"].notna().sum())
    themes = _themes(res)
    model = cfg.model_name.split("/")[-1]
    s = [f"The corpus contained {n + int(reasons.sum())} records{span}."]
    rm = []
    if dup:
        rm.append(f"{dup} duplicates (same EID, DOI and title, or text, or near-identical title and abstract, S3 ≥ "
                  f"{cfg.dedupe_threshold:g})")
    rm += [f"{k} records ({r})" for r, k in other.items()]
    if rm:
        s.append(f"We removed {' and '.join(rm)}, leaving {n} papers.")
    s.append(f"Following the Semantic Similarity Score (S3) method (Bekamiri et al., 2027), we embedded each paper's "
             f"title and abstract with the {model} sentence-transformer model and computed the cosine similarity "
             f"between all pairs of papers" + (", keeping only links from newer to older papers (time masking)."
                                               if cfg.temporal_mask else "."))
    if cfg.map_weights:
        s.append(f"The map combined text similarity with citation relations ({map_label(cfg.map_weights)}), "
                 "an extension of the published method" +
                 ("; for coupling and co-citation only ties more frequent than chance were kept "
                  f"(degree-preserving null model, false discovery rate {cfg.backbone_alpha:g})." if cfg.backbone else "."))
    if b:
        how = ("fixed in advance" if b.get("fixed") else
               "the threshold with the highest Sensitivity Inclusion Index (SII = silhouette × retained papers)")
        s.append(f"We scanned similarity thresholds from {lo:g} to {hi:g} and used {b['threshold']:.2f}, {how}.")
        s.append(f"Communities were detected with the Leiden algorithm and directed modularity (Traag, Waltman & van "
                 f"Eck, 2019). At this threshold, {clustered} papers ({clustered / n:.0%}) form "
                 f"{len(res.communities())} communities (silhouette {b['silhouette']:.3f}); we describe the "
                 f"{len(themes)} largest as the themes of the field.")
    s.append("Each theme is described by its most distinctive terms (c-TF-IDF), its most central papers "
             "(eigenvector centrality) and its most cited papers inside the corpus (local) and overall (global)"
             + (", and named with the labels saved with the run." if res.labels else "."))
    if res.citation_pairs:
        how = ("retrieved from the Scopus API" if res.citation_source == "scopus_api" else
               "identified by matching titles in the reference lists" +
               (" (short titles also had to match the year and first author)" if cfg.citation_match == "strict" else ""))
        s.append(f"In-corpus citations ({len(res.citation_pairs)} links) were {how}; we used them to compute each "
                 "theme's Quality Index (within-theme citations per paper) and Maturity Index (share of the field's "
                 "in-corpus citations).")
    if level >= 2:
        s.append("Authors, affiliations, journals, funders and references were used to characterise the themes "
                 "(two-mode networks and a knowledge graph), as extensions of the published method.")
    if level >= 3:
        s.append("Forecasts used temporal splits (the last five years as test window) and were compared with simple "
                 "baselines.")
    s.append(f"The analysis was run with s3-bibliometric {__version__} (seed {cfg.seed}).")
    return " ".join(s) + "\n\nReferences\n" + S3_PAPER + \
        "\nTraag, V. A., Waltman, L., & van Eck, N. J. (2019). From Louvain to Leiden: guaranteeing well-connected " \
        "communities. Scientific Reports, 9, 5233. https://doi.org/10.1038/s41598-019-41695-z"


def _data_uri(text: str, mime: str = "text/plain") -> str:
    return f"data:{mime};charset=utf-8," + urllib.parse.quote(text)


# ------------------------------------------------------------------ blocks --
def _brief(res, answers, level: int, tt: pd.DataFrame) -> str:
    P = res.papers
    yrs = P["year"].dropna()
    n = len(P)
    clustered = int(P["community"].notna().sum())
    stats = [(f"{n:,}", "papers"), (f"{int(yrs.min())}–{int(yrs.max())}" if len(yrs) else "n/a", "years"),
             (str(len(tt)), "main themes"), (f"{clustered / n:.0%}" if n else "0%", "papers in a theme"),
             (f"{len(res.citation_pairs):,}", "citation links in the corpus")]
    cards = ['<div class="card"><h3>The field in numbers</h3><div class="stats">' +
             "".join(f'<div class="stat"><b>{_e(v)}</b><span>{_e(k)}</span></div>' for v, k in stats) + "</div></div>"]
    items = "".join(f'<li><a href="#theme-{r.community}">{_e(r.theme)}</a> <span class="mut">· {r.papers} papers'
                    + (f" · {_e(r.status)}" if r.status else "") + "</span></li>" for r in tt.itertuples())
    note = ("" if res.labels else '<p class="mut">Themes are named by their top terms. Give them names '
            '(res.labels, or save_community_labels in Claude) and rebuild the report.</p>')
    cards.append(f'<div class="card"><h3>Themes</h3><ol>{items}</ol>{note}</div>')
    first = "".join(f'<li>{_e(r["most central paper"])} <span class="mut">· {_e(r.theme)}</span></li>'
                    for _, r in tt.iterrows() if r["most central paper"])
    cards.append(f'<div class="card"><h3>Read first: one paper per theme</h3><ol>{first}</ol></div>')
    if "local_citations" in P and P["local_citations"].sum() > 0:
        f = P.sort_values(["local_citations", "cited_by"], ascending=False).head(5)
        fl = "".join(f'<li>{_e(_paper(r))} <span class="mut">· cited {int(r["local_citations"])}× in the corpus</span></li>'
                     for _, r in f.iterrows())
        cards.append(f'<div class="card"><h3>Foundations the field builds on</h3><ol>{fl}</ol></div>')
    moving = tt[tt["status"].isin(["emerging", "growing"])]
    falling = tt[tt["status"] == "declining"]
    mv = "".join(f'<li><b>{_e(r.status)}</b>: {_e(r.theme)}</li>' for r in moving.itertuples()) or \
        '<li class="mut">No main theme grows faster than the field.</li>'
    mv += "".join(f'<li><b>declining</b>: {_e(r.theme)}</li>' for r in falling.itertuples())
    cards.append(f'<div class="card"><h3>Where the field is going</h3><ul>{mv}</ul></div>')
    gaps = _tab(answers, "Q11", "Semantic similarity vs citations (RQ4)")
    if len(gaps) and "gap" in gaps:
        g = gaps[gaps["gap"]].head(3)
        gl = "".join(f'<li>{_e(r["community A"])} ↔ {_e(r["community B"])} <span class="mut">· similarity '
                     f'{r["semantic similarity"]}, {int(r["citations between"])} citations</span></li>'
                     for _, r in g.iterrows())
        if gl:
            cards.append(f'<div class="card"><h3>Gaps to discuss</h3><ul>{gl}</ul>'
                         f'<p class="mut">Close in content, rarely citing each other. More in the research agenda.</p></div>')
    if level >= 2:
        au = _tab(answers, "Q15", "Leading authors (RFI)").head(5)
        if len(au):
            al = "".join(f'<li>{_e(r["author"])} <span class="mut">· {int(r["papers"])} papers · '
                         f'{_e(r["main community"])}</span></li>' for _, r in au.iterrows())
            cards.append(f'<div class="card"><h3>Leading authors</h3><ol>{al}</ol></div>')
    return '<h2 id="brief">The field at a glance</h2><div class="brief">' + "".join(cards) + "</div>"


def _theme_cards(res, answers, level: int, tt: pd.DataFrame) -> str:
    P = res.papers
    comms = list(tt["community"])
    central = _tab(answers, "Q7", "Most central papers per community")
    found = _tab(answers, "Q8", "Foundational papers (Appendix Table C1)")
    kw = _tab(answers, "Q6", "Topic profile per community")
    kw = kw.set_index("community") if len(kw) else kw
    C = res.citation_matrix() if res.citation_pairs else pd.DataFrame()
    sim = _centroid_sim(res, comms) if comms else np.zeros((0, 0))
    geo = _tab(answers, "Q18", "Countries and institutions per community (RQ7/RQ8)")
    jr = _tab(answers, "Q19", "Top journals per community")
    fu = _tab(answers, "Q20", "Funding per community (RQ10)")
    life, _, _ = lifecycle(res, comms)
    life = life.set_index("community") if len(life) else life
    out = []
    for k, (_, r) in enumerate(tt.iterrows()):
        c = int(r['community'])
        sub = P[(P["community"] == c).fillna(False)]
        about = []
        if len(kw) and c in kw.index:
            about.append(f"<li><b>Terms:</b> {_e(kw.at[c, 'top terms (c-TF-IDF)'])}</li>")
            if kw.at[c, "top author keywords"]:
                about.append(f"<li><b>Author keywords:</b> {_e(kw.at[c, 'top author keywords'])}</li>")
        yrs = sub["year"].dropna()
        if len(yrs):
            about.append(f"<li><b>Years:</b> {int(yrs.min())}–{int(yrs.max())}, median {int(yrs.median())}</li>")
        if len(life) and c in life.index:
            L = life.loc[c]
            about.append(f"<li><b>Growth:</b> {_e(L['status'])} (momentum {L['relative momentum']}, first paper "
                         f"{L['first year']}, peak {L['peak year']})</li>")
        if pd.notna(r["QI"]):
            about.append(f"<li><b>Citations:</b> QI {r['QI']} · MI {r['MI (%)']}%</li>")
        read = central[central["community"] == c].sort_values("eigenvector", ascending=False).head(3) \
            if len(central) else pd.DataFrame()
        rl = "".join(f"<li>{_e(_paper(x))} <span class='mut'>· cited {int(x['cited_by'])}</span></li>"
                     for _, x in read.iterrows())
        fd = found[(found["community"] == c)] if len(found) else pd.DataFrame()
        if len(fd):
            loc = fd[fd["type"] == "local"].head(3)
            fd = loc if len(loc) else fd.head(3)
        fl = "".join(f"<li>{_e(_paper(x))} <span class='mut'>· local {x['local citations'] if pd.notna(x['local citations']) else '–'}"
                     f", global {int(x['global citations'])}</span></li>" for _, x in fd.iterrows())
        links = []
        if len(C) and c in C.index:
            row, col = C.loc[c].drop(c, errors="ignore"), C[c].drop(c, errors="ignore")
            on = row[row > 0].sort_values(ascending=False).head(2)
            by = col[col > 0].sort_values(ascending=False).head(2)
            if len(on):
                links.append("<li><b>Builds on:</b> " + "; ".join(f"{_e(_display_name(res, x))} ({int(v)})" for x, v in on.items()) + "</li>")
            if len(by):
                links.append("<li><b>Used by:</b> " + "; ".join(f"{_e(_display_name(res, x))} ({int(v)})" for x, v in by.items()) + "</li>")
        if len(comms) > 1:
            srow = sim[k].copy()
            srow[k] = -np.inf
            near = np.argsort(srow)[::-1][:2]
            links.append("<li><b>Closest in content:</b> " + "; ".join(
                f"{_e(_display_name(res, comms[j]))} ({srow[j]:.2f})" for j in near) + "</li>")
        blocks = [f"<div><h4>About</h4><ul>{''.join(about)}</ul></div>",
                  f"<div><h4>Read first</h4><ol>{rl}</ol></div>" if rl else "",
                  f"<div><h4>Foundations</h4><ol>{fl}</ol></div>" if fl else "",
                  f"<div><h4>Links to other themes</h4><ul>{''.join(links)}</ul></div>" if links else ""]
        if level >= 2:
            act = []
            cnt: Dict[str, int] = {}
            for v in sub.get("authors", pd.Series(dtype=object)):
                for a in set(_authors(v)):
                    cnt[a] = cnt.get(a, 0) + 1
            top_a = sorted(cnt.items(), key=lambda x: -x[1])[:3]
            if top_a and top_a[0][1] > 1:
                act.append("<li><b>Authors:</b> " + "; ".join(f"{_e(a)} ({n})" for a, n in top_a if n > 1) + "</li>")
            for t, col, lab in ((geo, "top countries", "Countries"), (jr, "top journals", "Journals"),
                                (fu, "top funders", "Funders")):
                if len(t) and col in t and (t["community"] == c).any():
                    v = t.loc[t["community"] == c, col].iloc[0]
                    if _s(v):
                        act.append(f"<li><b>{lab}:</b> {_e(_clip(v, 220))}</li>")
            if act:
                blocks.append(f"<div><h4>Who works on it</h4><ul>{''.join(act)}</ul></div>")
        out.append(f'<details class="theme" id="theme-{c}" style="--tc:{_col(c)}"><summary>{_e(r['theme'])} '
                   f'<span class="mut">· {r['papers']} papers'
                   + (f" · {_e(r['status'])}" if r["status"] else "") + f'</span></summary>'
                   f'<div class="theme-grid">{"".join(blocks)}</div></details>')
    return "".join(out)


def _agenda(res, answers, level: int, tt: pd.DataFrame) -> str:
    items = []
    gaps = _tab(answers, "Q11", "Semantic similarity vs citations (RQ4)")
    if len(gaps) and "gap" in gaps:
        for _, r in gaps[gaps["gap"]].head(5).iterrows():
            items.append(f"<li><b>Knowledge gap:</b> {_e(r['community A'])} and {_e(r['community B'])} are close in "
                         f"content (similarity {r['semantic similarity']}) but have only {int(r['citations between'])} "
                         "citations between them. A review could connect them.</li>")
    for r in tt[tt["status"].isin(["emerging", "growing"])].itertuples():
        items.append(f"<li><b>{_e(r.status.capitalize())} theme:</b> {_e(r.theme)} (momentum {r.momentum}).</li>")
    if level >= 2:
        col = _tab(answers, "Q17", "Collaboration gaps (RQ6)")
        if len(col) and "opportunity" in col:
            for _, r in col[col["opportunity"]].head(3).iterrows():
                items.append(f"<li><b>Collaboration gap:</b> {_e(r['community A'])} and {_e(r['community B'])} "
                             f"are similar (similarity {r['semantic similarity']}) but share only "
                             f"{int(r['shared authors'])} authors.</li>")
    a = answers.get("Q22")
    if a is not None and not isinstance(a, Exception) and a.findings:
        items.append(f"<li><b>Interdisciplinarity:</b> {_e(a.findings[0])}</li>")
    if level >= 3:
        for q in ("Q33", "Q25"):
            a = answers.get(q)
            if a is not None and not isinstance(a, Exception) and a.findings:
                items.append(f"<li><b>Forecast ({q}):</b> {_e(a.findings[0])}</li>")
    return f'<div class="card"><h3>Summary</h3><ul>{"".join(items)}</ul></div>' if items else ""


def _scope(res) -> str:
    t = prisma(res)
    rows = "".join(f"<tr><td>{_e(s)}</td><td class='n'>{p:,}</td></tr>" for s, p in t.itertuples(index=False))
    return f'<div class="card"><h3>Records in and out</h3><div class="tw"><table><tr><th>step</th><th>papers</th></tr>{rows}</table></div></div>'


def _methods(res, level: int) -> str:
    gl = "".join(f"<dt>{_e(k)}</dt><dd>{_e(v)}</dd>" for lv, k, v in GLOSSARY if lv <= level)
    return (f'<h3>Methods paragraph</h3><p class="intro">Adapt this text for your methods section.</p>'
            f'<div class="methods">{_e(methods_text(res, level))}</div>'
            f'<h3 style="margin-top:16px">Glossary</h3><dl class="gloss">{gl}</dl>')


# ------------------------------------------------------------------ public --
def compute_answers(res, level: int) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for q in QUESTIONS:
        if not included(q, level) or not available(res, q)[0]:
            continue
        try:
            out[q.qid] = answer(res, q.qid, prediction=True if q.prediction else None)
        except Exception as ex:  # keep the report going
            out[q.qid] = ex
    return out


def build_review(res, path: Union[str, Path], title: Optional[str] = None, level: Optional[int] = None,
                 prediction: Optional[bool] = None) -> Path:
    lv = report_level(res, level, prediction)
    path = Path(path).expanduser()
    title = title or "S3 literature review report"
    answers = compute_answers(res, lv)
    tt = themes_table(res, answers)
    rl = reading_list(res)
    by_id = {q.qid: q for q in QUESTIONS}
    dl = [("reading_list.ris", to_ris(rl), "application/x-research-info-systems",
           "Reading list (RIS: Zotero, EndNote, Mendeley)"),
          ("reading_list.bib", to_bibtex(rl), "application/x-bibtex", "Reading list (BibTeX)"),
          ("themes.csv", tt.to_csv(index=False), "text/csv", "Themes table (CSV)"),
          ("prisma.csv", prisma(res).to_csv(index=False), "text/csv", "Records in and out (CSV)"),
          ("methods.txt", methods_text(res, lv), "text/plain", "Methods paragraph")]
    downloads = '<div class="dl">' + "".join(
        f'<a download="{f}" href="{_data_uri(t, m)}">{_e(lab)}</a>' for f, t, m, lab in dl) + "</div>"
    toc = [('<a href="#brief">The field at a glance</a>')]
    body = [_brief(res, answers, lv, tt)]
    for sid, head, intro, qids in SECTIONS:
        qs = [by_id[x] for x in qids if x in by_id and included(by_id[x], lv)]
        if not qs and sid != "themes":
            continue
        toc.append(f'<a href="#{sid}">{_e(head)}</a>')
        extra = ""
        if sid == "scope":
            extra = _scope(res)
        elif sid == "themes":
            extra = _theme_cards(res, answers, lv, tt) + '<h3 style="margin-top:18px">Details behind the theme cards</h3>'
        elif sid == "agenda":
            extra = _agenda(res, answers, lv, tt)
        elif sid == "methods":
            extra = _methods(res, lv) + '<h3 style="margin-top:18px">Can the map be trusted?</h3>'
        secs = "".join(q_section(res, q, lv >= 3, a=answers.get(q.qid), note=NOTES.get(q.qid)) for q in qs)
        body.append(f'<h2 id="{sid}">{_e(head)}</h2><p class="intro">{_e(intro)}</p>{extra}{secs}')
    header = meta_line(res, lv) + f'<nav class="toc">{"".join(toc)}</nav>' + \
        '<p class="mut" style="margin:6px 0 0">Downloads for writing the review:</p>' + downloads
    doc = page(title, header, "".join(body))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(doc, encoding="utf-8")
    return path


def export_review(res, directory: Union[str, Path], level: Optional[int] = None) -> Dict[str, Path]:
    lv = report_level(res, level)
    d = Path(directory).expanduser()
    d.mkdir(parents=True, exist_ok=True)
    answers = {q: answer(res, q) for q in ("Q6", "Q10") if available(res, by_q(q))[0]}
    rl = reading_list(res)
    files = {"reading_list.ris": to_ris(rl), "reading_list.bib": to_bibtex(rl),
             "themes.csv": themes_table(res, answers).to_csv(index=False),
             "prisma.csv": prisma(res).to_csv(index=False), "methods.txt": methods_text(res, lv)}
    out = {}
    for name, text in files.items():
        (d / name).write_text(text, encoding="utf-8")
        out[name] = d / name
    return out


def by_q(qid: str):
    return next(q for q in QUESTIONS if q.qid == qid)
