"""
The S3 question catalogue: every analysis in the S3 method, ranked by the
research question it answers, in the order a researcher should go through them.

    res.questions()        -> table of questions and whether your data supports them
    res.answer("Q7")       -> Answer(findings, tables, chart)
    res.report("r.html")   -> all answers in one HTML report

Stages
  1. Trust the map       (is the S3 structure valid?)
  2. Map the field       (what is it made of?)
  3. Knowledge flows     (how do communities use each other?)
  4. Evolution           (how does the field change?)
  5. Actors              (who does the work?)
  6. Interdisciplinarity (where do fields meet?)
"""
from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd

STAGES = {
    1: "Trust the map",
    2: "Map the field",
    3: "Knowledge flows",
    4: "Evolution",
    5: "Actors",
    6: "Interdisciplinarity",
    7: "Link prediction",
    8: "Graph learning (GNN)",
}
PREDICTION_STAGES = (7, 8)  # off by default: S3Config(prediction=True) or report(prediction=True)


@dataclass
class Answer:
    qid: str
    question: str
    findings: List[str] = field(default_factory=list)
    tables: Dict[str, pd.DataFrame] = field(default_factory=dict)
    chart: Optional[Dict[str, Any]] = None  # {"type": "bar"|"line"|"heatmap"|"stack"|"sankey"|"network", ...}
    charts: List[Dict[str, Any]] = field(default_factory=list)  # further charts for the same answer

    def __repr__(self) -> str:
        lines = [f"{self.qid}  {self.question}"] + [f"  - {f}" for f in self.findings]
        for name, t in self.tables.items():
            lines.append(f"\n[{name}]\n{t.head(10).to_string(index=False)}")
        return "\n".join(lines)

    def to_dict(self, max_rows: int = 25) -> Dict[str, Any]:
        import json
        return {
            "qid": self.qid, "question": self.question, "findings": self.findings,
            "tables": {k: json.loads(v.head(max_rows).to_json(orient="records"))
                       for k, v in self.tables.items()},
        }


@dataclass
class Question:
    qid: str
    stage: int
    priority: str          # "Essential" | "Recommended" | "Optional"
    text: str
    why: str
    method: str
    needs: Tuple[str, ...]  # data columns needed beyond Title + Abstract
    fn: Callable[[Any], Answer]

    @property
    def prediction(self) -> bool:
        return self.stage in PREDICTION_STAGES


# ===========================================================================
# helpers
# ===========================================================================
def _name(res, c) -> str:
    c = int(c)
    lab = res.labels.get(c, "")
    return f"C{c} {lab}".strip()


def _map_score(res) -> str:
    """What the map threshold applies to: 'S3', or the hybrid score with its mix."""
    from .hybrid import map_label
    w = res.config.map_weights
    return "S3" if not w else f"the map score ({map_label(w)})"


def _display_name(res, c, n_terms: int = 3) -> str:
    """Name shown on graphs: the saved label, or else the community's top c-TF-IDF terms."""
    c = int(c)
    lab = res.labels.get(c, "")
    if not lab:
        try:
            lab = " · ".join(t for t, _ in res.top_terms(c, n_terms))
        except Exception:  # pragma: no cover - no text for this community
            lab = ""
    return f"C{c} {lab}".strip()


def _clustered(res) -> pd.DataFrame:
    return res.papers.dropna(subset=["community"])


def _main_comms(res, n: Optional[int] = None) -> List[int]:
    ids = res.communities()["community"].tolist()
    return ids if n is None else ids[:n]


def _has(res, col: str) -> bool:
    if col == "citations":
        return bool(res.citation_pairs)
    if col == "years":  # a time span, not just a year column
        return res.papers["year"].nunique() >= 3
    if col == "impact":  # enough cited papers to compare impact
        return (res.papers["cited_by"] > 0).mean() >= 0.2
    s = res.papers.get(col)
    return s is not None and s.notna().any() and (s.astype(str).str.strip() != "").any()


def _split(v, sep=";") -> List[str]:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return []
    return [p.strip() for p in str(v).split(sep) if p.strip() and p.strip().lower() != "nan"]


_REF = re.compile(r"[^;].*?\(\d{4}[a-z]?\)")


def _split_refs(v) -> List[str]:
    """Split a Scopus References cell into single references. Each reference ends with
    '(YYYY)'. Splitting on ';' alone breaks newer exports, which also use ';' between
    the authors inside one reference."""
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return []
    return [m.group(0).strip(" ;") for m in _REF.finditer(str(v))]


def _authors(v) -> List[str]:
    return [a for a in _split(v) if len(a) > 1]


def _centroid_sim(res, comms: List[int]) -> np.ndarray:
    """Mean pairwise S3 between communities = dot product of mean embeddings."""
    P = res.papers
    E = res.embeddings
    means = np.vstack([E[(P["community"] == c).fillna(False).to_numpy(dtype=bool)].mean(axis=0)
                       for c in comms])
    return means @ means.T


def _shannon(counts) -> float:
    v = np.asarray(list(counts), dtype=float)
    v = v[v > 0]
    if v.sum() == 0:
        return 0.0
    p = v / v.sum()
    return float(-(p * np.log(p)).sum())


_INST = re.compile(r"universit|college|institut|school|cent(er|re)|hospital|academy|"
                   r"polytechn|laborator|business school|ministry|foundation|bank|council|"
                   r"agency|corporation|company|inc\b|ltd|gmbh", re.I)


def _parse_affiliation(a: str) -> Tuple[str, str]:
    """('Aalborg University', 'Denmark') from 'Dept X, Aalborg University, Aalborg, Denmark'."""
    parts = [p.strip() for p in a.split(",") if p.strip()]
    if not parts:
        return "", ""
    country = re.sub(r"[^A-Za-z .&-]", "", parts[-1]).strip() if len(parts) > 1 else ""
    inst = next((p for p in parts[:-1] if _INST.search(p)), parts[0])
    return inst, country


def _paper_affils(res) -> pd.DataFrame:
    """One row per (paper, institution, country)."""
    rows = []
    for i, v in res.papers["affiliations"].items():
        for a in _split(v):
            inst, country = _parse_affiliation(a)
            rows.append((i, inst, country))
    return pd.DataFrame(rows, columns=["paper", "institution", "country"]).drop_duplicates()


def _semantic_integration(res) -> pd.Series:
    """Rao-Stirling diversity of each paper's S3 links over communities,
    with disparity = 1 - mean S3 between communities. 0 = links stay in one community."""
    if getattr(res, "_rs", None) is not None:
        return res._rs
    G = res.graph()
    comm = res.papers["community"]
    comms = _main_comms(res)
    idx = {c: k for k, c in enumerate(comms)}
    D = 1.0 - _centroid_sim(res, comms)
    np.fill_diagonal(D, 0.0)
    out = np.full(len(res.papers), np.nan)
    for i in range(len(res.papers)):
        w = defaultdict(float)
        for j, d in G[i].items():
            cj = comm.iat[j]
            if pd.notna(cj) and int(cj) in idx:
                w[idx[int(cj)]] += d.get("weight", 1.0)
        tot = sum(w.values())
        if tot == 0:
            continue
        ks = list(w)
        p = np.array([w[k] / tot for k in ks])
        out[i] = float(p @ D[np.ix_(ks, ks)] @ p)
    res._rs = pd.Series(out, index=res.papers.index, name="integration")
    return res._rs


def _network_spec(res, n_comms: int = 20, max_edges: int = 20000) -> Optional[Dict[str, Any]]:
    """Paper-level S3 network map (paper Panel 1): communities are laid out first, then the
    papers of each community around its centre. Deterministic (seeded)."""
    G = res.graph()
    P = res.papers
    comms = _main_comms(res, n_comms)
    if not comms:
        return None
    members = {c: list(P.index[(P["community"] == c).fillna(False)]) for c in comms}
    of = {i: c for c, ns in members.items() for i in ns}
    CG = nx.Graph()
    CG.add_nodes_from(comms)
    for u, v in G.edges():
        cu, cv = of.get(u), of.get(v)
        if cu is not None and cv is not None and cu != cv:
            CG.add_edge(cu, cv, weight=CG.get_edge_data(cu, cv, {}).get("weight", 0) + 1)
    seed = res.config.seed
    cpos = nx.spring_layout(CG, seed=seed, weight="weight", k=2.2 / np.sqrt(len(comms)), iterations=300)
    big = max(len(v) for v in members.values())
    radius = {c: 0.06 + 0.32 * np.sqrt(len(members[c]) / big) for c in comms}
    # push overlapping community circles apart; linked communities stay neighbours
    centre = {c: np.asarray(cpos[c], dtype=float) * 2.6 for c in comms}
    for _ in range(400):
        moved = False
        for a_i, a in enumerate(comms):
            for b in comms[a_i + 1:]:
                d = centre[b] - centre[a]
                dist = float(np.hypot(*d)) or 1e-6
                need = (radius[a] + radius[b]) * 1.15
                if dist < need:
                    push = d / dist * (need - dist) / 2
                    centre[a] -= push
                    centre[b] += push
                    moved = True
        if not moved:
            break
    pos = {}
    for c, ns in members.items():
        sub = G.subgraph(ns)
        lay = nx.spring_layout(sub, seed=seed, iterations=80) if len(ns) > 1 else {ns[0]: np.zeros(2)}
        arr = np.array([lay[i] for i in ns])
        arr = arr - arr.mean(axis=0)
        arr = arr / (np.abs(arr).max() or 1) * radius[c]
        for i, xy in zip(ns, arr):
            pos[i] = centre[c] + xy
    names = {c: _display_name(res, c) for c in comms}
    order = [i for c in comms for i in members[c]]
    idx = {i: k for k, i in enumerate(order)}
    cites = P["cited_by"].to_numpy()
    nodes = [[round(float(pos[i][0]), 4), round(float(pos[i][1]), 4),
              int(of[i]) if of[i] < res.config.top_n_communities else -1,
              round(float(2 + 1.4 * np.log1p(cites[i]) ** 0.9), 2),
              str(P.at[i, "title"])[:140], _int(P.at[i, "year"]), int(cites[i]), names[of[i]],
              _first_author(P.at[i, "authors"]), _clip(P.at[i, "abstract"], 900)]
             for i in order]
    edges = [[idx[u], idx[v]] for u, v in G.edges() if u in idx and v in idx][:max_edges]
    legend = [{"id": int(c), "name": _display_name(res, c), "n": len(members[c])} for c in comms
              if c < res.config.top_n_communities]
    from .hybrid import map_label
    kind = "S3 network map (paper Panel 1)" if not res.config.map_weights else \
        f"Network map ({map_label(res.config.map_weights)})"
    return {"type": "network", "title": f"{kind}: {len(order)} papers in the "
                                        f"{len(comms)} largest communities; top {len(legend)} in colour",
            "nodes": nodes, "edges": edges, "legend": legend}


def _int(x):
    return None if pd.isna(x) else int(x)


def _clip(v, n: int) -> str:
    v = "" if v is None or (isinstance(v, float) and np.isnan(v)) else str(v)
    return v if len(v) <= n else v[:n].rsplit(" ", 1)[0] + " …"


def _first_author(v) -> str:
    """'Lokshin B.; Belderbos R.' -> 'Lokshin B. et al.'"""
    names = _split(v) if isinstance(v, str) else []
    if len(names) == 1 and "," in names[0] and names[0].count(",") > 1:  # older Scopus: 'A., B., C.'
        names = [x.strip() for x in names[0].split(",") if x.strip()]
    if not names:
        return ""
    return names[0] + (" et al." if len(names) > 1 else "")


# ===========================================================================
# Stage 1 - Trust the map
# ===========================================================================
def q_threshold(res) -> Answer:
    scan = pd.DataFrame(res.scan)
    a = Answer("Q1", "")
    if scan.empty or not res.best:
        a.findings.append("No threshold produced a usable community structure.")
        return a
    b = res.best
    sii_best = scan.loc[scan["sii"].idxmax()]
    ms = _map_score(res)
    head = (f"The threshold is fixed at {ms} ≥ {b['threshold']:.2f} (SII = {b['sii']:.1f}); the highest SII in "
            f"the scan is at {sii_best['threshold']:.2f} (SII = {sii_best['sii']:.1f})." if b.get("fixed") else
            f"The best threshold is {ms} ≥ {b['threshold']:.2f} (highest SII = {b['sii']:.1f}).")
    a.findings += [head]
    if res.config.map_weights:
        a.findings.append("The map combines text similarity with citation relations; its thresholds link as many "
                          "pairs as the S3 thresholds 0.75-0.85 would on this corpus, so the scan is comparable.")
    a.findings += [
        f"At this threshold {b['n_papers']} of {len(res.papers)} papers "
        f"({b['n_papers'] / len(res.papers):.0%}) fall into {b['n_communities']} communities, "
        f"with a silhouette of {b['silhouette']:.3f}.",
    ]
    lo, hi = scan["threshold"].min(), scan["threshold"].max()
    if not b.get("fixed") and b["threshold"] in (lo, hi):
        edge = "lowest" if b["threshold"] == lo else "highest"
        a.findings.append(f"Warning: the best threshold is the {edge} value scanned, so SII may peak "
                          f"outside the range. Re-run with a wider range, e.g. "
                          f"S3Config(thresholds=...), as a sensitivity check.")
    share = b["n_papers"] / len(res.papers)
    if share < 0.5:
        a.findings.append(f"Only {share:.0%} of papers cluster: the corpus is probably broad or mixes several "
                          "fields. Consider narrowing the search, or read the communities as the "
                          "field's tightest topic clusters.")
    a.tables["Threshold scan"] = scan.round(4)
    a.chart = {"type": "line", "title": "Sensitivity Inclusion Index across thresholds (paper Fig. 3)",
               "x": scan["threshold"].tolist(), "series": {"SII": scan["sii"].round(2).tolist()},
               "x_label": "S3 threshold" if not res.config.map_weights else "map threshold", "mark": b["threshold"],
               "mark_label": f"{'fixed' if b.get('fixed') else 'best'} {b['threshold']:.2f}"}
    return a


def q_quality(res) -> Answer:
    a = Answer("Q2", "")
    G = res.graph()
    P = res.papers
    groups = [set(P.index[P["community"] == c]) for c in _main_comms(res)]
    singletons = [{i} for i in P.index[P["community"].isna()]]
    try:
        mod = nx.algorithms.community.modularity(G, groups + singletons, weight="weight")
    except Exception:
        mod = float("nan")
    dens = nx.density(G)
    comps = nx.number_connected_components(G.subgraph([n for n in G if G.degree(n) > 0]))
    a.findings += [
        f"Modularity = {mod:.3f} ({'strong' if mod >= 0.4 else 'moderate' if mod >= 0.3 else 'weak'} "
        f"community structure; above 0.3 is usually considered meaningful).",
        f"Silhouette = {res.best['silhouette']:.3f} "
        f"({'well separated' if res.best['silhouette'] >= 0.5 else 'reasonably separated' if res.best['silhouette'] >= 0.25 else 'weakly separated'}); "
        f"graph density = {dens:.4f}; "
        f"{comps} connected component(s) among linked papers.",
    ]
    a.tables["Structure quality"] = pd.DataFrame([{
        "threshold": res.best["threshold"], "silhouette": round(res.best["silhouette"], 4),
        "SII": round(res.best["sii"], 2), "modularity": round(mod, 4), "density": round(dens, 5),
        "papers": len(P), "clustered": int(P["community"].notna().sum()),
        "outliers": int(P["is_outlier"].sum()), "communities": len(groups), "components": comps}])
    return a


def q_outliers(res) -> Answer:
    a = Answer("Q3", "")
    o = res.outliers()
    n = len(res.papers)
    unclustered = int(res.papers["community"].isna().sum())
    a.findings.append(f"{len(o)} paper(s) ({len(o) / n:.0%}) have no S3 link above the threshold; "
                      f"{unclustered} paper(s) in total are outside any community.")
    if len(o):
        top = o.iloc[0]
        a.findings.append(f"The most cited outlier is “{top['title']}” ({int(top['cited_by'])} citations). "
                          "Check whether outliers are off-topic (remove from the corpus) or a niche "
                          "worth keeping.")
    t = o[["title", "year", "cited_by", "nearest_similarity", "suggested_community"]].head(25)
    a.tables["Outliers (most cited first)"] = t
    return a


def q_validation(res) -> Answer:
    """S3 vs bibliographic coupling (BC) and co-citation (CC)."""
    import scipy.sparse as sp
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    a = Answer("Q4", "")
    P = res.papers
    n = len(P)
    rng = np.random.default_rng(res.config.seed)
    sample = np.sort(rng.choice(n, size=min(n, 3000), replace=False))

    # bibliographic coupling: Jaccard of normalised reference strings
    vocab: Dict[str, int] = {}
    rows, cols = [], []
    for k, i in enumerate(sample):
        for r in _split_refs(P["references"].iat[i]):
            key = re.sub(r"[^a-z0-9]+", " ", r.lower()).strip()[:150]
            if key:
                rows.append(k)
                cols.append(vocab.setdefault(key, len(vocab)))
    m = len(sample)
    R = sp.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(m, max(1, len(vocab))))
    R.data[:] = 1.0
    inter = (R @ R.T).toarray()
    sizes = np.asarray(R.sum(axis=1)).ravel()
    union = sizes[:, None] + sizes[None, :] - inter
    BC = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)

    S = (res.embeddings[sample] @ res.embeddings[sample].T)
    iu = np.triu_indices(m, 1)
    methods = {"S3": S[iu], "BC": BC[iu]}
    if res.citation_pairs:
        pos = {int(i): k for k, i in enumerate(sample)}
        cr, cc = zip(*[(pos[i], pos[j]) for i, j in res.citation_pairs if i in pos and j in pos]) \
            if any(i in pos and j in pos for i, j in res.citation_pairs) else ((), ())
        A = sp.csr_matrix((np.ones(len(cr)), (cr, cc)), shape=(m, m))
        CC = (A.T @ A).toarray().astype(float)
        np.fill_diagonal(CC, 0)
        methods["CC"] = CC[iu]
    names = list(methods)
    corr = pd.DataFrame(np.corrcoef(np.vstack([methods[k] for k in names])), index=names, columns=names)

    # top-10 neighbour overlap S3 vs BC, for papers with references
    has_refs = sizes > 0
    ov = []
    for k in np.where(has_refs)[0]:
        s_row, b_row = S[k].copy(), BC[k].copy()
        s_row[k] = b_row[k] = -1
        if (b_row > 0).sum() < 3:
            continue
        top_s = set(np.argsort(s_row)[-10:])
        top_b = set(np.argsort(b_row)[-10:])
        ov.append(len(top_s & top_b) / len(top_s | top_b))

    # community agreement: Louvain on BC vs S3 communities
    agree = {}
    G_bc = nx.Graph()
    G_bc.add_nodes_from(range(m))
    bi, bj = np.where(np.triu(BC, 1) > 0)
    G_bc.add_weighted_edges_from(zip(bi.tolist(), bj.tolist(), BC[bi, bj].tolist()))
    if G_bc.number_of_edges():
        from community import community_louvain
        part = community_louvain.best_partition(G_bc, weight="weight", random_state=res.config.seed)
        lab_s3 = P["community"].to_numpy()[sample]
        mask = np.array([pd.notna(x) for x in lab_s3]) & has_refs
        if mask.sum() > 5:
            l1 = lab_s3[mask].astype(int)
            l2 = np.array([part[k] for k in np.where(mask)[0]])
            agree = {"ARI": adjusted_rand_score(l1, l2), "NMI": normalized_mutual_info_score(l1, l2)}

    if not has_refs.any():
        a.findings.append("No reference lists in the export, so S3 cannot be compared with "
                          "bibliographic coupling.")
        return a
    cc_ok = "CC" in corr and not np.isnan(corr.loc["S3", "CC"])
    a.findings.append(f"S3 correlates r = {corr.loc['S3', 'BC']:.2f} with bibliographic coupling"
                      + (f" and r = {corr.loc['S3', 'CC']:.2f} with co-citation." if cc_ok else
                         " (too few in-corpus citations to compare with co-citation)."))
    if ov:
        a.findings.append(f"On average {np.mean(ov):.0%} of a paper's 10 nearest neighbours are shared "
                          "between S3 and bibliographic coupling (Jaccard).")
    if agree:
        a.findings.append(f"Communities from bibliographic coupling agree with S3 communities at "
                          f"NMI = {agree['NMI']:.2f}, ARI = {agree['ARI']:.2f}.")
    a.findings.append(f"{int((~has_refs).sum())} of {m} papers have no references and can only be "
                      "mapped by S3 (an advantage of the semantic approach).")
    a.tables["Correlation between methods (Pearson, paper pairs)"] = corr.round(3).reset_index().rename(columns={"index": "method"})
    a.tables["Agreement summary"] = pd.DataFrame([{
        "papers compared": m, "papers with references": int(has_refs.sum()),
        "top-10 neighbour overlap (S3 vs BC)": round(float(np.mean(ov)), 3) if ov else None,
        "NMI (S3 vs BC communities)": round(agree.get("NMI", float("nan")), 3) if agree else None,
        "ARI (S3 vs BC communities)": round(agree.get("ARI", float("nan")), 3) if agree else None}])
    a.chart = {"type": "heatmap", "title": "Correlation between similarity methods",
               "rows": names, "cols": names, "matrix": corr.round(2).values.tolist()}
    return a


# ===========================================================================
# Stage 2 - Map the field
# ===========================================================================
def q_communities(res) -> Answer:
    a = Answer("Q5", "")
    c = res.communities()
    clustered = int(c["n_papers"].sum())
    big = c.head(3)
    a.findings.append(f"The field splits into {len(c)} communities covering {clustered} papers.")
    a.findings.append("The largest are " + ", ".join(
        f"{_name(res, r.community)} ({r.n_papers} papers, {r.n_papers / clustered:.0%})"
        for r in big.itertuples()) + ".")
    if not res.labels:
        a.findings.append("Communities have no labels yet: ask your AI assistant to label them "
                          "(MCP) or set res.labels = {0: '...', ...}.")
    cols = ["community", "label", "n_papers", "total_citations", "mean_citations",
            "year_min", "year_median", "year_max"]
    a.tables["Communities"] = c[cols].round(1)
    top_n = res.config.top_n_communities
    a.chart = {"type": "bar", "title": f"Community sizes in descending order (elbow plot, paper Fig. 4); "
                                       f"the top {top_n} are highlighted",
               "labels": [_name(res, x) for x in c["community"]], "values": c["n_papers"].tolist(),
               "highlight": top_n, "max": 30, "color_ids": c["community"].tolist()}
    net = _network_spec(res)
    if net:
        a.charts.append(net)
    return a


def q_topics(res) -> Answer:
    a = Answer("Q6", "")
    rows = []
    for c in _main_comms(res):
        rows.append({"community": c, "label": res.labels.get(c, ""),
                     "top terms (c-TF-IDF)": ", ".join(t for t, _ in res.top_terms(c, 8)),
                     "top author keywords": ", ".join(k for k, _ in res.top_keywords(c, 6))})
    t = pd.DataFrame(rows)
    a.findings.append("Each community is described by its most distinctive title/abstract terms "
                      "and its most frequent author keywords. Use these with Q7 to name communities.")
    a.tables["Topic profile per community"] = t
    sets = {}
    kw = Counter(k.strip().lower() for v in res.papers["keywords"].dropna() for k in str(v).split(";") if k.strip())
    if kw:
        sets["Whole field: author keywords"] = {"words": [[k, int(n)] for k, n in kw.most_common(50)], "color": -1}
    for c in _main_comms(res, 10):
        terms = res.top_terms(c, 50)
        if terms:
            sets[_display_name(res, c) + ": distinctive terms"] = {"words": [[w, round(float(v), 3)] for w, v in terms],
                                                                   "color": int(c)}
    if sets:
        a.charts.append({"type": "wordcloud", "title": "Word cloud: choose a community and how many words "
                                                       "(size = c-TF-IDF weight, or keyword frequency for the field)",
                         "sets": sets, "n": 10})
    return a


def q_central(res) -> Answer:
    a = Answer("Q7", "")
    G = res.graph()
    P = res.papers
    deg = nx.degree_centrality(G)
    k = min(len(G), 300) if len(G) > 500 else None
    bet = nx.betweenness_centrality(G, k=k, weight=None, seed=res.config.seed)
    rows = []
    for c in _main_comms(res):
        sub = res.community_papers(c, n=5)
        for i, r in sub.iterrows():
            rows.append({"community": c, "title": r["title"], "year": r["year"], "cited_by": r["cited_by"],
                         "eigenvector": round(r["eigen_centrality"], 4),
                         "degree": round(deg.get(i, 0), 4), "betweenness": round(bet.get(i, 0), 4)})
    t = pd.DataFrame(rows)
    a.findings.append("Central papers are the semantic core of each community (highest eigenvector "
                      "centrality in the S3 graph). They are the best papers to read first.")
    if len(t):
        topb = max(bet, key=bet.get)
        a.findings.append(f"The paper with the highest betweenness (most 'between' topics) is "
                          f"“{P.at[topb, 'title']}”.")
    a.tables["Most central papers per community"] = t
    return a


def q_foundational(res) -> Answer:
    """Table 3: top local vs top global papers per community."""
    a = Answer("Q8", "")
    P = res.papers
    has_local = "local_citations" in P and res.citation_pairs
    rows = []
    for c in _main_comms(res, 10):
        sub = P[P["community"] == c]
        glob = sub.sort_values("cited_by", ascending=False).head(3)
        for _, r in glob.iterrows():
            rows.append({"community": c, "type": "global", "title": r["title"], "year": r["year"],
                         "global citations": int(r["cited_by"]),
                         "local citations": int(r["local_citations"]) if has_local else None})
        if has_local:
            loc = sub.sort_values(["local_citations", "cited_by"], ascending=False).head(3)
            for _, r in loc.iterrows():
                rows.append({"community": c, "type": "local", "title": r["title"], "year": r["year"],
                             "global citations": int(r["cited_by"]),
                             "local citations": int(r["local_citations"])})
    t = pd.DataFrame(rows)
    a.findings.append("Global = most cited overall (Scopus 'Cited by'); local = most cited by other "
                      "papers in this corpus. Local foundations show what the field "
                      "itself builds on.")
    if not has_local:
        a.findings.append("Local citations need References and EID in the export; only global "
                          "papers are shown.")
    cited = (P["cited_by"] > 0).mean()
    if cited < 0.2:
        a.findings.append(f"Only {cited:.0%} of papers have been cited so far, so citation-based "
                          "rankings here are still unstable.")
    a.tables["Foundational papers (Appendix Table C1)"] = t
    return a


def q_similarity(res) -> Answer:
    a = Answer("Q9", "")
    comms = _main_comms(res, 12)
    M = _centroid_sim(res, comms)
    names = [_name(res, c) for c in comms]
    off = [(M[i, j], i, j) for i in range(len(comms)) for j in range(i + 1, len(comms))]
    off.sort(reverse=True)
    if off:
        s, i, j = off[0]
        a.findings.append(f"The closest communities are {names[i]} and {names[j]} "
                          f"(mean S3 = {s:.3f}).")
        s, i, j = off[-1]
        a.findings.append(f"The most distant are {names[i]} and {names[j]} (mean S3 = {s:.3f}).")
    a.tables["Mean S3 between communities"] = pd.DataFrame(M.round(3), index=names, columns=names) \
        .reset_index().rename(columns={"index": "community"})
    a.chart = {"type": "heatmap", "title": "Mean S3 similarity between communities",
               "rows": names, "cols": names, "matrix": M.round(3).tolist()}
    return a


# ===========================================================================
# Stage 3 - Knowledge flows
# ===========================================================================
def q_flows(res) -> Answer:
    a = Answer("Q10", "")
    m = res.citation_metrics()
    C = res.citation_matrix()
    names = [_name(res, c) for c in C.index]
    src = ("from the Scopus API (exact)" if res.citation_source == "scopus_api" else
           "by matching titles in the References text (approximate; add a Scopus API key for exact links)")
    a.findings.append(f"{len(res.citation_pairs)} citation links were found inside the corpus, {src}.")
    if not m.empty:
        mat = m.sort_values("qi", ascending=False).iloc[0]
        a.findings.append(f"The most self-referential (mature) community is {_name(res, mat.community)} "
                          f"(QI = {mat.qi:.2f} within-community citations per paper).")
        imp = m.sort_values("inward_citations", ascending=False).iloc[0]
        a.findings.append(f"The community most cited by others is {_name(res, imp.community)} "
                          f"({int(imp.inward_citations)} inward citations): a knowledge source for the field.")
        basis = ("links made by papers retained at the threshold" if res.config.mi_denominator != "all"
                 else "all in-corpus links")
        a.findings.append(f"MI is each community's share of the field's {res.mi_denominator()} citations "
                          f"({basis}).")
    a.tables["Community citation metrics (Table 1)"] = m.round(3)
    a.tables["Citation flows between communities (Table 2, rows cite columns)"] = C.set_axis(names, axis=0).set_axis(names, axis=1) \
        .reset_index().rename(columns={"index": "citing community"})
    a.chart = {"type": "heatmap", "title": "Citations between communities (rows cite columns; paper Table 2)",
               "rows": names, "cols": names, "matrix": C.values.tolist()}
    links = [(i, j, int(C.values[i, j])) for i in range(len(C)) for j in range(len(C))
             if i != j and C.values[i, j] > 0]
    if links:
        a.charts.append({"type": "sankey", "title": "Citation flows between communities "
                                                     "(left cites right; within-community citations left out)",
                         "names": names, "color_ids": [int(c) for c in C.index], "links": links})
    return a


def q_gaps(res) -> Answer:
    a = Answer("Q11", "")
    comms = [c for c in res.citation_matrix().index]
    M = _centroid_sim(res, comms)
    C = res.citation_matrix().values
    rows = []
    for i in range(len(comms)):
        for j in range(i + 1, len(comms)):
            rows.append({"community A": _name(res, comms[i]), "community B": _name(res, comms[j]),
                         "semantic similarity": round(float(M[i, j]), 3),
                         "citations between": int(C[i, j] + C[j, i])})
    t = pd.DataFrame(rows)
    if t.empty:
        return a
    t["gap"] = (t["semantic similarity"] >= t["semantic similarity"].quantile(0.75)) & \
               (t["citations between"] <= t["citations between"].median())
    t = t.sort_values(["gap", "semantic similarity"], ascending=[False, False])
    gaps = t[t["gap"]]
    a.findings.append(f"{len(gaps)} community pairs are among the most similar in content (top quarter) "
                      "but cite each other less than the median pair: possible knowledge gaps.")
    if len(gaps):
        g = gaps.iloc[0]
        a.findings.append(f"Strongest gap: {g['community A']} ↔ {g['community B']} "
                          f"(similarity {g['semantic similarity']}, {g['citations between']} citations).")
    a.tables["Semantic similarity vs citations (RQ4)"] = t
    return a


def q_bridges(res) -> Answer:
    a = Answer("Q12", "")
    G = res.graph()
    P = res.papers
    rows = []
    for i in G.nodes:
        k = G.degree(i)
        ci = P["community"].iat[i]
        if k < 3 or pd.isna(ci):
            continue
        cnt = Counter(P["community"].iat[j] for j in G[i] if pd.notna(P["community"].iat[j]))
        tot = sum(cnt.values())
        pc = 1 - sum((v / tot) ** 2 for v in cnt.values()) if tot else 0
        rows.append({"title": P.at[i, "title"], "year": P.at[i, "year"], "community": int(ci),
                     "links": k, "communities linked": len(cnt),
                     "participation": round(pc, 3), "cited_by": int(P.at[i, "cited_by"])})
    t = pd.DataFrame(rows)
    if t.empty:
        a.findings.append("No paper links to more than one community.")
        return a
    t = t.sort_values(["participation", "links"], ascending=False)
    n_b = int((t["participation"] >= 0.3).sum())
    a.findings.append(f"{n_b} papers spread at least 30% of their S3 links outside their own community "
                      "(participation ≥ 0.3). These are bridging papers between topics.")
    a.tables["Bridging papers"] = t.head(25)
    return a


# ===========================================================================
# Stage 4 - Evolution
# ===========================================================================
def q_growth(res) -> Answer:
    a = Answer("Q13", "")
    P = _clustered(res).dropna(subset=["year"])
    comms = _main_comms(res, 8)
    P = P[P["community"].isin(comms)]
    if P.empty:
        a.findings.append("No publication years available.")
        return a
    years = list(range(int(P["year"].min()), int(P["year"].max()) + 1))
    pivot = P.pivot_table(index="year", columns="community", values="title", aggfunc="count") \
        .reindex(years).fillna(0).astype(int)
    last = years[-1]
    rows = []
    for c in comms:
        s = pivot[c]
        recent = s[s.index > last - 3].sum()
        share_recent = recent / max(1, s.sum())
        field_recent = pivot.loc[pivot.index > last - 3].values.sum() / max(1, pivot.values.sum())
        ratio = share_recent / field_recent if field_recent else 0
        status = "emerging" if s.index[s > 0].min() > last - 5 else \
            "growing" if ratio >= 1.2 else "declining" if ratio <= 0.8 else "stable"
        rows.append({"community": c, "label": res.labels.get(c, ""), "first year": int(s.index[s > 0].min()),
                     "peak year": int(s.idxmax()), "papers": int(s.sum()),
                     "share in last 3 years": round(share_recent, 2),
                     "relative momentum": round(ratio, 2), "status": status})
    t = pd.DataFrame(rows)
    for st in ("emerging", "growing", "declining"):
        ids = t.loc[t["status"] == st, "community"].tolist()
        if ids:
            a.findings.append(f"{st.capitalize()}: " + ", ".join(_name(res, c) for c in ids) + ".")
    a.findings.append("Momentum compares a community's share of papers in the last 3 years with the "
                      "field's (1 = same pace, >1.2 growing, <0.8 declining).")
    a.tables["Community lifecycle"] = t
    a.tables["Papers per year"] = pivot.rename(columns=lambda c: _name(res, c)).reset_index()
    a.chart = {"type": "stack", "title": "Papers per year by community (stacked)", "x": years,
               "series": {_name(res, c): pivot[c].tolist() for c in comms},
               "color_ids": [int(c) for c in comms], "x_label": "year"}
    from .ktree import ktree_spec
    kt = ktree_spec(res)
    if kt:
        a.charts.append(kt)
    return a


def q_concepts(res) -> Answer:
    """Rising/fading concepts: share of papers using a keyword in the last 5 years vs before."""
    a = Answer("Q14", "")
    P = res.papers.dropna(subset=["year"]).copy()
    P["kws"] = [set(k.lower() for k in _split(v) if len(k) > 2) for v in P["keywords"]]
    P = P[P["kws"].map(len) > 0]
    if P.empty:
        a.findings.append("No author keywords in the export.")
        return a
    per_year = P.groupby("year").size()
    kept = per_year[per_year >= 20].index
    if len(kept) < 6:
        kept = per_year.index
    P = P[P["year"].isin(kept)]
    last = sorted(kept)[-5:]
    recent, earlier = P[P["year"].isin(last)], P[~P["year"].isin(last)]
    counts = Counter(k for ks in P["kws"] for k in ks)
    n = len(P)
    rows = []
    for k, cnt in counts.most_common(80):
        if cnt < 10:
            break
        r = sum(k in ks for ks in recent["kws"]) / max(1, len(recent))
        e = sum(k in ks for ks in earlier["kws"]) / max(1, len(earlier))
        has = P[P["kws"].map(lambda ks: k in ks)]
        comm = Counter(has["community"].dropna().astype(int))
        rows.append({"keyword": k, "papers": cnt, "share of papers": round(cnt / n, 3),
                     "first year": int(has["year"].min()),
                     f"share {int(last[0])}-{int(last[-1])}": round(r, 4), "share before": round(e, 4),
                     "growth (recent / before)": round(r / e, 2) if e > 0 else float("inf"),
                     "communities": len(comm),
                     "main community": _name(res, comm.most_common(1)[0][0]) if comm else ""})
    t = pd.DataFrame(rows)
    if t.empty:
        a.findings.append("No keyword is used by at least 10 papers.")
        return a
    t = t.sort_values("growth (recent / before)", ascending=False)
    specific = t[t["share of papers"] < 0.10]  # leave out near-universal search terms
    rising = specific[specific["growth (recent / before)"] >= 1.5].head(6)
    fading = specific[specific["growth (recent / before)"] <= 0.67].sort_values("growth (recent / before)").head(6)
    span = f"{int(last[0])}-{int(last[-1])}"
    if len(rising):
        a.findings.append(f"Rising concepts (share of papers in {span} vs before): " + ", ".join(
            f"{k} (×{g:.1f})" if g != float("inf") else f"{k} (new)"
            for k, g in zip(rising["keyword"], rising["growth (recent / before)"])) + ".")
    if len(fading):
        a.findings.append("Fading concepts: " + ", ".join(
            f"{k} (×{g:.2f})" for k, g in zip(fading["keyword"], fading["growth (recent / before)"])) + ".")
    if len(specific):
        spread = specific.sort_values("communities", ascending=False).iloc[0]
        a.findings.append(f"The most widely diffused specific concept is “{spread['keyword']}”, used in "
                          f"{spread['communities']} communities.")
    a.tables["Concept trends (RQ5)"] = t
    chart = specific[np.isfinite(specific["growth (recent / before)"])].head(12)
    a.chart = {"type": "bar", "title": f"Keyword growth: share in {span} ÷ share before",
               "labels": chart["keyword"].tolist(), "values": chart["growth (recent / before)"].tolist()}
    return a


# ===========================================================================
# Stage 5 - Actors
# ===========================================================================
def _author_table(res) -> pd.DataFrame:
    rows = []
    for i, r in res.papers.iterrows():
        for au in _authors(r["authors"]):
            rows.append({"author": au, "paper": i, "year": r["year"], "cited_by": r["cited_by"],
                         "community": r["community"]})
    return pd.DataFrame(rows)


def q_authors(res) -> Answer:
    """RFI: recency, frequency, impact, each scored 1-5 by quintile."""
    a = Answer("Q15", "")
    A = _author_table(res)
    if A.empty:
        a.findings.append("No author data.")
        return a
    g = A.groupby("author").agg(R=("year", "max"), F=("paper", "nunique"), I=("cited_by", "sum"))

    def q5(s):
        r = s.rank(method="average", pct=True)
        return np.ceil(r * 5).clip(1, 5).astype(int)

    g["R score"], g["F score"], g["I score"] = q5(g["R"].fillna(g["R"].min())), q5(g["F"]), q5(g["I"])
    g["RFI"] = g[["R score", "F score", "I score"]].sum(axis=1)
    main = A.dropna(subset=["community"]).groupby("author")["community"].agg(
        lambda s: _name(res, Counter(s.astype(int)).most_common(1)[0][0]))
    g["main community"] = main
    g = g.sort_values(["RFI", "I", "F"], ascending=False).reset_index()
    g = g.rename(columns={"R": "last year", "F": "papers", "I": "citations"})
    top = g.iloc[0]
    a.findings.append(f"{len(g)} authors. Top by RFI (recency + frequency + impact, each 1-5): "
                      f"{top['author']} ({int(top['papers'])} papers, {int(top['citations'])} citations).")
    a.tables["Leading authors (RFI)"] = g.head(25)
    return a


def q_brokers(res) -> Answer:
    a = Answer("Q16", "")
    A = _author_table(res).dropna(subset=["community"])
    if A.empty:
        a.findings.append("No author data.")
        return a
    rows = []
    for au, s in A.groupby("author"):
        cnt = Counter(s["community"].astype(int))
        if len(cnt) > 1:
            rows.append({"author": au, "communities": len(cnt), "papers": int(s["paper"].nunique()),
                         "diversity (Shannon)": round(_shannon(cnt.values()), 3),
                         "community mix": ", ".join(f"{_name(res, c)} ({v})" for c, v in cnt.most_common())})
    t = pd.DataFrame(rows)
    if t.empty:
        a.findings.append("No author publishes in more than one community.")
        return a
    t = t.sort_values(["communities", "diversity (Shannon)"], ascending=False)
    a.findings.append(f"{len(t)} authors publish in more than one community: they carry knowledge "
                      "between topics.")
    a.tables["Broker authors (RQ3)"] = t.head(25)
    comms = _main_comms(res, 10)
    top_auth = A.groupby("author")["paper"].nunique().sort_values(ascending=False).head(20).index
    M = (A[A["author"].isin(top_auth) & A["community"].isin(comms)]
         .groupby(["author", "community"])["paper"].nunique().unstack(fill_value=0)
         .reindex(index=top_auth, columns=comms, fill_value=0))
    a.chart = {"type": "heatmap", "title": "Papers by the 20 most productive authors in each community",
               "rows": list(M.index), "cols": [_name(res, c) for c in comms], "matrix": M.values.tolist()}
    return a


def q_collab(res) -> Answer:
    a = Answer("Q17", "")
    A = _author_table(res).dropna(subset=["community"])
    comms = _main_comms(res, 12)
    ac = A.groupby("author")["community"].agg(lambda s: set(s.astype(int)))
    shared = Counter()
    for cs in ac:
        cs = sorted(c for c in cs if c in comms)
        for i in range(len(cs)):
            for j in range(i + 1, len(cs)):
                shared[(cs[i], cs[j])] += 1
    M = _centroid_sim(res, comms)
    ix = {c: k for k, c in enumerate(comms)}
    rows = [{"community A": _name(res, c1), "community B": _name(res, c2),
             "semantic similarity": round(float(M[ix[c1], ix[c2]]), 3),
             "shared authors": shared.get((c1, c2), 0)}
            for k1, c1 in enumerate(comms) for c2 in comms[k1 + 1:]]
    t = pd.DataFrame(rows)
    if t.empty:
        return a
    t["opportunity"] = (t["semantic similarity"] >= t["semantic similarity"].quantile(0.75)) & \
                       (t["shared authors"] <= t["shared authors"].median())
    t = t.sort_values(["opportunity", "semantic similarity"], ascending=[False, False])
    a.findings.append(f"{int(t['opportunity'].sum())} community pairs are among the most similar in content "
                      "(top quarter) but share fewer authors than the median pair: collaboration opportunities.")
    a.tables["Collaboration gaps (RQ6)"] = t
    return a


def q_geo(res) -> Answer:
    a = Answer("Q18", "")
    F = _paper_affils(res)
    if F.empty:
        a.findings.append("No affiliation data.")
        return a
    F["community"] = res.papers["community"].reindex(F["paper"]).to_numpy()
    F["year"] = res.papers["year"].reindex(F["paper"]).to_numpy()
    cpp = F[F["country"] != ""].drop_duplicates(["paper", "country"])
    top_c = cpp["country"].value_counts().head(10)
    a.findings.append("Leading countries: " + ", ".join(f"{k} ({v})" for k, v in top_c.head(5).items()) + ".")
    rows = []
    for c in _main_comms(res, 10):
        sub = F[F["community"] == c]
        countries = sub.drop_duplicates(["paper", "country"])["country"].replace("", np.nan).dropna()
        insts = sub.drop_duplicates(["paper", "institution"])["institution"]
        rows.append({"community": c, "label": res.labels.get(c, ""),
                     "top countries": ", ".join(f"{k} ({v})" for k, v in countries.value_counts().head(4).items()),
                     "top institutions": "; ".join(f"{k} ({v})" for k, v in insts.value_counts().head(3).items())})
    a.tables["Countries and institutions per community (RQ7/RQ8)"] = pd.DataFrame(rows)
    a.tables["Top institutions"] = F.drop_duplicates(["paper", "institution"])["institution"] \
        .value_counts().head(20).rename_axis("institution").reset_index(name="papers")
    by_year = cpp[cpp["country"].isin(top_c.index[:6])].pivot_table(
        index="year", columns="country", values="paper", aggfunc="count").fillna(0).astype(int)
    if not by_year.empty:
        a.tables["Countries over time (RQ9)"] = by_year.reset_index()
        a.chart = {"type": "line", "title": "Papers per year, top countries",
                   "x": [int(x) for x in by_year.index], "series": {k: by_year[k].tolist() for k in by_year.columns},
                   "x_label": "year"}
    comms = _main_comms(res, 10)
    cc = (cpp[cpp["community"].isin(comms) & cpp["country"].isin(top_c.index[:12])]
          .groupby(["country", "community"])["paper"].nunique().unstack(fill_value=0)
          .reindex(index=top_c.index[:12], columns=comms, fill_value=0))
    a.charts.append({"type": "heatmap", "title": "Papers by country and community",
                     "rows": list(cc.index), "cols": [_name(res, c) for c in comms], "matrix": cc.values.tolist()})
    a.findings.append("Institution names are taken from the affiliation part that names a university, "
                      "institute, school etc.; spelling variants may need merging.")
    return a


def q_journals(res) -> Answer:
    a = Answer("Q19", "")
    P = res.papers.dropna(subset=["source"])
    if P.empty:
        a.findings.append("No source/journal data.")
        return a
    rows = []
    for j, s in P.groupby("source"):
        cnt = Counter(s["community"].dropna().astype(int))
        rows.append({"journal": j, "papers": len(s), "citations": int(s["cited_by"].sum()),
                     "communities": len(cnt), "topic diversity (Shannon)": round(_shannon(cnt.values()), 3),
                     "main community": _name(res, cnt.most_common(1)[0][0]) if cnt else ""})
    t = pd.DataFrame(rows).sort_values("papers", ascending=False)
    top = t.iloc[0]
    a.findings.append(f"The main outlet is {top['journal']} ({top['papers']} papers).")
    mixed = t[t["papers"] >= 3].sort_values("topic diversity (Shannon)", ascending=False)
    if len(mixed):
        a.findings.append(f"The journal that mixes most communities is {mixed.iloc[0]['journal']}; "
                          "specialised journals have diversity near 0.")
    per_c = []
    for c in _main_comms(res, 10):
        vc = P[P["community"] == c]["source"].value_counts().head(3)
        per_c.append({"community": c, "label": res.labels.get(c, ""),
                      "top journals": "; ".join(f"{k} ({v})" for k, v in vc.items())})
    a.tables["Journals (RQ11)"] = t.head(25)
    a.tables["Top journals per community"] = pd.DataFrame(per_c)
    return a


def _funders(v) -> List[str]:
    """Funder names from a Scopus funding cell ('NSFC, NSFC (71872...)' -> 'NSFC')."""
    out = []
    for f in _split(v):
        name = re.sub(r",?\s*\([^)]*\)\s*$", "", f).strip()
        parts = [x.strip() for x in name.split(",")]
        if len(parts) > 1 and parts[-1].replace("-", "").isupper() and len(parts[-1]) <= 10:
            parts = parts[:-1]
        name = ", ".join(parts)
        if len(name) > 3 and not name.isdigit():
            out.append(name)
    return out


def q_funding(res) -> Answer:
    a = Answer("Q20", "")
    rows = [(i, name) for i, v in res.papers["funding"].items() for name in _funders(v)]
    if not rows:
        a.findings.append("No funding data in the export.")
        return a
    F = pd.DataFrame(rows, columns=["paper", "funder"]).drop_duplicates()
    F["community"] = res.papers["community"].reindex(F["paper"]).to_numpy()
    share = F["paper"].nunique() / len(res.papers)
    top = F["funder"].value_counts().head(15)
    a.findings.append(f"{share:.0%} of papers report funding. Top funder: {top.index[0]} ({top.iloc[0]} papers).")
    per_c = []
    for c in _main_comms(res, 10):
        sub = F[F["community"] == c]
        n_p = (res.papers["community"] == c).sum()
        per_c.append({"community": c, "label": res.labels.get(c, ""),
                      "funded share": round(sub["paper"].nunique() / max(1, n_p), 2),
                      "top funders": "; ".join(f"{k} ({v})" for k, v in sub["funder"].value_counts().head(3).items())})
    a.tables["Funding per community (RQ10)"] = pd.DataFrame(per_c)
    a.tables["Top funders"] = top.rename_axis("funder").reset_index(name="papers")
    return a


# ===========================================================================
# Stage 6 - Interdisciplinarity
# ===========================================================================
def q_integration(res) -> Answer:
    a = Answer("Q21", "")
    rs = _semantic_integration(res)
    P = res.papers.assign(integration=rs)
    rows = []
    for c in _main_comms(res):
        s = P.loc[P["community"] == c, "integration"].dropna()
        rows.append({"community": c, "label": res.labels.get(c, ""), "papers": len(s),
                     "mean integration": round(float(s.mean()), 4) if len(s) else None,
                     "share integrative (>0)": round(float((s > 0).mean()), 3) if len(s) else None})
    t = pd.DataFrame(rows).sort_values("mean integration", ascending=False)
    a.findings.append("Integration is the Rao-Stirling diversity of each paper's S3 links across "
                      "communities, weighted by how different those communities are (1 - mean S3). "
                      "It needs no subject categories.")
    big = t[t["papers"] >= 10]
    if len(big) >= 2:
        a.findings.append(f"Among communities with at least 10 papers, the most interdisciplinary is "
                          f"{_name(res, big.iloc[0]['community'])} and the most self-contained is "
                          f"{_name(res, big.iloc[-1]['community'])}.")
    top = P.dropna(subset=["integration"]).sort_values("integration", ascending=False).head(15)
    a.tables["Integration per community"] = t
    a.tables["Most integrative papers"] = top[["title", "year", "community", "cited_by", "integration"]].round(4)
    a.chart = {"type": "bar", "title": "Mean integration per community",
               "labels": [_name(res, c) for c in t["community"]], "values": t["mean integration"].fillna(0).tolist()}
    return a


def q_integration_impact(res) -> Answer:
    from scipy.stats import spearmanr
    a = Answer("Q22", "")
    rs = _semantic_integration(res)
    P = res.papers.assign(integration=rs).dropna(subset=["integration", "year"])
    if len(P) < 10:
        a.findings.append("Too few linked papers with a year.")
        return a
    now = int(P["year"].max()) + 1
    P = P.assign(cites_per_year=P["cited_by"] / (now - P["year"]).clip(lower=1))
    rho, p = spearmanr(P["integration"], P["cites_per_year"])
    P["group"] = np.where(P["integration"] > 0, "integrative", "single-community")
    g = P.groupby("group").agg(papers=("title", "size"), median_cites_per_year=("cites_per_year", "median"),
                               mean_cites_per_year=("cites_per_year", "mean")).round(2).reset_index()
    if np.isnan(rho):
        a.findings.append("Integration or citations do not vary enough to test a relationship.")
        return a
    strength = "negligible" if abs(rho) < 0.1 else "weak" if abs(rho) < 0.3 else "moderate" if abs(rho) < 0.5 else "strong"
    sig = "statistically significant" if p < 0.05 else "not statistically significant"
    a.findings.append(f"Spearman correlation between integration and citations per year: "
                      f"ρ = {rho:.2f} (p = {p:.3f}): a {strength}, {sig} relationship.")
    a.findings.append("A positive ρ would mean that papers connecting communities are cited more "
                      "(age-adjusted); compare the medians in the table.")
    a.tables["Integration vs impact"] = g
    return a


def q_integration_trend(res) -> Answer:
    a = Answer("Q23", "")
    rs = _semantic_integration(res)
    P = res.papers.assign(integration=rs).dropna(subset=["integration", "year"])
    if P.empty:
        return a
    y = P.groupby("year")["integration"].agg(["mean", "size"]).reset_index()
    y["year"] = y["year"].astype(int)
    slope = float(np.polyfit(y["year"], y["mean"], 1)[0]) if len(y) > 2 else 0.0
    base = float(y["mean"].mean()) or 1e-9
    rel10 = slope * 10 / base
    trend = "roughly stable" if abs(rel10) < 0.2 else ("becoming more interdisciplinary" if slope > 0
                                                        else "becoming less interdisciplinary")
    a.findings.append(f"Mean integration changes by {slope:+.5f} per year ({rel10:+.0%} per decade "
                      f"relative to its average): the field is {trend}.")
    share_int = float((P["integration"] > 0).mean())
    if share_int < 0.1:
        a.findings.append(f"Only {share_int:.0%} of linked papers connect to more than one community, so "
                          "this trend rests on few papers.")
    a.tables["Integration by year"] = y.rename(columns={"mean": "mean integration", "size": "papers"}).round(4)
    a.chart = {"type": "line", "title": "Mean integration by year", "x": y["year"].tolist(),
               "series": {"integration": y["mean"].round(4).tolist()}, "x_label": "year"}
    return a


# ===========================================================================
# Catalogue (ranked: the order is the guided pipeline)
# ===========================================================================
def q_predict(res) -> Answer:
    """Temporal link prediction: does S3 rank the earlier papers a new paper will cite?"""
    from .predict import citation_prediction
    a = Answer("Q24", "")
    r = citation_prediction(res)
    if r is None:
        a.findings.append("Too few new papers cite earlier papers in the corpus to test prediction "
                          "(needs at least 10 citing papers in the last 5 years).")
        return a
    t = r["table"].set_index("method")
    k = r["k"]
    s3 = t.loc["S3 similarity (text only)"]
    pop = t.loc["Popularity (citations up to split year)"]
    a.findings.append(
        f"Split: papers up to {r['split']} are known; {r['new_papers']} papers published {r['test']} "
        f"make {r['citations']} citations to the {r['candidates']} earlier papers. For each new paper, every earlier "
        f"paper is ranked and checked against what it really cites (chance: "
        f"{r['base_rate']:.2%} of pairs are citations).")
    a.findings.append(
        f"S3 similarity alone, from title and abstract only, ranks a cited paper above a non-cited one "
        f"{s3['ROC AUC']:.0%} of the time (ROC AUC {s3['ROC AUC']:.2f}; random 0.50). "
        f"Its top {k} contain {s3[f'recall@{k}']:.0%} of a new paper's in-corpus citations on average "
        f"(popularity: {pop[f'recall@{k}']:.0%}, AUC {pop['ROC AUC']:.2f}).")
    combo = "S3 + popularity (fitted on earlier years)"
    if combo in t.index:
        c = t.loc[combo]
        a.findings.append(
            f"Combining S3 with popularity, fitted on the previous 5 years only, "
            f"gives AUC {c['ROC AUC']:.2f} and recall@{k} {c[f'recall@{k}']:.0%}: "
            + ("content and popularity add up." if c["ROC AUC"] > max(s3["ROC AUC"], pop["ROC AUC"]) + 0.005
               else "popularity adds little beyond content.")
            + (f" At the very top, popularity alone is better (its first-ranked paper is cited "
               f"{pop['hit@1']:.0%} of the time vs {c['hit@1']:.0%}): new papers cite the field's classics first."
               if pop["hit@1"] > c["hit@1"] + 0.02 else ""))
    bc = "Bibliographic coupling (needs the new paper's references)"
    if bc in t.index:
        b = t.loc[bc]
        a.findings.append(
            f"For comparison, bibliographic coupling reaches AUC {b['ROC AUC']:.2f} (recall@{k} "
            f"{b[f'recall@{k}']:.0%}), but it needs the new paper's own reference list, so it cannot "
            f"forecast. S3 needs only the text.")
    if r["linked"] and r["linked"]["pairs"]:
        L = r["linked"]
        a.findings.append(
            f"Of the {L['pairs']} (new paper, earlier paper) pairs above the S3 threshold "
            f"{L['threshold']:.2f}, {L['cited'] / L['pairs']:.0%} became citations, "
            f"{L['cited'] / L['pairs'] / max(r['base_rate'], 1e-12):.0f}× the chance rate: S3 links point "
            f"to where citations go, and the uncited ones are candidates for Q11's knowledge gaps.")
    a.findings.append("Citations are " + ("exact (Scopus API)." if res.citation_source == "scopus_api" else
                      "matched from the References text; with exact Scopus citations the test is cleaner."))
    a.tables["Citation prediction: new papers vs earlier papers"] = r["table"]
    a.chart = {"type": "line", "title": f"Recall@k: share of a new paper's citations among its top-k "
                                        f"ranked earlier papers ({r['test']})",
               "x": r["ks"], "x_labels": [str(x) for x in r["ks"]],
               "series": {m.split(" (")[0]: v for m, v in r["curves"].items()},
               "x_label": "k (top-ranked earlier papers)"}
    a.charts.append({"type": "bar", "title": "ROC AUC by method (0.5 = chance)",
                     "labels": [m.split(" (")[0] for m in r["table"]["method"]],
                     "values": r["table"]["ROC AUC"].tolist()})
    return a


# ===========================================================================
# Multiplex, homophily and direction (Q35-Q37)
# ===========================================================================
def q_layers(res) -> Answer:
    """Q35: overlap between the text, citation and co-authorship layers."""
    from .layers import overlap
    a = Answer("Q35", "")
    J, lift, summ = overlap(res)
    if len(J) < 2:
        a.findings.append("Only one layer could be built (no references, citations or authors in the export).")
        return a
    text_kind = "the links of the text map" if not res.config.map_weights else \
        "the same number of most similar pairs by S3 as the map has links"
    a.findings.append(f"The same {len(res.papers)} papers are tied in {len(J)} ways: " +
                      ", ".join(f"{r.layer} ({r.ties:,} ties)" for r in summ.itertuples()) +
                      f". The text layer is {text_kind}.")
    if "text" in lift.index:
        for b in [x for x in J.columns if x != "text"]:
            l = lift.loc[b, "text"]
            sh = summ.set_index("layer").at[b, "share also a text link"]
            if pd.notna(l):
                pc = (lambda x: f"{x:.1%}" if 0 < x < 0.01 or 0.99 < x < 1 else f"{x:.0%}")
                a.findings.append(f"{pc(sh)} of {b} ties are also text links, {l:.0f}× more than for a random "
                                  f"pair; the other {pc(1 - sh)} are relations the text map does not show."
                                  if l >= 10 else
                                  f"{pc(sh)} of {b} ties are also text links, {l:.1f}× more than for a random "
                                  f"pair; the other {pc(1 - sh)} are relations the text map does not show.")
    off = [(J.iat[i, j], J.index[i], J.columns[j]) for i in range(len(J)) for j in range(i + 1, len(J))]
    off.sort(reverse=True)
    a.findings.append(f"The most similar layers are {off[0][1]} and {off[0][2]} (Jaccard = {off[0][0]:.2f}); "
                      f"the least similar are {off[-1][1]} and {off[-1][2]} (Jaccard = {off[-1][0]:.2f}).")
    a.findings.append("Lift = how many times more often a tie of one layer is also a tie of the other than a "
                      "random pair is (1 = chance). Layers that rarely overlap with text add the most when "
                      "mixed into the map (S3Config(map_weights=...)).")
    s = summ.copy()
    s["density"] = s["density"].map(lambda x: f"{x:.5f}")
    s["share also a text link"] = s["share also a text link"].round(3)
    a.tables["Layers"] = s
    a.tables["Jaccard overlap between layers"] = J.round(3).reset_index().rename(columns={"index": "layer"})
    a.tables["Lift over chance (rows: tie in this layer; columns: also a tie here)"] = \
        lift.round(1).reset_index().rename(columns={"index": "layer"})
    a.chart = {"type": "heatmap", "title": "Overlap between layers (Jaccard index of their ties)",
               "rows": list(J.index), "cols": list(J.columns), "matrix": J.round(3).values.tolist()}
    return a


def q_homophily(res) -> Answer:
    """Q36: attribute assortativity per layer."""
    from .layers import homophily
    a = Answer("Q36", "")
    h = homophily(res)
    if h.empty:
        a.findings.append("No paper attributes (journal, country, funder) to compare.")
        return a
    R = h.pivot(index="attribute", columns="layer", values="r")
    R = R[[c for c in h["layer"].unique()]]
    a.findings.append("Assortativity r runs from -1 to +1: r > 0 means ties fall within the same group "
                      "(same community, journal, country or funder) more often than chance; r ≈ 0 no pattern; "
                      "r < 0 ties cross groups.")
    few = h.pivot(index="attribute", columns="layer", values="ties") < 20
    for att in R.index:
        row = R.loc[att][~few.loc[att, R.columns]].dropna()
        if row.empty:
            continue
        hi, lo = row.idxmax(), row.idxmin()
        a.findings.append(f"{att.capitalize()}: strongest in {hi} (r = {row[hi]:.2f}), weakest in {lo} "
                          f"(r = {row[lo]:.2f}).")
    if "community" in R.index and "text" in R.columns:
        from .hybrid import NAMES
        mixed = [NAMES[k] for k in (res.config.map_weights or {}) if k != "s3"]
        a.findings.append("Communities are built from the map, so community homophily in the text layer"
                          + (f" (and in {', '.join(mixed)}, which is mixed into this map)" if mixed else "")
                          + " is high by construction; read it in the other layers.")
    a.findings.append("Findings leave out layers with fewer than 20 ties for an attribute; the table shows all.")
    a.findings.append("Assortativity describes, it does not explain: shared journals or countries may cause "
                      "the ties, or follow from them.")
    t = h.copy()
    t["within"], t["expected"], t["r"] = t["within"].round(3), t["expected"].round(3), t["r"].round(3)
    t = t.rename(columns={"ties": "ties with both values", "within": "share within the same group",
                          "expected": "expected by chance", "r": "assortativity r"})
    a.tables["Homophily by attribute and layer"] = t
    a.chart = {"type": "heatmap", "title": "Homophily (assortativity r) by attribute and layer",
               "rows": list(R.index), "cols": list(R.columns),
               "matrix": R.round(2).fillna(0).values.tolist()}
    return a


def q_direction(res) -> Answer:
    """Q37: PageRank on citations and one-way vs two-way citation flows between communities."""
    from .layers import flow_direction, pagerank
    from scipy.stats import spearmanr
    a = Answer("Q37", "")
    P = res.papers
    pr = pagerank(res)
    indeg = pd.Series(0, index=P.index)
    outdeg = pd.Series(0, index=P.index)
    for i, j in res.citation_pairs:
        indeg.iat[j] += 1
        outdeg.iat[i] += 1
    t = pd.DataFrame({"title": P["title"], "year": P["year"],
                      "community": [_display_name(res, c) if pd.notna(c) else "" for c in P["community"]],
                      "PageRank": pr, "cited in corpus (in-degree)": indeg,
                      "cites in corpus (out-degree)": outdeg, "global citations": P["cited_by"]})
    t = t.sort_values("PageRank", ascending=False)
    t["rank by PageRank"] = range(1, len(t) + 1)
    t["rank by in-degree"] = t["cited in corpus (in-degree)"].rank(ascending=False, method="min").astype(int)
    cited = t[t["cited in corpus (in-degree)"] > 0]
    a.findings.append("Citations have a direction: a paper can only cite older papers, so paper-level "
                      "reciprocity is always zero. In-degree = times cited inside the corpus; PageRank also "
                      "counts who cites: being cited by well-cited papers counts more.")
    top = t.iloc[0]
    a.findings.append(f"Highest PageRank: “{top['title']}” ({_int(top['year'])}), cited "
                      f"{int(top['cited in corpus (in-degree)'])} times in the corpus.")
    if len(cited) >= 5:
        rho = spearmanr(cited["PageRank"], cited["cited in corpus (in-degree)"]).correlation
        a.findings.append(f"PageRank and in-degree agree at Spearman ρ = {rho:.2f} over the {len(cited)} cited papers.")
        jump = cited.head(50)
        jump = jump[jump["rank by in-degree"] - jump["rank by PageRank"] >= 10]
        if len(jump):
            j = jump.iloc[0]
            a.findings.append(f"Ranked much higher by PageRank than by counts: “{j['title']}” "
                              f"(#{j['rank by PageRank']} vs #{j['rank by in-degree']}): few citations, "
                              "but from central papers.")
    show = t.head(25).copy()
    show["PageRank"] = show["PageRank"].round(5)
    a.tables["Papers by PageRank on in-corpus citations"] = show
    comm, pairs, tot = flow_direction(res)
    if not comm.empty and tot["flows"]:
        a.findings.append(f"Between communities, {int(tot['flows'])} citations flow; reciprocity = "
                          f"{tot['reciprocity']:.2f} (1 = every flow returned in full, 0 = all one-way), against "
                          f"{tot['expected']:.2f} if each citation went either way at random. "
                          + ("Flows are more one-way than chance: knowledge moves in a hierarchy."
                             if tot["reciprocity"] < tot["expected"] - 0.02 else
                             "Flows are about as two-way as chance."))
        c = comm.dropna(subset=["source score"]).sort_values("source score", ascending=False)
        if len(c):
            s, b = c.iloc[0], c.iloc[-1]
            a.findings.append(f"Main knowledge source: {_name(res, s.community)} (source score "
                              f"{s['source score']:+.2f}, median year {_int(s['median year'])}); most dependent on "
                              f"others: {_name(res, b.community)} ({b['source score']:+.2f}, median year "
                              f"{_int(b['median year'])}). Older communities tend to be sources because "
                              "citations point back in time.")
        comm = comm.assign(community=[_name(res, x) for x in comm["community"]])
        comm["source score"] = comm["source score"].round(3)
        a.tables["Knowledge sources and users (source score +1 = only cited, -1 = only citing)"] = comm
        if not pairs.empty:
            pairs = pairs.sort_values(["total", "asymmetry"], ascending=False)
            pairs["A"] = [_name(res, x) for x in pairs["A"]]
            pairs["B"] = [_name(res, x) for x in pairs["B"]]
            pairs["asymmetry"] = pairs["asymmetry"].round(3)
            a.tables["Direction of flows between community pairs"] = pairs.head(40)
        cc = c.assign(name=[_display_name(res, x) for x in c["community"]])
        a.chart = {"type": "bar", "title": "Knowledge source score per community "
                                           "(+1 = only cited by other communities, -1 = only citing them)",
                   "labels": cc["name"].tolist(), "values": cc["source score"].round(3).tolist(), "max": 30}
    a.charts.append({"type": "bar", "title": "Top papers by PageRank on in-corpus citations",
                     "labels": [_clip(x, 60) for x in show["title"]],
                     "values": show["PageRank"].tolist(), "max": 15})
    return a


def _pq(fn: str, res) -> Answer:
    from . import predq
    return getattr(predq, fn)(res)


QUESTIONS: List[Question] = [
    Question("Q1", 1, "Essential", "What similarity threshold structures this field best?",
             "Shows the S3 threshold choice is data-driven, not arbitrary.",
             "Threshold scan 0.75-0.85, SII = silhouette × retained papers", (), q_threshold),
    Question("Q2", 1, "Essential", "How clear is the community structure?",
             "Reviewers ask whether the communities are real.", "Silhouette, modularity, density", (), q_quality),
    Question("Q3", 1, "Essential", "Which papers do not belong to the field?",
             "Cleans the corpus; S3 detects off-topic papers that keyword searches let in.",
             "Papers without S3 links above the threshold", (), q_outliers),
    Question("Q4", 1, "Recommended", "Does S3 agree with classic bibliometric methods?",
             "Validates S3 against bibliographic coupling and co-citation.",
             "Pearson correlation, top-10 neighbour overlap, NMI/ARI vs BC communities",
             ("references",), q_validation),
    Question("Q35", 1, "Recommended", "Do text, citation and co-authorship ties agree?",
             "Shows what each map option adds: relations the text map misses.",
             "Multiplex layers (text, coupling, co-citation, direct citation, co-authorship): "
             "Jaccard overlap and lift over chance", (), q_layers),
    Question("Q5", 2, "Essential", "What are the main research communities?",
             "The core result: the intellectual structure of the field.",
             "Leiden (directed modularity) on the time-ordered S3 graph", (), q_communities),
    Question("Q6", 2, "Essential", "What is each community about?",
             "Turns clusters into named topics.", "c-TF-IDF terms and author keywords per community", (), q_topics),
    Question("Q7", 2, "Essential", "Which papers are at the core of each community?",
             "Tells readers what to read first.", "Eigenvector, degree and betweenness centrality", (), q_central),
    Question("Q8", 2, "Essential", "Which papers is each community built on?",
             "Appendix Table C1 of the S3 paper: local vs global foundations.",
             "Top local (in-community) and global (Scopus) citations", (), q_foundational),
    Question("Q9", 2, "Recommended", "How close are the communities to each other?",
             "Shows the map's layout: neighbouring vs distant topics.",
             "Mean S3 between communities", (), q_similarity),
    Question("Q10", 3, "Essential", "How do communities cite each other, and which are mature?",
             "Tables 1-2 of the S3 paper: knowledge flows, QI and MI.",
             "In-corpus citation matrix, QI, MI", ("citations",), q_flows),
    Question("Q11", 3, "Recommended", "Where are the knowledge gaps?",
             "Close topics that ignore each other are research opportunities.",
             "Semantic similarity vs citations per community pair (RQ4)", ("citations",), q_gaps),
    Question("Q12", 3, "Recommended", "Which papers bridge communities?",
             "Finds boundary-spanning work.", "Participation coefficient of S3 links", (), q_bridges),
    Question("Q37", 3, "Recommended", "Which papers and communities are knowledge sources, and are flows one-way?",
             "Direction reveals hierarchy: who builds on whom.",
             "PageRank on in-corpus citations; source score and reciprocity of community flows vs chance",
             ("citations",), q_direction),
    Question("Q13", 4, "Essential", "Which communities are emerging, growing or declining?",
             "Shows where the field is heading.", "Papers per year, momentum vs field", ("years",), q_growth),
    Question("Q14", 4, "Recommended", "Which concepts are rising or fading, and how far have they spread?",
             "Tracks concept diffusion across time and communities.",
             "Keyword share trend and community spread (RQ5)", ("keywords", "years"), q_concepts),
    Question("Q15", 5, "Recommended", "Who are the leading authors?",
             "Identifies key researchers by recency, output and impact.",
             "RFI score (recency, frequency, impact)", ("authors",), q_authors),
    Question("Q16", 5, "Recommended", "Which authors work across communities?",
             "Brokers carry ideas between topics.", "Authors in ≥ 2 communities, Shannon diversity (RQ3)",
             ("authors",), q_brokers),
    Question("Q17", 5, "Optional", "Where could communities collaborate more?",
             "Similar topics with few shared authors are collaboration opportunities.",
             "Semantic similarity vs shared authors (RQ6)", ("authors",), q_collab),
    Question("Q18", 5, "Recommended", "Which countries and institutions lead each community?",
             "Geography of the field and its change over time.",
             "Affiliation parsing; countries per community and per year (RQ7-RQ9)",
             ("affiliations",), q_geo),
    Question("Q19", 5, "Recommended", "Where is each community published, and which journals mix topics?",
             "Helps choose outlets; shows specialised vs integrative journals.",
             "Journals per community, Shannon diversity (RQ11)", ("source",), q_journals),
    Question("Q20", 5, "Optional", "Who funds which communities?",
             "Shows funding priorities.", "Funding agencies per community (RQ10)", ("funding",), q_funding),
    Question("Q36", 5, "Recommended", "Do papers link within their own community, journal, country or funder?",
             "Homophily: whether ties stay inside groups, and in which layers.",
             "Attribute assortativity r per layer, with the share expected by chance", (), q_homophily),
    Question("Q21", 6, "Recommended", "How interdisciplinary is each community?",
             "Measures integration across topics with S3 instead of subject categories.",
             "Rao-Stirling diversity of S3 links, disparity = 1 - mean S3", (), q_integration),
    Question("Q22", 6, "Optional", "Are interdisciplinary papers cited more?",
             "Tests whether integration pays off in impact.",
             "Spearman: integration vs citations per year", ("years", "impact"), q_integration_impact),
    Question("Q23", 6, "Optional", "Is the field becoming more interdisciplinary?",
             "Trend of integration over time.", "Mean integration per year", ("years",), q_integration_trend),
    Question("Q24", 7, "Recommended", "Does S3 similarity predict which earlier papers new papers will cite?",
             "Predictive validation of S3: content similarity now, citations later.",
             "Temporal link prediction (last 5 years vs earlier papers): ROC AUC, MAP, recall@10 for "
             "S3, popularity, both, and bibliographic coupling", ("citations", "years"), q_predict),
    Question("Q25", 7, "Recommended", "Which knowledge gaps will close: which communities will start citing each other?",
             "Turns Q11 into a forecast.", "Temporal link prediction on community pairs: S3 centroid similarity vs sizes",
             ("citations", "years"), lambda r: _pq("q_gaps_close", r)),
    Question("Q26", 7, "Optional", "Which concepts will enter which communities?",
             "Turns Q14 into a forecast.", "Temporal link prediction (keyword, community): S3 similarity vs frequency",
             ("keywords", "years"), lambda r: _pq("q_concepts_enter", r)),
    Question("Q27", 7, "Optional", "Which authors will move into a new community?",
             "Turns Q16 into a forecast.", "Temporal link prediction (author, community): S3 similarity vs community size",
             ("authors", "years"), lambda r: _pq("q_authors_move", r)),
    Question("Q28", 7, "Optional", "Who will co-author next?",
             "Turns Q17 into a forecast.", "Temporal link prediction on author pairs: S3 similarity vs Adamic-Adar",
             ("authors", "years"), lambda r: _pq("q_coauthors", r)),
    Question("Q29", 7, "Optional", "Which institutions and funders will enter which communities?",
             "Turns Q18 and Q20 into forecasts.", "Temporal link prediction (institution/funder, community)",
             ("affiliations", "years"), lambda r: _pq("q_orgs_enter", r)),
    Question("Q30", 8, "Optional", "Which community does a new or outlier paper belong to?",
             "Improves Q3's nearest-paper rule.", "Node classification: GraphSAGE vs nearest paper, centroid, text classifier",
             (), lambda r: _pq("q_gnn_classify", r)),
    Question("Q31", 8, "Optional", "Which recent papers will be highly cited, and is impact driven by content or network position?",
             "Extends Q8 and Q22.", "Node regression: text model vs network-position model vs GraphSAGE (temporal split)",
             ("years", "impact"), lambda r: _pq("q_gnn_impact", r)),
    Question("Q32", 8, "Optional", "Who fits a topic: authors, institutions and funders for expertise search or partner matching",
             "Extends Q15-Q20.", "Heterogeneous graph learning: R-GCN on the knowledge graph, held-out link test",
             ("authors",), lambda r: _pq("q_gnn_hetero", r)),
    Question("Q33", 8, "Optional", "Which communities will grow, emerge or decline next?",
             "Extends Q13 and Q23.", "Temporal graph learning: GCN + GRU over yearly community shares",
             ("years",), lambda r: _pq("q_gnn_temporal", r)),
    Question("Q34", 8, "Optional", "Which links are missing from the knowledge graph?",
             "Extends Q17, Q18 and Q20.", "KG embeddings: TransE and RotatE, filtered MRR/Hits@10, composed entity → community links",
             ("authors",), lambda r: _pq("q_gnn_kg", r)),
]
_BY_ID = {q.qid: q for q in QUESTIONS}

# Analyses in the web app that are not part of the guided pipeline yet.
PLANNED = [
    ("Topic evolution with merge/split events", "Evolution"),
    ("Sub-communities and community dendrogram", "Map the field"),
    ("Full-text (PDF) content analysis", "Map the field"),
]


def available(res, q: Question) -> Tuple[bool, str]:
    if not res.best:
        return False, "No community structure found."
    missing = [n for n in q.needs if not _has(res, n)]
    if missing:
        msgs = {"citations": "no in-corpus citations were found (needs References and EID)",
                "years": "the papers cover fewer than 3 publication years",
                "impact": "fewer than 20% of the papers have been cited yet"}
        return False, "Not enough data: " + "; ".join(msgs.get(m, f"missing column '{m}'") for m in missing) + "."
    return True, ""


def prediction_on(res, prediction: Optional[bool] = None) -> bool:
    return bool(res.config.prediction if prediction is None else prediction)


def question_table(res, available_only: bool = False, prediction: Optional[bool] = None) -> pd.DataFrame:
    """The ranked questions. The prediction stages (7-8) are listed only when enabled."""
    rows = []
    on = prediction_on(res, prediction)
    for rank, q in enumerate([q for q in QUESTIONS if on or not q.prediction], 1):
        ok, why = available(res, q)
        rows.append({"rank": rank, "id": q.qid, "stage": f"{q.stage}. {STAGES[q.stage]}",
                     "priority": q.priority, "question": q.text, "method": q.method,
                     "available": ok, "missing": why})
    t = pd.DataFrame(rows)
    return t[t["available"]] if available_only else t


PREDICTION_OFF = ("Prediction questions (Q24-Q34) are off. Enable them with "
                  "S3Config(prediction=True), or answer(..., prediction=True).")


def answer(res, qid: str, prediction: Optional[bool] = None) -> Answer:
    q = _BY_ID.get(qid.upper())
    if q is None:
        raise ValueError(f"Unknown question {qid}. Valid: {', '.join(_BY_ID)}")
    if q.prediction and not prediction_on(res, prediction):
        return Answer(q.qid, q.text, findings=[PREDICTION_OFF])
    ok, why = available(res, q)
    if not ok:
        return Answer(q.qid, q.text, findings=[f"Not available for this dataset. {why}"])
    a = q.fn(res)
    a.qid, a.question = q.qid, q.text
    return a
