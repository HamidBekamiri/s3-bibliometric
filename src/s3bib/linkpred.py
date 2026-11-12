"""Link prediction (Q25-Q29): which links will appear?

Every task uses the same honest design as Q24:
  1. split the corpus in time (default: the last 5 years are the test window),
  2. score candidate links with information from before the split only,
  3. check the scores against the links that really appeared in the test window
     (ROC AUC, average precision, precision@k), against simple baselines,
  4. then score the links that do not exist yet using all years: the forecast.

S3 scores are cosines between centroids (mean embeddings of the papers of a community,
author, keyword, institution or funder). Communities are the S3 communities of the whole
run, so community-level tasks know the final partition (stated in each answer).
"""
from __future__ import annotations

from collections import Counter, defaultdict
from itertools import combinations
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _years(res) -> np.ndarray:
    return res.papers["year"].to_numpy(dtype=float)


def _split(res, test_years: int) -> Tuple[int, int]:
    ymax = int(np.nanmax(_years(res)))
    return ymax - test_years, ymax


def _centroids(E: np.ndarray, groups: Dict, rows_ok: Optional[np.ndarray] = None) -> Dict:
    out = {}
    for g, rows in groups.items():
        rows = [r for r in rows if rows_ok is None or rows_ok[r]]
        if rows:
            v = E[rows].mean(axis=0)
            out[g] = v / (np.linalg.norm(v) or 1.0)
    return out


def _rank_mean(*scores: np.ndarray) -> np.ndarray:
    from scipy.stats import rankdata
    return np.mean([rankdata(s) for s in scores], axis=0)


def evaluate(scores: Dict[str, np.ndarray], y: np.ndarray, k: int = 20, seed: int = 42) -> pd.DataFrame:
    from sklearn.metrics import average_precision_score, roc_auc_score
    rng = np.random.default_rng(seed)
    rows = []
    for name, s in list(scores.items()) + [("Random", rng.random(len(y)))]:
        s = np.asarray(s, dtype=float) + rng.random(len(y)) * 1e-9  # break ties reproducibly
        top = np.argsort(-s)[:k]
        boot = []
        for _ in range(200):  # 95% interval over resampled candidates
            ix = rng.integers(0, len(y), len(y))
            if 0 < y[ix].sum() < len(ix):
                boot.append(roc_auc_score(y[ix], s[ix]))
        lo, hi = (np.percentile(boot, [2.5, 97.5]) if boot else (np.nan, np.nan))
        rows.append({"method": name, "ROC AUC": round(float(roc_auc_score(y, s)), 3),
                     "AUC 95% CI": f"{lo:.2f}-{hi:.2f}",
                     "average precision": round(float(average_precision_score(y, s)), 3),
                     f"precision@{k}": round(float(y[top].mean()), 3)})
    return pd.DataFrame(rows)


def _membership(res, kind: str) -> Dict[str, List[int]]:
    """entity -> paper rows, for author / keyword / institution / funder."""
    from .networks import _memberships
    g: Dict[str, List[int]] = defaultdict(list)
    for i, name, _ in _memberships(res, kind):
        g[name].append(int(i))
    return g


def _main_comms(res, n: int) -> List[int]:
    return [int(c) for c in res.communities()["community"].head(n)]


def _comm_rows(res, comms: Sequence[int]) -> Dict[int, List[int]]:
    c = res.papers["community"]
    return {k: list(np.where((c == k).fillna(False).to_numpy())[0]) for k in comms}


def _result(name: str, table: pd.DataFrame, n_pos: int, n_cand: int, split: int, ymax: int,
            forecast: pd.DataFrame, note: str = "") -> Dict:
    return {"task": name, "table": table, "positives": int(n_pos), "candidates": int(n_cand),
            "split": split, "test": f"{split + 1}-{ymax}", "forecast": forecast, "note": note}


# --------------------------------------------------------------------------
# Q25 knowledge gaps that close: community pairs that start citing each other
# --------------------------------------------------------------------------
def gap_closing(res, test_years: int = 5, n_comms: int = 30, min_new: int = 2) -> Optional[Dict]:
    split, ymax = _split(res, test_years)
    yr = _years(res)
    comms = _main_comms(res, n_comms)
    if len(comms) < 4 or not res.citation_pairs:
        return None
    of = {i: int(c) for i, c in res.papers["community"].items() if pd.notna(c) and int(c) in comms}
    rows = _comm_rows(res, comms)
    E = res.embeddings
    pre_ok = yr <= split
    cen_pre = _centroids(E, rows, pre_ok)
    pre, new, total = Counter(), Counter(), Counter()
    for a, b in res.citation_pairs:
        ca, cb = of.get(a), of.get(b)
        if ca is None or cb is None or ca == cb:
            continue
        key = tuple(sorted((ca, cb)))
        total[key] += 1
        if yr[a] <= split:
            pre[key] += 1
        elif yr[a] <= ymax:
            new[key] += 1
    G = res.graph()
    links_pre = Counter()
    for u, v in G.edges():
        cu, cv = of.get(u), of.get(v)
        if cu is not None and cv is not None and cu != cv and yr[u] <= split and yr[v] <= split:
            links_pre[tuple(sorted((cu, cv)))] += 1
    size_pre = {c: int(sum(pre_ok[r] for r in rows[c])) for c in comms}
    cand = [p for p in combinations(sorted(comms), 2) if pre[p] <= 1 and p[0] in cen_pre and p[1] in cen_pre]
    if len(cand) < 10:
        return None
    y = np.array([int(new[p] >= min_new) for p in cand])
    if y.sum() < 3 or y.sum() == len(y):
        return None
    sim = np.array([float(cen_pre[a] @ cen_pre[b]) for a, b in cand])
    lk = np.array([links_pre[p] for p in cand], dtype=float)
    size = np.array([size_pre[a] * size_pre[b] for a, b in cand], dtype=float)
    table = evaluate({"S3 centroid similarity": sim, "S3 links between the communities": lk,
                      "Community sizes (baseline)": size, "S3 similarity + sizes": _rank_mean(sim, size)}, y)
    # forecast: pairs that still barely cite each other, using all years
    cen = _centroids(E, rows)
    open_pairs = [p for p in combinations(sorted(comms), 2) if total[p] <= 1]
    f = pd.DataFrame([{"community A": a, "community B": b, "S3 similarity": round(float(cen[a] @ cen[b]), 3),
                       "citations so far": total[(a, b)]} for a, b in open_pairs])
    f = f.sort_values("S3 similarity", ascending=False).head(15) if len(f) else f
    return _result("Community pairs that start citing each other", table, y.sum(), len(cand), split, ymax, f,
                   f"Candidates: pairs of the {len(comms)} largest communities with at most 1 citation between "
                   f"them up to {split}; a gap 'closes' if papers of {split + 1}-{ymax} make ≥{min_new} citations "
                   "between them.")


# --------------------------------------------------------------------------
# Q26 concepts that enter communities
# --------------------------------------------------------------------------
def concept_entry(res, test_years: int = 5, n_comms: int = 15, min_papers: int = 5) -> Optional[Dict]:
    split, ymax = _split(res, test_years)
    yr = _years(res)
    comms = _main_comms(res, n_comms)
    kw = _membership(res, "keyword")
    if not kw or len(comms) < 3:
        return None
    E = res.embeddings
    pre_ok = yr <= split
    rows = _comm_rows(res, comms)
    cen_c = _centroids(E, rows, pre_ok)
    of = {i: int(c) for i, c in res.papers["community"].items() if pd.notna(c)}
    n = len(res.papers)
    kw = {k: v for k, v in kw.items() if sum(pre_ok[r] for r in v) >= min_papers and len(v) < 0.1 * n}
    cen_k = _centroids(E, kw, pre_ok)
    size_pre = {c: sum(pre_ok[r] for r in rows[c]) for c in comms}
    cand, y, sim, pop = [], [], [], []
    for k, papers in kw.items():
        pre_c = {of.get(r) for r in papers if pre_ok[r]}
        new_c = {of.get(r) for r in papers if split < yr[r] <= ymax}
        for c in comms:
            if c in pre_c or c not in cen_c or k not in cen_k:
                continue
            cand.append((k, c))
            y.append(int(c in new_c))
            sim.append(float(cen_k[k] @ cen_c[c]))
            pop.append(sum(pre_ok[r] for r in papers) * size_pre[c])
    y = np.array(y)
    if len(cand) < 20 or y.sum() < 5:
        return None
    sim, pop = np.array(sim), np.array(pop, dtype=float)
    table = evaluate({"S3 similarity (keyword ↔ community)": sim, "Keyword frequency × community size (baseline)": pop,
                      "S3 similarity + frequency": _rank_mean(sim, pop)}, y)
    # forecast with all years
    cen_c_all, cen_k_all = _centroids(E, rows), _centroids(E, kw)
    rows_f = []
    for k, papers in kw.items():
        have = {of.get(r) for r in papers}
        for c in comms:
            if c not in have and k in cen_k_all and c in cen_c_all:
                rows_f.append({"keyword": k, "community": c, "papers with keyword": len(papers),
                               "S3 similarity": round(float(cen_k_all[k] @ cen_c_all[c]), 3)})
    f = pd.DataFrame(rows_f)
    f = f.sort_values("S3 similarity", ascending=False).head(15) if len(f) else f
    return _result("Concepts that enter communities", table, y.sum(), len(cand), split, ymax, f,
                   f"Candidates: author keywords with ≥{min_papers} papers up to {split} (leaving out near-universal "
                   f"search terms) × the {len(comms)} largest communities they had not appeared in; positive if a "
                   f"paper of that community uses the keyword in {split + 1}-{ymax}.")


# --------------------------------------------------------------------------
# Q27 / Q29 entities (authors, institutions, funders) that move into communities
# --------------------------------------------------------------------------
def entity_entry(res, kind: str = "author", test_years: int = 5, n_comms: int = 15,
                 min_papers: int = 2) -> Optional[Dict]:
    split, ymax = _split(res, test_years)
    yr = _years(res)
    comms = _main_comms(res, n_comms)
    groups = _membership(res, kind)
    if not groups or len(comms) < 3:
        return None
    E = res.embeddings
    pre_ok = yr <= split
    rows = _comm_rows(res, comms)
    cen_c = _centroids(E, rows, pre_ok)
    of = {i: int(c) for i, c in res.papers["community"].items() if pd.notna(c)}
    size_pre = {c: sum(pre_ok[r] for r in rows[c]) for c in comms}
    act = {g: v for g, v in groups.items()
           if sum(pre_ok[r] for r in v) >= min_papers and any(split < yr[r] <= ymax and of.get(r) in comms for r in v)}
    cen_g = _centroids(E, act, pre_ok)
    cand, y, sim, size = [], [], [], []
    for g, papers in act.items():
        pre_c = {of.get(r) for r in papers if pre_ok[r]}
        new_c = {of.get(r) for r in papers if split < yr[r] <= ymax}
        for c in comms:
            if c in pre_c or c not in cen_c:
                continue
            cand.append((g, c))
            y.append(int(c in new_c))
            sim.append(float(cen_g[g] @ cen_c[c]))
            size.append(size_pre[c])
    y = np.array(y)
    if len(cand) < 20 or y.sum() < 5:
        return None
    sim, size = np.array(sim), np.array(size, dtype=float)
    table = evaluate({f"S3 similarity ({kind} ↔ community)": sim, "Community size (baseline)": size,
                      "S3 similarity + size": _rank_mean(sim, size)}, y)
    # forecast: active entities (≥ min_papers overall) and communities they are not in yet
    groups_f = {g: v for g, v in groups.items() if len(v) >= min_papers}
    cen_all, cen_c_all = _centroids(E, groups_f), _centroids(E, rows)
    rows_f = []
    for g, papers in groups_f.items():
        have = {of.get(r) for r in papers}
        best = sorted(((float(cen_all[g] @ cen_c_all[c]), c) for c in comms if c not in have and c in cen_c_all),
                      reverse=True)[:1]
        for s, c in best:
            rows_f.append({kind: g, "papers": len(papers), "next community": c, "S3 similarity": round(s, 3)})
    f = pd.DataFrame(rows_f)
    f = f.sort_values("S3 similarity", ascending=False).head(15) if len(f) else f
    return _result(f"{kind.capitalize()}s that move into new communities", table, y.sum(), len(cand), split, ymax, f,
                   f"Candidates: {kind}s with ≥{min_papers} papers up to {split} that publish in {split + 1}-{ymax}, × the "
                   f"{len(comms)} largest communities they had not published in; positive if they publish there "
                   f"in {split + 1}-{ymax}.")


# --------------------------------------------------------------------------
# Q28 future co-authorship
# --------------------------------------------------------------------------
def coauthorship(res, test_years: int = 5, min_papers: int = 2, max_authors: int = 1500) -> Optional[Dict]:
    split, ymax = _split(res, test_years)
    yr = _years(res)
    groups = _membership(res, "author")
    if not groups:
        return None
    E = res.embeddings
    pre_ok = yr <= split
    from .questions import _authors
    pre_g, new_g = defaultdict(set), defaultdict(set)
    for i, v in res.papers["authors"].items():
        au = list(dict.fromkeys(_authors(v)))
        tgt = pre_g if pre_ok[i] else new_g if yr[i] <= ymax else None
        if tgt is None:
            continue
        for a, b in combinations(au, 2):
            tgt[a].add(b)
            tgt[b].add(a)
    active = [g for g, v in groups.items()
              if sum(pre_ok[r] for r in v) >= min_papers and any(split < yr[r] <= ymax for r in v)]
    active = sorted(active, key=lambda g: -sum(pre_ok[r] for r in groups[g]))[:max_authors]
    if len(active) < 10:
        return None
    cen = _centroids(E, {g: groups[g] for g in active}, pre_ok)
    deg = {g: len(pre_g[g]) for g in pre_g}
    cand, y, sim, aa, cn = [], [], [], [], []
    for a, b in combinations(sorted(active), 2):
        if b in pre_g[a]:
            continue
        common = pre_g[a] & pre_g[b]
        cand.append((a, b))
        y.append(int(b in new_g[a]))
        sim.append(float(cen[a] @ cen[b]))
        aa.append(sum(1 / np.log(deg[c]) for c in common if deg.get(c, 0) > 1))
        cn.append(len(common))
    y = np.array(y)
    if y.sum() < 5:
        return None
    sim, aa, cn = np.array(sim), np.array(aa), np.array(cn, dtype=float)
    table = evaluate({"S3 similarity (author centroids)": sim, "Adamic-Adar (co-author network)": aa,
                      "Common co-authors": cn, "S3 similarity + Adamic-Adar": _rank_mean(sim, aa)}, y, k=50)
    # forecast: pairs of active authors (all years) who never co-authored
    groups_f = {g: v for g, v in groups.items() if len(v) >= max(3, min_papers)}
    top = sorted(groups_f, key=lambda g: -len(groups_f[g]))[:400]
    cen_f = _centroids(E, {g: groups_f[g] for g in top})
    all_g = defaultdict(set)
    for d in (pre_g, new_g):
        for a, s in d.items():
            all_g[a] |= s
    rows_f = [{"author A": a, "author B": b, "S3 similarity": round(float(cen_f[a] @ cen_f[b]), 3),
               "common co-authors": len(all_g[a] & all_g[b])}
              for a, b in combinations(sorted(top), 2) if b not in all_g[a]]
    f = pd.DataFrame(rows_f)
    f = f.sort_values("S3 similarity", ascending=False).head(15) if len(f) else f
    return _result("Future co-authorships", table, y.sum(), len(cand), split, ymax, f,
                   f"Candidates: pairs of authors with ≥{min_papers} papers up to {split} who both publish in "
                   f"{split + 1}-{ymax} and had not co-authored; positive if they co-author in {split + 1}-{ymax}. "
                   "Scopus abbreviates names, so different people can share a name.")
