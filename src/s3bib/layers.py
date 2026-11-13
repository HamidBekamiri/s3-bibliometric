"""Multiplex view of a corpus: the same papers tied in several ways at once.

Layers (undirected paper pairs):
  text          S3 similarity: the links of the text map (or, when the map is a hybrid,
                the same number of most similar pairs by S3 alone)
  coupling      the two papers share at least one reference
  co-citation   the two papers are cited together by at least one corpus paper
  direct        one paper cites the other inside the corpus
  co-authorship the two papers share at least one author

Used by three questions:
  Q35  layer overlap (Jaccard, and how much more than chance)
  Q36  homophily (attribute assortativity per layer: community, journal, country, funder)
  Q37  direction (PageRank on the citation network, one-way vs two-way community flows)
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp

LAYERS = ("text", "coupling", "co-citation", "direct citation", "co-authorship")


# ---------------------------------------------------------------- layers --
def _pairs(M) -> np.ndarray:
    """Upper-triangle nonzero pairs of a square sparse matrix as an int64 key i*n+j."""
    M = sp.triu(sp.csr_matrix(M), k=1).tocoo()
    keep = M.data > 0
    n = M.shape[0]
    return np.unique(M.row[keep].astype(np.int64) * n + M.col[keep].astype(np.int64))


def _key(pairs: Sequence[Tuple[int, int]], n: int) -> np.ndarray:
    if not len(pairs):
        return np.zeros(0, dtype=np.int64)
    a = np.asarray(pairs, dtype=np.int64)
    lo, hi = a.min(axis=1), a.max(axis=1)
    k = lo * n + hi
    return np.unique(k[lo != hi])


def _text_layer(res) -> np.ndarray:
    n = len(res.papers)
    G = res.graph()
    if not res.config.map_weights:
        return _key(list(G.edges()), n)
    k = G.number_of_edges()  # hybrid map: as many S3 pairs as the map has links
    S = res.embeddings @ res.embeddings.T
    iu = np.triu_indices(n, 1)
    v = S[iu]
    if k <= 0:
        return np.zeros(0, dtype=np.int64)
    top = np.argpartition(-v, min(k, v.size) - 1)[:k]
    return np.unique(iu[0][top].astype(np.int64) * n + iu[1][top])


def _author_matrix(res):
    from .questions import _authors
    vocab: Dict[str, int] = {}
    r, c = [], []
    for i, v in enumerate(res.papers["authors"].tolist()):
        for a in set(x.strip().lower() for x in _authors(v)):
            c.append(vocab.setdefault(a, len(vocab)))
            r.append(i)
    return sp.csr_matrix((np.ones(len(r), dtype=np.float32), (r, c)),
                         shape=(len(res.papers), max(1, len(vocab))))


def layers(res) -> Dict[str, np.ndarray]:
    """{layer name: sorted unique pair keys i*n+j (i<j)}; empty layers are left out. Cached."""
    if getattr(res, "_layers", None) is not None:
        return res._layers
    from .hybrid import citation_matrix, reference_matrix
    P = res.papers
    n = len(P)
    out: Dict[str, np.ndarray] = {"text": _text_layer(res)}
    if "references" in P and P["references"].notna().any():
        R = reference_matrix(P)
        out["coupling"] = _pairs(R @ R.T)
    if res.citation_pairs:
        C = citation_matrix(n, res.citation_pairs)
        out["co-citation"] = _pairs(C.T @ C)
        out["direct citation"] = _key(res.citation_pairs, n)
    if "authors" in P and P["authors"].notna().any():
        A = _author_matrix(res)
        out["co-authorship"] = _pairs(A @ A.T)
    res._layers = {k: v for k, v in out.items() if v.size}
    return res._layers


# --------------------------------------------------------------- overlap --
def overlap(res) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(Jaccard matrix, lift matrix, per-layer summary).
    lift[a, b] = P(tie in b | tie in a) / density(b): how many times more often a tie of
    layer a is also a tie of layer b than a random pair is (1 = chance)."""
    L = layers(res)
    n = len(res.papers)
    pairs_total = n * (n - 1) / 2
    names = list(L)
    J = pd.DataFrame(1.0, index=names, columns=names)
    lift = pd.DataFrame(np.nan, index=names, columns=names)
    for a in names:
        for b in names:
            if a == b:
                continue
            inter = np.intersect1d(L[a], L[b], assume_unique=True).size
            union = L[a].size + L[b].size - inter
            J.loc[a, b] = inter / union if union else 0.0
            dens_b = L[b].size / pairs_total
            lift.loc[a, b] = (inter / L[a].size) / dens_b if L[a].size and dens_b else np.nan
    rows = []
    t = L.get("text", np.zeros(0, dtype=np.int64))
    for k, v in L.items():
        in_text = np.intersect1d(v, t, assume_unique=True).size if k != "text" else v.size
        rows.append({"layer": k, "ties": int(v.size), "density": v.size / pairs_total,
                     "papers with a tie": int(np.unique(np.concatenate([v // n, v % n])).size),
                     "share also a text link": in_text / v.size if v.size else np.nan})
    return J, lift, pd.DataFrame(rows)


# ------------------------------------------------------------- homophily --
def paper_attributes(res) -> pd.DataFrame:
    """One categorical value per paper: community, journal, country, funder (NaN if unknown).
    Country = the most frequent country among a paper's affiliations; funder = the first funder."""
    from .questions import _funders, _main_comms, _paper_affils
    P = res.papers
    out = pd.DataFrame(index=P.index)
    top = set(_main_comms(res, res.config.top_n_communities))
    out["community"] = [int(c) if pd.notna(c) and int(c) in top else np.nan for c in P["community"]]
    if "source" in P:
        s = P["source"].astype("string").str.strip().str.lower()
        out["journal"] = s.where(s.notna() & (s != "") & (s != "nan"))
    if "affiliations" in P and P["affiliations"].notna().any():
        A = _paper_affils(res)
        A = A[A["country"].astype(str).str.len() > 1]
        if len(A):
            out["country"] = A.groupby("paper")["country"].agg(lambda s: s.value_counts().index[0])
    if "funding" in P and P["funding"].notna().any():
        f = {i: fs[0] for i, v in P["funding"].items() if (fs := _funders(v))}
        if f:
            out["funder"] = pd.Series(f)
    return out.dropna(axis=1, how="all")


def assortativity(keys: np.ndarray, n: int, values: pd.Series) -> Dict[str, float]:
    """Newman's attribute assortativity r for an undirected layer, on ties whose two papers
    both have a value; plus the observed and chance-expected share of within-group ties."""
    v = values.reindex(range(n))
    codes, _ = pd.factorize(v)  # NaN -> -1
    u, w = keys // n, keys % n
    cu, cw = codes[u], codes[w]
    ok = (cu >= 0) & (cw >= 0)
    m = int(ok.sum())
    if m == 0:
        return {"ties": 0, "within": np.nan, "expected": np.nan, "r": np.nan}
    cu, cw = cu[ok], cw[ok]
    within = float((cu == cw).mean())
    a = np.bincount(np.concatenate([cu, cw])) / (2 * m)
    exp = float((a ** 2).sum())
    r = (within - exp) / (1 - exp) if exp < 1 else np.nan
    return {"ties": m, "within": within, "expected": exp, "r": r}


def homophily(res) -> pd.DataFrame:
    """Long table: attribute, layer, ties, within, expected, r."""
    L = layers(res)
    n = len(res.papers)
    A = paper_attributes(res)
    rows = []
    for att in A.columns:
        if A[att].nunique() < 2:
            continue
        for name, keys in L.items():
            s = assortativity(keys, n, A[att])
            rows.append({"attribute": att, "layer": name, **s})
    return pd.DataFrame(rows)


# ------------------------------------------------------------- direction --
def pagerank(res, alpha: float = 0.85) -> pd.Series:
    """PageRank on the in-corpus citation network (citing -> cited): a paper ranks high when
    it is cited by papers that are themselves cited."""
    import networkx as nx
    G = nx.DiGraph()
    G.add_nodes_from(range(len(res.papers)))
    G.add_edges_from(res.citation_pairs)
    return pd.Series(nx.pagerank(G, alpha=alpha), name="pagerank").sort_index()


def flow_direction(res) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float]]:
    """(per community, per community pair, totals) from the citation matrix of the top
    communities (rows cite columns), within-community citations left out.

    source score = (cited by other communities - citing other communities) / (sum of both):
      +1 = only cited (a pure knowledge source), -1 = only citing (builds on others).
    pair asymmetry = |A->B - B->A| / (A->B + B->A).
    reciprocity = sum over pairs of 2·min(A->B, B->A) / sum of all flows between communities
      (1 = every flow is returned in full; 0 = all one-way). 'expected' is the same measure
      if each citation between A and B went either way with probability 1/2."""
    from scipy.stats import binom
    C = res.citation_matrix()
    ids = list(C.index)
    M = C.values.astype(float)
    np.fill_diagonal(M, 0)
    inward, outward = M.sum(axis=0), M.sum(axis=1)
    P = res.papers
    rows = []
    for k, c in enumerate(ids):
        tot = inward[k] + outward[k]
        yrs = P.loc[(P["community"] == c).fillna(False), "year"].dropna()
        rows.append({"community": int(c), "cited by other communities": int(inward[k]),
                     "citing other communities": int(outward[k]),
                     "source score": (inward[k] - outward[k]) / tot if tot else np.nan,
                     "median year": float(yrs.median()) if len(yrs) else np.nan})
    pairs, obs, exp, tot_all = [], 0.0, 0.0, 0.0
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            ab, ba = M[i, j], M[j, i]
            N = int(ab + ba)
            if N == 0:
                continue
            obs += 2 * min(ab, ba)
            x = np.arange(N + 1)
            exp += 2 * float((np.minimum(x, N - x) * binom.pmf(x, N, 0.5)).sum())
            tot_all += N
            pairs.append({"A": int(ids[i]), "B": int(ids[j]), "A cites B": int(ab), "B cites A": int(ba),
                          "total": N, "asymmetry": abs(ab - ba) / N,
                          "main direction": f"C{ids[i]} → C{ids[j]}" if ab > ba else
                          f"C{ids[j]} → C{ids[i]}" if ba > ab else "balanced"})
    totals = {"reciprocity": obs / tot_all if tot_all else np.nan,
              "expected": exp / tot_all if tot_all else np.nan, "flows": tot_all}
    return pd.DataFrame(rows), pd.DataFrame(pairs), totals
