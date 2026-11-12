"""Q24: does S3 similarity predict future citations? (temporal link prediction)

Split the corpus in time. Papers published in the last `test_years` years are the "new"
papers; every paper published up to the split year is a candidate. For each new paper that
cites at least one earlier corpus paper, rank all candidates and check where the papers it
really cites end up.

Scores (higher = more likely to be cited):
  S3 similarity            text only: cosine of the two papers' embeddings
  popularity               in-corpus citations the candidate received up to the split year
                           (preferential attachment)
  S3 + popularity          logistic regression on both, fitted on an earlier split
                           (citations made in the previous window), never on the test years
  bibliographic coupling   shared references with the new paper; NOT a forecast, because it
                           needs the new paper's own reference list (shown for comparison)
  random                   chance level

Metrics: ROC AUC over all (new paper, candidate) pairs; mean average precision over new
papers; recall@10 = share of a new paper's in-corpus citations among its top 10 candidates.
"""
from __future__ import annotations

from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def _ref_matrix(res, rows: np.ndarray):
    """Sparse paper x reference matrix (references normalised to 80 characters)."""
    from scipy.sparse import csr_matrix
    from .pipeline import _norm_text
    from .questions import _split_refs
    vocab: Dict[str, int] = {}
    r_idx, c_idx = [], []
    for k, i in enumerate(rows):
        for ref in set(_norm_text(x)[:80] for x in _split_refs(res.papers.at[i, "references"])):
            if len(ref) > 20:
                c_idx.append(vocab.setdefault(ref, len(vocab)))
                r_idx.append(k)
    return csr_matrix((np.ones(len(r_idx)), (r_idx, c_idx)), shape=(len(rows), max(1, len(vocab))))


def _metrics(score: np.ndarray, truth: np.ndarray, k: int = 10) -> Dict[str, float]:
    """score, truth: (new papers x candidates). Ties are broken at random (seeded upstream)."""
    from sklearn.metrics import roc_auc_score
    auc = float(roc_auc_score(truth.ravel(), score.ravel()))
    order = np.argsort(-score, axis=1, kind="stable")
    hit = np.take_along_axis(truth, order, axis=1)
    npos = hit.sum(axis=1)
    rank = np.arange(1, hit.shape[1] + 1)
    ap = ((np.cumsum(hit, axis=1) / rank) * hit).sum(axis=1) / np.maximum(npos, 1)
    return {"ROC AUC": round(auc, 3), "MAP": round(float(ap.mean()), 3),
            f"recall@{k}": round(float((hit[:, :k].sum(axis=1) / npos).mean()), 3),
            f"precision@{k}": round(float((hit[:, :k].sum(axis=1) / k).mean()), 4),
            "hit@1": round(float(hit[:, 0].mean()), 3)}


def _setup(res, split: float, horizon: float):
    """New papers (split < year <= split + horizon) that cite >= 1 candidate (year <= split)."""
    years = res.papers["year"].to_numpy(dtype=float)
    old = years <= split
    new = (years > split) & (years <= split + horizon)
    pos = [(i, j) for i, j in res.citation_pairs if new[i] and old[j]]
    citing = np.array(sorted({i for i, _ in pos}), dtype=int)
    cand = np.where(old)[0]
    col = {j: k for k, j in enumerate(cand)}
    row = {i: k for k, i in enumerate(citing)}
    truth = np.zeros((len(citing), len(cand)), dtype=np.int8)
    for i, j in pos:
        truth[row[i], col[j]] = 1
    pop = Counter(j for i, j in res.citation_pairs if old[i] and old[j])
    popularity = np.log1p(np.array([pop.get(j, 0) for j in cand], dtype=float))
    return citing, cand, truth, popularity


def citation_prediction(res, test_years: int = 5, k: int = 10) -> Optional[Dict[str, object]]:
    from sklearn.linear_model import LogisticRegression
    years = res.papers["year"].to_numpy(dtype=float)
    if np.isnan(years).all():
        return None
    ymax = int(np.nanmax(years))
    split = ymax - test_years
    E = res.embeddings
    rng = np.random.default_rng(res.config.seed)

    citing, cand, truth, pop = _setup(res, split, test_years)
    if len(citing) < 10 or truth.sum() < 20:
        return None
    s3 = E[citing] @ E[cand].T
    pop_m = np.broadcast_to(pop, s3.shape)

    # fit the combination on the previous window only (no test-year information)
    v_citing, v_cand, v_truth, v_pop = _setup(res, split - test_years, test_years)
    combo = None
    if len(v_citing) >= 10 and v_truth.sum() >= 20:
        v_s3 = E[v_citing] @ E[v_cand].T
        P_i, P_j = np.where(v_truth == 1)
        neg = rng.integers(0, v_truth.size, size=min(v_truth.size, 30 * len(P_i)))
        N_i, N_j = np.unravel_index(neg, v_truth.shape)
        keep = v_truth[N_i, N_j] == 0
        X = np.vstack([np.c_[v_s3[P_i, P_j], v_pop[P_j]], np.c_[v_s3[N_i[keep], N_j[keep]], v_pop[N_j[keep]]]])
        y = np.r_[np.ones(len(P_i)), np.zeros(int(keep.sum()))]
        lr = LogisticRegression(max_iter=1000).fit(X, y)
        combo = lr.coef_[0, 0] * s3 + lr.coef_[0, 1] * pop_m
        coef = {"S3": round(float(lr.coef_[0, 0]), 2), "popularity": round(float(lr.coef_[0, 1]), 2)}
    else:
        coef = None

    jitter = rng.random(s3.shape) * 1e-9  # break ties (popularity, coupling) reproducibly
    methods: List[Tuple[str, np.ndarray, str]] = [
        ("S3 similarity (text only)", s3, "forecast"),
        ("Popularity (citations up to split year)", pop_m + jitter, "forecast"),
    ]
    if combo is not None:
        methods.append(("S3 + popularity (fitted on earlier years)", combo, "forecast"))
    if res.papers["references"].notna().any():
        R = _ref_matrix(res, np.r_[citing, cand])
        bc = (R[: len(citing)] @ R[len(citing):].T).toarray()
        methods.append(("Bibliographic coupling (needs the new paper's references)", bc + jitter, "reference point"))
    methods.append(("Random", rng.random(s3.shape), "baseline"))

    rows = [{"method": m, "type": t, **_metrics(sc, truth, k)} for m, sc, t in methods]
    table = pd.DataFrame(rows)

    # how far down the S3 ranking do real citations sit? recall@k curve
    ks = [1, 2, 5, 10, 20, 50, 100, 200]
    ks = [x for x in ks if x <= len(cand)]
    curves = {}
    for m, sc, t in methods:
        order = np.argsort(-sc, axis=1, kind="stable")
        hit = np.take_along_axis(truth, order, axis=1)
        npos = hit.sum(axis=1)
        curves[m] = [round(float((hit[:, :x].sum(axis=1) / npos).mean()), 3) for x in ks]

    # S3 links above the threshold: what share became citations?
    t = res.best["threshold"] if res.best else None
    linked = None
    if t is not None:
        sim_pairs = s3 >= t
        if sim_pairs.any():
            linked = {"pairs": int(sim_pairs.sum()), "cited": int((truth.astype(bool) & sim_pairs).sum()),
                      "threshold": float(t)}
    return {"split": split, "test": f"{split + 1}-{ymax}", "new_papers": int(len(citing)),
            "candidates": int(len(cand)), "citations": int(truth.sum()),
            "base_rate": float(truth.mean()), "table": table, "ks": ks, "curves": curves,
            "coef": coef, "linked": linked, "k": k}
