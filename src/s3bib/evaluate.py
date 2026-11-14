"""The S3 paper's evaluation (Bekamiri, Østergaard & Timmermans 2027, Tables 5-7):
S3 communities against bibliographic coupling, BERTopic (if installed) and random clusters.

  TCI         Topic Coherence Index = 1 - JSD between each paper's TF-IDF word distribution
              and the mean distribution of its community (Jensen-Shannon, base 2), averaged
              over papers in a community, then over communities with more than two papers.
              TCI depends on community size, so every method also gets a random baseline with
              its own community sizes; the gain over that baseline is the fair comparison.
  silhouette  on the SBERT embeddings for every method (cosine), so all methods are judged
              in the same semantic space.
  QI          Quality Index: within-community citations per paper, averaged over each
              method's largest communities (S3Config.top_n_communities).
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd


def coupling_partition(res) -> np.ndarray:
    """Leiden (modularity) on the bibliographic coupling network (Salton cosine weights).
    -1 = paper without references or in a community of one."""
    import igraph as ig
    import leidenalg as la
    import scipy.sparse as sp
    from .hybrid import reference_matrix
    from .backbone import _binary
    R = _binary(reference_matrix(res.papers))
    deg = np.asarray(R.sum(axis=1)).ravel()
    S = sp.triu(R @ R.T, k=1).tocoo()
    w = S.data / np.sqrt(deg[S.row] * deg[S.col])
    n = len(res.papers)
    lab = np.full(n, -1)
    if not S.nnz:
        return lab
    g = ig.Graph(n=n, edges=list(zip(S.row.tolist(), S.col.tolist())))
    part = la.find_partition(g, la.ModularityVertexPartition, weights=w.tolist(), seed=res.config.seed)
    m = np.asarray(part.membership)
    sizes = np.bincount(m)
    keep = (sizes[m] >= 2) & (deg > 0)
    lab[keep] = m[keep]
    return lab


def bertopic_partition(res) -> Optional[np.ndarray]:
    """BERTopic on the same SBERT embeddings (UMAP + HDBSCAN, seeded). None if not installed."""
    try:
        from bertopic import BERTopic
        from umap import UMAP
    except ImportError:
        return None
    umap = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric="cosine", random_state=res.config.seed)
    model = BERTopic(umap_model=umap, calculate_probabilities=False, verbose=False)
    topics, _ = model.fit_transform(res.papers["text"].fillna("").tolist(), embeddings=res.embeddings)
    return np.asarray(topics)


def s3_partition(res) -> np.ndarray:
    c = res.papers["community"]
    lab = np.where(c.notna(), c.fillna(-1), -1).astype(int)
    sizes = pd.Series(lab[lab >= 0]).value_counts()
    small = set(sizes[sizes < 2].index)
    return np.array([-1 if (x < 0 or x in small) else x for x in lab])


def random_partition(labels: np.ndarray, seed: int) -> np.ndarray:
    """The same community sizes, papers assigned at random (among the clustered papers)."""
    out = labels.copy()
    idx = np.where(labels >= 0)[0]
    out[idx] = np.random.default_rng(seed).permutation(labels[idx])
    return out


def tfidf_matrix(df: pd.DataFrame):
    """L1-normalised TF-IDF word distributions of the papers (for TCI)."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import normalize
    X = TfidfVectorizer(stop_words="english", min_df=2, token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z-]+\b") \
        .fit_transform(df["text"].fillna(""))
    return normalize(X, norm="l1").tocsr()


def _tfidf(res):
    return tfidf_matrix(res.papers)


def tci(P, labels: np.ndarray) -> float:
    """Mean over communities (> 2 papers) of 1 - mean JSD(paper, community mean)."""
    vals = []
    for c in np.unique(labels[labels >= 0]):
        rows = np.where(labels == c)[0]
        if len(rows) <= 2:
            continue
        sub = P[rows]
        cols = np.unique(sub.indices)
        if cols.size == 0:
            continue
        A = sub[:, cols].toarray()
        A = A[A.sum(axis=1) > 0]
        if len(A) <= 2:
            continue
        q = A.mean(axis=0)
        M = (A + q) / 2
        with np.errstate(divide="ignore", invalid="ignore"):
            kp = np.where(A > 0, A * np.log2(A / M), 0.0).sum(axis=1)
            kq = np.where(q > 0, q * np.log2(q / M), 0.0).sum(axis=1)
        vals.append(1 - float(np.mean(0.5 * kp + 0.5 * kq)))
    return float(np.mean(vals)) if vals else float("nan")


def silhouette(emb: np.ndarray, labels: np.ndarray, seed: int, max_n: int = 10000) -> float:
    from sklearn.metrics import silhouette_score
    m = labels >= 0
    if len(set(labels[m])) < 2 or m.sum() <= len(set(labels[m])):
        return float("nan")
    return float(silhouette_score(emb[m], labels[m], metric="cosine",
                                  sample_size=min(int(m.sum()), max_n), random_state=seed))


def qi(pairs, labels: np.ndarray, top_n: int) -> float:
    """Mean QI (within-community citations / papers) over the top_n largest communities."""
    if not pairs:
        return float("nan")
    sizes = pd.Series(labels[labels >= 0]).value_counts().head(top_n)
    within = pd.Series(0, index=sizes.index, dtype=float)
    for i, j in pairs:
        a, b = labels[i], labels[j]
        if a >= 0 and a == b and a in within.index:
            within[a] += 1
    return float((within / sizes).mean())


def evaluate(res, bertopic: Optional[bool] = None, random_repeats: int = 5) -> Dict[str, object]:
    """Table of methods x (papers clustered, communities, TCI, silhouette, QI)."""
    P = _tfidf(res)
    seed = res.config.seed
    top = res.config.top_n_communities
    methods: Dict[str, np.ndarray] = {"S3": s3_partition(res)}
    note = ""
    if "references" in res.papers and res.papers["references"].notna().any():
        methods["Bibliographic coupling"] = coupling_partition(res)
    if bertopic is not False:
        bt = bertopic_partition(res)
        if bt is not None:
            methods["BERTopic"] = bt
        else:
            note = "BERTopic is not installed (pip install 's3-bibliometric[bertopic]'), so it is left out."
    rows = []
    for name, lab in methods.items():
        # TCI depends on community size, so each method is compared with random clusters of
        # its own sizes (the paper's random baseline per cluster size)
        rnd = [random_partition(lab, seed + k) for k in range(random_repeats)]
        t, t0 = tci(P, lab), float(np.nanmean([tci(P, r) for r in rnd]))
        rows.append({"method": name, "papers clustered": int((lab >= 0).sum()),
                     "communities": int(len(set(lab[lab >= 0]))), "TCI": t,
                     "TCI random (same sizes)": t0, "TCI gain over random": t - t0,
                     "silhouette": silhouette(res.embeddings, lab, seed),
                     "silhouette random (same sizes)": float(np.nanmean([silhouette(res.embeddings, r, seed)
                                                                         for r in rnd])),
                     "QI (top communities)": qi(res.citation_pairs, lab, top),
                     "QI random (same sizes)": float(np.nanmean([qi(res.citation_pairs, r, top) for r in rnd]))})
    return {"table": pd.DataFrame(rows), "note": note}
