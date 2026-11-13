"""Projection checks for two-mode data: significant ties and four-cycle clustering.

Coupling, co-citation and co-authorship are one-mode projections of two-mode networks
(papers-references, cited-citing papers, papers-authors). Projections inflate ties and
clustering: one popular reference links every paper that cites it, and a paper with ten
authors makes a clique of ten (Jankowski et al. 2026, Commun. Phys. 9:35; Neal 2014).

backbone()             keeps only ties where two papers share more items than chance predicts
                       from how many items each paper has and how popular the items are
                       (stochastic degree sequence model, Poisson approximation with Chung-Lu
                       probabilities), with Benjamini-Hochberg false-discovery control over all
                       paper pairs.
bipartite_clustering() clustering measured on the two-mode network itself, with four-cycles
                       (Robins & Alexander 2004), to compare with the clustering of the projection.
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import scipy.sparse as sp


def _binary(B) -> sp.csr_matrix:
    B = sp.csr_matrix(B, dtype=np.float32)
    B.data[:] = 1.0
    B.eliminate_zeros()
    return B


def shared(B) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(row i, row j, number of shared items) for every pair i<j that shares at least one item."""
    B = _binary(B)
    S = sp.triu(B @ B.T, k=1).tocoo()
    keep = S.data > 0
    return S.row[keep].astype(np.int64), S.col[keep].astype(np.int64), np.rint(S.data[keep]).astype(np.int64)


def pvalues(B) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(i, j, shared, p): p = P(Poisson(λ_ij) ≥ shared), where λ_ij = Σ_r p_ir p_jr is the
    expected number of shared items if row i took item r with probability
    p_ir = min(1, k_i d_r / E) (k = items per row, d = rows per item, E = all row-item ties;
    Chung-Lu probabilities, which keep every row's and item's degree on average)."""
    from scipy.stats import poisson
    B = _binary(B)
    k = np.asarray(B.sum(axis=1)).ravel().astype(float)
    d = np.asarray(B.sum(axis=0)).ravel().astype(float)
    E = d.sum()
    i, j, s = shared(B)
    if E == 0 or s.size == 0:
        return i, j, s, np.ones(s.size)
    # λ depends only on the two row degrees: compute it once per pair of distinct degrees
    ku, kinv = np.unique(k, return_inverse=True)
    du, dcount = np.unique(d[d > 0], return_counts=True)
    F = np.minimum(1.0, np.outer(ku, du) / E)            # distinct k x distinct d
    L = (F * dcount) @ F.T                               # λ for each pair of distinct degrees
    lam = L[kinv[i], kinv[j]]
    return i, j, s, poisson.sf(s - 1, lam)


def backbone(B, alpha: float = 0.05) -> Dict[str, np.ndarray]:
    """Significant ties of the projection of B (rows = papers): pair keys i*n+j (i<j) and their
    p-values. Benjamini-Hochberg over all n(n-1)/2 pairs (pairs sharing nothing have p = 1)."""
    n = B.shape[0]
    i, j, s, p = pvalues(B)
    m = n * (n - 1) / 2
    order = np.argsort(p, kind="stable")
    ok = p[order] <= alpha * np.arange(1, p.size + 1) / m
    r = int(np.nonzero(ok)[0].max()) + 1 if ok.any() else 0
    keep = np.sort(order[:r])
    return {"keys": np.unique(i[keep] * n + j[keep]), "p": p[keep], "tested": int(p.size)}


def bipartite_clustering(B) -> float:
    """Four-cycle clustering of the two-mode network: 4 × squares / paths of length 3.
    A square = two rows sharing two items; 1 = every 3-path closes into a square."""
    B = _binary(B)
    _, _, s = shared(B)
    squares = float((s * (s - 1) / 2).sum())
    k = np.asarray(B.sum(axis=1)).ravel()
    d = np.asarray(B.sum(axis=0)).ravel()
    C = B.tocoo()
    paths3 = float(((k[C.row] - 1) * (d[C.col] - 1)).sum())
    return 4 * squares / paths3 if paths3 else float("nan")


def projection_clustering(keys: np.ndarray, n: int) -> float:
    """Global clustering (transitivity) of a one-mode layer given as pair keys i*n+j."""
    if keys.size == 0:
        return float("nan")
    i, j = keys // n, keys % n
    A = sp.csr_matrix((np.ones(keys.size, dtype=np.float32), (i, j)), shape=(n, n))
    A = (A + A.T).tocsr()
    deg = np.asarray(A.sum(axis=1)).ravel()
    triples = float((deg * (deg - 1)).sum())
    closed = float((A @ A).multiply(A).sum())  # 6 × triangles
    return closed / triples if triples else float("nan")
