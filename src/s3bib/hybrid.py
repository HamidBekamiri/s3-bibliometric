"""Hybrid maps: S3 similarity combined with citation-based relatedness.

The published S3 method builds the map from text similarity alone. A hybrid map adds the
classic bibliometric relations, each scaled to 0-1, as a weighted sum:

  s3          S3 similarity (cosine of the embeddings, negative values set to 0)
  coupling    bibliographic coupling: shared references (Salton cosine)
  direct      direct citation: 1 if one paper cites the other inside the corpus
  cocitation  co-citation: cited together by papers of the corpus (Salton cosine)

S3Config(map_weights="hybrid") uses 25% for each; pass a dict to choose your own weights,
e.g. {"s3": 0.5, "coupling": 0.5}. Weights are normalised to sum to 1.

Because most pairs have no citation relation, hybrid values are on a different scale than
S3 (a pair with S3 0.8 and no citation tie scores 0.2 at 25% weights). The threshold scan
therefore matches link density: each hybrid threshold gives the same number of links as
one S3 threshold (0.75-0.85) on the same corpus, and SII picks the best one, as for S3.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

COMPONENTS = ("s3", "coupling", "direct", "cocitation")
EQUAL = {c: 0.25 for c in COMPONENTS}


def resolve_weights(w: Union[None, str, bool, Dict[str, float]]) -> Optional[Dict[str, float]]:
    """None/"s3" -> None (pure S3 map); "hybrid"/True -> 25% each; dict -> normalised."""
    if w is None or w is False or (isinstance(w, str) and w.lower() == "s3"):
        return None
    if w is True or (isinstance(w, str) and w.lower() == "hybrid"):
        return dict(EQUAL)
    if isinstance(w, str):  # "s3=0.5,coupling=0.5"
        w = {k.strip(): float(v) for k, v in (p.split("=") for p in w.split(",") if p.strip())}
    bad = set(w) - set(COMPONENTS)
    if bad:
        raise ValueError(f"Unknown map components {sorted(bad)}; use {COMPONENTS}")
    tot = sum(max(0.0, float(v)) for v in w.values())
    if tot <= 0:
        raise ValueError("map_weights must contain a positive weight")
    out = {c: round(max(0.0, float(w.get(c, 0.0))) / tot, 6) for c in COMPONENTS}
    return None if out == {"s3": 1.0, "coupling": 0.0, "direct": 0.0, "cocitation": 0.0} else out


def reference_matrix(df: pd.DataFrame):
    """Sparse papers x cited works (references normalised to their first 80 characters)."""
    from scipy.sparse import csr_matrix
    from .pipeline import _norm_text
    from .questions import _split_refs
    vocab: Dict[str, int] = {}
    r, c = [], []
    for i, v in enumerate(df["references"].tolist()):
        for ref in set(_norm_text(x)[:80] for x in _split_refs(v)):
            if len(ref) > 20:
                c.append(vocab.setdefault(ref, len(vocab)))
                r.append(i)
    return csr_matrix((np.ones(len(r), dtype=np.float32), (r, c)), shape=(len(df), max(1, len(vocab))))


def _salton(M) -> np.ndarray:
    """Salton cosine of the rows of a sparse 0/1 matrix: |A∩B| / sqrt(|A||B|)."""
    deg = np.asarray(M.sum(axis=1)).ravel().astype(np.float32)
    S = (M @ M.T).toarray().astype(np.float32)
    norm = np.sqrt(np.outer(deg, deg))
    with np.errstate(divide="ignore", invalid="ignore"):
        S = np.where(norm > 0, S / norm, 0.0).astype(np.float32)
    np.fill_diagonal(S, 0.0)
    return S


def coupling_matrix(df: pd.DataFrame) -> np.ndarray:
    return _salton(reference_matrix(df))


def citation_matrix(n: int, pairs: Sequence[Tuple[int, int]]):
    from scipy.sparse import csr_matrix
    if not pairs:
        return csr_matrix((n, n), dtype=np.float32)
    i, j = zip(*pairs)
    return csr_matrix((np.ones(len(i), dtype=np.float32), (i, j)), shape=(n, n))


def direct_matrix(n: int, pairs: Sequence[Tuple[int, int]]) -> np.ndarray:
    D = np.zeros((n, n), dtype=np.float32)
    for a, b in pairs:
        D[a, b] = D[b, a] = 1.0
    return D


def cocitation_matrix(n: int, pairs: Sequence[Tuple[int, int]]) -> np.ndarray:
    """Papers cited together by the same corpus papers (columns of the citation matrix)."""
    C = citation_matrix(n, pairs)
    return _salton(C.T.tocsr())


def hybrid_matrix(emb: np.ndarray, df: pd.DataFrame, pairs: Sequence[Tuple[int, int]],
                  weights: Dict[str, float]) -> np.ndarray:
    n = len(df)
    H = np.zeros((n, n), dtype=np.float32)
    if weights.get("s3"):
        H += weights["s3"] * np.clip(emb @ emb.T, 0, None).astype(np.float32)
    if weights.get("coupling"):
        H += weights["coupling"] * coupling_matrix(df)
    if weights.get("direct"):
        H += weights["direct"] * direct_matrix(n, pairs)
    if weights.get("cocitation"):
        H += weights["cocitation"] * cocitation_matrix(n, pairs)
    np.fill_diagonal(H, 1.0)
    return H


def density_thresholds(H_dir: np.ndarray, link_counts: Sequence[int]) -> List[float]:
    """Thresholds that give the hybrid map the same numbers of links as the S3 scan."""
    iu = np.triu_indices(H_dir.shape[0], k=1)
    v = np.maximum(H_dir[iu], H_dir.T[iu])
    v = np.sort(v[v > 0])[::-1]
    if v.size == 0:
        return [0.5]
    ts = [float(v[min(max(k, 1), v.size) - 1]) for k in link_counts if k > 0]
    return sorted(set(round(t, 4) for t in ts)) or [float(v[0])]


def component_shares(H_parts: Dict[str, np.ndarray], edges: Sequence[Tuple[int, int]],
                     weights: Dict[str, float]) -> Dict[str, float]:
    """How much each component contributes to the links of the final map."""
    tot = {c: 0.0 for c in weights}
    for c, M in H_parts.items():
        tot[c] = float(sum(M[a, b] for a, b in edges)) * weights[c]
    s = sum(tot.values()) or 1.0
    return {c: round(v / s, 3) for c, v in tot.items()}
