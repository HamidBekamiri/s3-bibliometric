"""Hybrid maps: S3 similarity combined with citation-based relatedness.

The published S3 method builds the map from text similarity alone. A hybrid map adds the
classic bibliometric relations, each scaled to 0-1, as a weighted sum:

  s3          S3 similarity (cosine of the embeddings, negative values set to 0)
  coupling    bibliographic coupling: shared references (Salton cosine)
  direct      direct citation: 1 if one paper cites the other inside the corpus
  cocitation  co-citation: cited together by papers of the corpus (Salton cosine)

Choose one of four maps with S3Config(map_weights=...):
  "text"             text only (S3, the published method; the default)
  "text+coupling"    75% text + 25% bibliographic coupling
  "text+cocitation"  75% text + 25% co-citation
  "text+direct"      75% text + 25% direct citation
A dict sets custom weights (normalised to sum to 1); "hybrid" gives all four 25%.
S3Config(backbone=True) keeps only the significant coupling and co-citation ties
(s3bib.backbone): sharing one very popular reference no longer links two papers.

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
NAMES = {"s3": "text (S3)", "coupling": "bibliographic coupling", "cocitation": "co-citation",
         "direct": "direct citation"}

# The four maps to choose from: text alone (the published method), or 75% text + 25% of one
# citation relation.
MAP_OPTIONS = {
    "text": None,
    "text+coupling": {"s3": 0.75, "coupling": 0.25},
    "text+cocitation": {"s3": 0.75, "cocitation": 0.25},
    "text+direct": {"s3": 0.75, "direct": 0.25},
}
_ALIASES = {"s3": "text", "text": "text", "coupling": "text+coupling", "bib": "text+coupling",
            "bibliographic": "text+coupling", "s3+coupling": "text+coupling", "text+bib": "text+coupling",
            "cocitation": "text+cocitation", "coc": "text+cocitation", "co-citation": "text+cocitation",
            "s3+cocitation": "text+cocitation", "text+coc": "text+cocitation",
            "direct": "text+direct", "s3+direct": "text+direct", "direct citation": "text+direct"}


def map_label(w: Optional[Dict[str, float]]) -> str:
    """'text (S3)' or e.g. '75% text (S3) + 25% co-citation'."""
    if not w:
        return "text (S3)"
    return " + ".join(f"{v:.0%} {NAMES[k]}" for k, v in w.items() if v > 0)


def resolve_weights(w: Union[None, str, bool, Dict[str, float]]) -> Optional[Dict[str, float]]:
    """The map: None or "text" -> text only (S3, the published method);
    "text+coupling", "text+cocitation", "text+direct" -> 75% text + 25% of that relation;
    a dict (or "s3=0.5,coupling=0.5") -> custom weights, normalised; "hybrid" -> 25% each."""
    if w is None or w is False:
        return None
    if isinstance(w, str):
        key = _ALIASES.get(w.strip().lower(), w.strip().lower())
        if key in MAP_OPTIONS:
            w = MAP_OPTIONS[key]
            if w is None:
                return None
    if w is True or (isinstance(w, str) and w.lower() == "hybrid"):
        return dict(EQUAL)
    if isinstance(w, str):  # "s3=0.5,coupling=0.5"
        if "=" not in w:
            raise ValueError(f"Unknown map {w!r}; choose one of {list(MAP_OPTIONS)} or give weights")
        w = {k.strip(): float(v) for k, v in (p.split("=") for p in w.split(",") if p.strip())}
    bad = set(w) - set(COMPONENTS)
    if bad:
        raise ValueError(f"Unknown map components {sorted(bad)}; use {COMPONENTS}")
    tot = sum(max(0.0, float(v)) for v in w.values())
    if tot <= 0:
        raise ValueError("map_weights must contain a positive weight")
    out = {c: round(max(0.0, float(w.get(c, 0.0))) / tot, 6) for c in COMPONENTS}
    out = {c: v for c, v in out.items() if v > 0}
    return None if out == {"s3": 1.0} else out


def _ids(v) -> List[str]:
    return [x.strip() for x in v.split(";") if x.strip()] if isinstance(v, str) else []


def reference_source(df: pd.DataFrame) -> Dict[str, int]:
    """How many papers' references come from Scopus IDs (after enrich_citations), from the
    References text, or are missing."""
    ids = df["reference_eids"].map(_ids) if "reference_eids" in df else pd.Series([[]] * len(df), index=df.index)
    text = df["references"].fillna("").astype(str).str.strip() != "" if "references" in df else ids.map(lambda _: False)
    use_ids = ids.map(len) > 0
    return {"scopus_ids": int(use_ids.sum()), "text": int((~use_ids & text).sum()),
            "none": int((~use_ids & ~text).sum())}


def reference_matrix(df: pd.DataFrame):
    """Sparse papers x cited works. A paper's cited works are its cited Scopus IDs when
    enrich_citations retrieved them (exact, complete lists), otherwise its References text
    normalised to the first 80 characters of each reference."""
    from scipy.sparse import csr_matrix
    from .pipeline import _norm_text
    from .questions import _split_refs
    vocab: Dict[str, int] = {}
    r, c = [], []
    refs = df["references"].tolist() if "references" in df else [None] * len(df)
    ids = df["reference_eids"].tolist() if "reference_eids" in df else [None] * len(df)
    for i, (v, e) in enumerate(zip(refs, ids)):
        keys = {"eid:" + x for x in _ids(e)}
        if not keys:
            keys = {ref for ref in (_norm_text(x)[:80] for x in _split_refs(v)) if len(ref) > 20}
        for k in keys:
            c.append(vocab.setdefault(k, len(vocab)))
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


def _significant(S: np.ndarray, B, alpha: float) -> np.ndarray:
    """Keep only the significant ties of a projection (s3bib.backbone), symmetric."""
    from .backbone import backbone
    n = S.shape[0]
    k = backbone(B, alpha)["keys"]
    M = np.zeros_like(S)
    i, j = k // n, k % n
    M[i, j] = S[i, j]
    M[j, i] = S[j, i]
    return M


def hybrid_matrix(emb: np.ndarray, df: pd.DataFrame, pairs: Sequence[Tuple[int, int]],
                  weights: Dict[str, float], backbone_alpha: Optional[float] = None) -> np.ndarray:
    """Weighted sum of the components. backbone_alpha: keep only the significant ties of
    coupling and co-citation (false discovery rate alpha), as in S3Config(backbone=True)."""
    n = len(df)
    H = np.zeros((n, n), dtype=np.float32)
    if weights.get("s3"):
        H += weights["s3"] * np.clip(emb @ emb.T, 0, None).astype(np.float32)
    if weights.get("coupling"):
        R = reference_matrix(df)
        C = _salton(R)
        H += weights["coupling"] * (_significant(C, R, backbone_alpha) if backbone_alpha else C)
    if weights.get("direct"):
        H += weights["direct"] * direct_matrix(n, pairs)
    if weights.get("cocitation"):
        Ct = citation_matrix(n, pairs).T.tocsr()
        C = _salton(Ct)
        H += weights["cocitation"] * (_significant(C, Ct, backbone_alpha) if backbone_alpha else C)
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
