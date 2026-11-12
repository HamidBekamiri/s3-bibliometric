"""
Core S3 (Semantic Similarity Score) bibliometric pipeline, following the published
method (Bekamiri, Østergaard & Timmermans, Journal of Business Research 219, 116582):

  1. Load a Scopus export (Title + Abstract required).
  2. Embed "Title. Abstract" with sentence-transformers/all-mpnet-base-v2.
  3. S3 matrix = cosine similarity between all paper embeddings.
  4. Time mask: keep S3[i, j] only when year_i <= year_j (a directed graph from
     earlier to later papers).
  5. Scan thresholds 0.75 .. 0.85. At each threshold t the retained papers are those
     with at least one link S3 >= t; communities come from Leiden with directed
     modularity; SII = mean silhouette of the retained papers × number retained.
  6. Keep the threshold with the highest SII (or a fixed threshold, to reproduce a
     published run).
  7. Centrality, outliers, c-TF-IDF terms, keywords and, if References + EID exist,
     in-corpus citations with QI and MI.

community_method="louvain" reproduces the S3 web app (undirected Louvain) instead.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import networkx as nx
import numpy as np
import pandas as pd

log = logging.getLogger("s3bib")

ProgressCallback = Callable[[str, float], None]

DEFAULT_THRESHOLDS = tuple(round(float(t), 2) for t in np.arange(0.75, 0.86, 0.01))

# Column aliases (case-insensitive), same candidates as the web app.
_COLUMNS = {
    "title": ["Title"],
    "abstract": ["Abstract"],
    "year": ["Year", "publication year"],
    "cited_by": ["Cited by", "cited_by", "citedby"],
    "references": ["References", "Refs"],
    "eid": ["EID"],
    "authors": ["Authors", "Author(s)", "Author Names", "Authors Names",
                "Authors with affiliations"],
    "affiliations": ["Affiliations", "Affiliation", "Affiliation(s)",
                     "Authors with affiliations"],
    "keywords": ["Author Keywords", "authkeywords", "Keywords"],
    "source": ["Source Title", "source_title", "Journal", "Publication Title"],
    "doi": ["DOI"],
    "funding": ["Funding Details", "Funding Texts", "Funding Text", "Funding Orgs",
                "funding_details", "Funding"],
}


# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    """Fix every random number generator S3 touches, so runs are reproducible."""
    import os
    import random
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
    except ImportError:  # pragma: no cover
        pass


@dataclass
class S3Config:
    model_name: str = "sentence-transformers/all-mpnet-base-v2"  # or a key: "specter", "clip", ...
    thresholds: Optional[Sequence[float]] = None  # None = 0.75-0.85 for mpnet, wider for other models
    temporal_mask: bool = True
    seed: int = 42
    batch_size: int = 32
    start_year: Optional[int] = None
    end_year: Optional[int] = None
    top_n_communities: int = 10      # communities used for citation matrix / QI / MI
    cache_dir: Optional[str] = None  # cache embeddings here (keyed by content hash)
    community_method: str = "leiden"  # "leiden" (directed, as in the paper) or "louvain" (web app)
    threshold: Optional[float] = None  # fix the threshold instead of choosing it by SII
    mi_denominator: str = "retained"   # MI denominator: links made by retained papers, or "all" links
    dedupe: bool = True                # remove duplicate records before the analysis
    dedupe_threshold: float = 0.98     # S3 at or above this = the same paper (reprints, double indexing)

    def __post_init__(self):
        from .models import default_thresholds, resolve_model
        self.model_name = resolve_model(self.model_name)
        if self.thresholds is None:
            self.thresholds = default_thresholds(self.model_name)
        self.thresholds = tuple(round(float(t), 4) for t in self.thresholds)


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
def _find(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    norm = {c.strip().lower(): c for c in df.columns}
    for c in candidates:
        if c.strip().lower() in norm:
            return norm[c.strip().lower()]
    return None


def _norm_text(s: Any) -> str:
    s = re.sub(r"[^a-z0-9 ]+", " ", str(s).lower())
    return re.sub(r"\s+", " ", s).strip()


def load_papers(
    source: Union[str, Path, pd.DataFrame],
    start_year: Optional[int] = None,
    end_year: Optional[int] = None,
    drop_empty: bool = True,
) -> pd.DataFrame:
    """Read a Scopus CSV/Excel export (or a DataFrame) into standard columns.
    Records with neither a title nor an abstract are dropped (drop_empty=True)."""
    if isinstance(source, pd.DataFrame):
        raw = source.copy()
    else:
        p = Path(source).expanduser()
        if p.suffix.lower() in (".xlsx", ".xls"):
            raw = pd.read_excel(p)
        else:
            raw = pd.read_csv(p, low_memory=False)

    cols = {k: _find(raw, v) for k, v in _COLUMNS.items()}
    if cols["title"] is None or cols["abstract"] is None:
        raise ValueError("File must contain 'Title' and 'Abstract' columns.")

    df = pd.DataFrame({k: (raw[c] if c else np.nan) for k, c in cols.items()})
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["cited_by"] = pd.to_numeric(df["cited_by"], errors="coerce").fillna(0).astype(int)
    for k in ("title", "abstract"):
        df[k] = df[k].fillna("").astype(str)
    df.loc[df["abstract"].str.startswith("[No abstract"), "abstract"] = ""

    if start_year is not None:
        df = df[df["year"] >= start_year]
    if end_year is not None:
        df = df[df["year"] <= end_year]
    df["title"] = df["title"].str.strip()
    df["abstract"] = df["abstract"].str.strip()
    if drop_empty:
        df = df[(df["title"] != "") | (df["abstract"] != "")]
    df = df.reset_index(drop=True)
    df["text"] = df["title"] + ". " + df["abstract"]
    if len(df) < 3:
        raise ValueError(f"Only {len(df)} papers after filtering; need at least 3.")
    return df


def _removed_row(df: pd.DataFrame, i: int, reason: str, kept: Optional[int] = None) -> Dict[str, Any]:
    return {"title": df.at[i, "title"], "year": _i(df.at[i, "year"]), "eid": df.at[i, "eid"],
            "cited_by": int(df.at[i, "cited_by"]), "reason": reason,
            "kept_title": df.at[kept, "title"] if kept is not None else None,
            "kept_eid": df.at[kept, "eid"] if kept is not None else None}


def _keep_order(df: pd.DataFrame, group: List[int]) -> List[int]:
    """Within a group of duplicates keep the most cited record, then the earliest, then the first."""
    return sorted(group, key=lambda i: (-int(df.at[i, "cited_by"]),
                                        df.at[i, "year"] if pd.notna(df.at[i, "year"]) else 9999, i))


def _groups(pairs: List[tuple], n: int) -> List[List[int]]:
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for a, b in pairs:
        parent[find(a)] = find(b)
    g: Dict[int, List[int]] = {}
    for i in range(n):
        g.setdefault(find(i), []).append(i)
    return [v for v in g.values() if len(v) > 1]


def exact_duplicates(df: pd.DataFrame) -> tuple[List[int], List[Dict[str, Any]]]:
    """Rows that repeat another record: same EID, same title + abstract, or same DOI with
    the same title (chapters of one edited book can share the book's DOI).
    Returns (rows to drop, removal records)."""
    title = df["title"].map(_norm_text)
    text = (df["title"] + " " + df["abstract"]).map(_norm_text)
    keys = {
        "eid": df["eid"].astype(str).str.strip().str.lower(),
        "text": text,
        "doi": df["doi"].astype(str).str.strip().str.lower() + "|" + title,
    }
    pairs = []
    for name, v in keys.items():
        bad = v.isin(["", "nan", "none"]) | v.str.startswith(("nan|", "|", "none|"))
        v = v[~bad]
        for _, idx in v.groupby(v).groups.items():
            idx = list(idx)
            pairs += [(idx[0], j) for j in idx[1:]]
    drop, removed = [], []
    for g in _groups(pairs, len(df)):
        keep, *rest = _keep_order(df, g)
        for j in rest:
            removed.append(_removed_row(df, j, "exact duplicate (same EID, DOI + title, or text)", keep))
            drop.append(j)
    return sorted(drop), removed


def near_duplicates(df: pd.DataFrame, emb: np.ndarray, threshold: float) -> tuple[List[int], List[Dict[str, Any]]]:
    """Papers whose S3 is >= threshold with another paper (reprints, the same paper indexed
    twice). Returns (rows to drop, removal records)."""
    S = emb @ emb.T
    np.fill_diagonal(S, 0)
    i, j = np.where(np.triu(S >= threshold, 1))
    drop, removed = [], []
    for g in _groups(list(zip(i.tolist(), j.tolist())), len(df)):
        keep, *rest = _keep_order(df, g)
        for r in rest:
            removed.append({**_removed_row(df, r, f"near duplicate (S3 ≥ {threshold})", keep),
                            "s3": round(float(S[min(r, keep), max(r, keep)]), 4)})
            drop.append(r)
    return sorted(drop), removed


# --------------------------------------------------------------------------
# Embeddings & S3 matrix
# --------------------------------------------------------------------------
_MODELS: Dict[str, Any] = {}


def _get_model(name: str):
    if name not in _MODELS:
        from sentence_transformers import SentenceTransformer
        _MODELS[name] = SentenceTransformer(name, device="cpu")
    return _MODELS[name]


def embed_texts(texts: List[str], model_name: str, batch_size: int = 32,
                cache_dir: Optional[str] = None,
                progress: Optional[ProgressCallback] = None) -> np.ndarray:
    """L2-normalised embeddings, shape (N, d). Cached by content hash if cache_dir."""
    cache_file = None
    if cache_dir:
        h = hashlib.sha1((model_name + "\x00" + "\x00".join(texts)).encode("utf-8")).hexdigest()
        cache_file = Path(cache_dir).expanduser() / f"{h}_emb.npy"
        if cache_file.exists():
            return np.load(cache_file)

    model = _get_model(model_name)
    chunks = []
    n_batches = max(1, (len(texts) + batch_size - 1) // batch_size)
    for b in range(n_batches):
        batch = texts[b * batch_size:(b + 1) * batch_size]
        if batch:
            chunks.append(model.encode(batch, batch_size=batch_size,
                                       show_progress_bar=False,
                                       convert_to_numpy=True))
        if progress:
            progress(f"Embedding papers ({b + 1}/{n_batches} batches)", (b + 1) / n_batches)
    emb = np.vstack(chunks).astype(np.float32)
    emb /= np.clip(np.linalg.norm(emb, axis=1, keepdims=True), 1e-12, None)

    if cache_file is not None:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_file, emb)
    return emb


def s3_matrix(emb: np.ndarray) -> np.ndarray:
    """Cosine similarity of normalised embeddings (the S3 matrix)."""
    return (emb @ emb.T).astype(np.float32)


def apply_temporal_mask(s3: np.ndarray, years: np.ndarray) -> np.ndarray:
    """Keep S3[i, j] only where year_i <= year_j (papers without a year drop out)."""
    if np.all(np.isnan(years)):
        return s3
    mask = years[:, None] <= years[None, :]
    return s3 * mask


# --------------------------------------------------------------------------
# Graph, Louvain, SII
# --------------------------------------------------------------------------
def build_graph(sim: np.ndarray, threshold: float) -> nx.Graph:
    """Undirected graph with an edge (i, j) where sim >= threshold (and > 0).

    Matches the web app: for a time-masked matrix, the surviving direction's
    weight is used.
    """
    n = sim.shape[0]
    sym = np.maximum(sim, sim.T)
    iu, ju = np.where(np.triu(sym >= threshold, k=1) & (np.triu(sym, k=1) > 0))
    G = nx.Graph()
    G.add_nodes_from(range(n))
    G.add_weighted_edges_from(zip(iu.tolist(), ju.tolist(), sym[iu, ju].astype(float).tolist()))
    return G


def _louvain(G: nx.Graph, seed: int) -> Dict[int, int]:
    from community import community_louvain  # python-louvain
    return community_louvain.best_partition(G, weight="weight", random_state=seed)


def _silhouette(s3_full: np.ndarray, labels: np.ndarray) -> tuple[float, int]:
    from sklearn.metrics import silhouette_score
    uniq, counts = np.unique(labels, return_counts=True)
    keep = uniq[counts > 1]
    mask = np.isin(labels, keep)
    n_kept = int(mask.sum())
    if n_kept < 2 or len(keep) < 2:
        return 0.0, n_kept
    dist = np.maximum(1.0 - s3_full[np.ix_(mask, mask)], 0.0)
    np.fill_diagonal(dist, 0.0)
    try:
        return float(silhouette_score(dist, labels[mask], metric="precomputed")), n_kept
    except Exception:  # pragma: no cover
        return 0.0, n_kept


def _leiden_directed(s3_directed: np.ndarray, threshold: float, seed: int):
    """Leiden with directed modularity on the time-masked graph (edges i -> j with
    S3 >= threshold). Returns ({node: community} for retained papers, n_edges)."""
    import igraph as ig
    import leidenalg as la
    A = (s3_directed >= threshold) & (s3_directed > 0)
    np.fill_diagonal(A, False)
    src, dst = np.where(A)
    if len(src) == 0:
        return {}, 0
    keep = np.unique(np.concatenate([src, dst]))
    pos = {int(v): k for k, v in enumerate(keep)}
    g = ig.Graph(n=len(keep), edges=[(pos[int(a)], pos[int(b)]) for a, b in zip(src, dst)], directed=True)
    g.es["weight"] = s3_directed[src, dst].astype(float).tolist()
    part = la.find_partition(g, la.ModularityVertexPartition, weights="weight", seed=seed)
    return {int(keep[k]): int(c) for k, c in enumerate(part.membership)}, len(src)


def _partition_at(s3_directed: np.ndarray, s3_full: np.ndarray, t: float, seed: int,
                  method: str):
    """One threshold: (scan row, {node: community} for every node) or (None, None)."""
    from sklearn.metrics import silhouette_score
    n = s3_full.shape[0]
    if method == "louvain":
        G = build_graph(s3_directed, t)
        if G.number_of_edges() == 0:
            return None, None
        part = _louvain(G, seed)
        labels = np.array([part[i] for i in range(n)])
        sil, n_kept = _silhouette(s3_full, labels)
        n_edges = G.number_of_edges()
    else:
        retained, n_edges = _leiden_directed(s3_directed, t, seed)
        if not retained:
            return None, None
        nodes = np.array(sorted(retained))
        labels_r = np.array([retained[i] for i in nodes])
        n_kept = len(nodes)
        sil = 0.0
        if len(set(labels_r)) > 1:
            dist = np.maximum(1.0 - s3_full[np.ix_(nodes, nodes)], 0.0)
            np.fill_diagonal(dist, 0.0)
            sil = float(silhouette_score(dist, labels_r, metric="precomputed"))
        # papers without links become their own (singleton) community
        part = {i: retained.get(i, -1 - i) for i in range(n)}
        labels = np.array([part[i] for i in range(n)])
    n_comm = int(sum(1 for c in Counter(labels).values() if c > 1))
    row = {"threshold": float(t), "silhouette": sil, "n_papers": int(n_kept),
           "sii": sil * n_kept, "n_communities": n_comm, "n_edges": int(n_edges)}
    return row, part


def threshold_scan(s3_directed: np.ndarray, s3_full: np.ndarray,
                   thresholds: Sequence[float], seed: int,
                   progress: Optional[ProgressCallback] = None, method: str = "leiden"):
    """Return (scan_rows, best_row, best_partition)."""
    rows, best, best_part = [], None, None
    for k, t in enumerate(thresholds):
        row, part = _partition_at(s3_directed, s3_full, float(t), seed, method)
        if row is not None:
            rows.append(row)
            if best is None or row["sii"] > best["sii"]:
                best, best_part = row, part
        if progress:
            progress(f"Threshold scan t={t:.2f}", (k + 1) / len(thresholds))
    return rows, best, best_part


# --------------------------------------------------------------------------
# Citations (in-corpus, matched from the References field)
# --------------------------------------------------------------------------
def citation_edges(df: pd.DataFrame) -> List[tuple[int, int]]:
    """(citing_idx, cited_idx) pairs: a paper cites another if the first 60
    chars of the other's normalised title appear in its References text."""
    if df["references"].isna().all() or df["eid"].isna().all():
        return []
    fp = df["title"].map(_norm_text).str[:60]
    refs = df["references"].fillna("").map(_norm_text)
    fp_to_idx: Dict[str, int] = {}
    for i, f in enumerate(fp):
        if f:
            fp_to_idx[f] = i  # last one wins, as in the web app
    eids = df["eid"].tolist()
    edges = set()
    for i, r in enumerate(refs):
        if not r:
            continue
        for f, j in fp_to_idx.items():
            if f in r and eids[i] != eids[j]:
                edges.add((i, j))
    return sorted(edges)


# --------------------------------------------------------------------------
# Result object
# --------------------------------------------------------------------------
@dataclass
class S3Result:
    papers: pd.DataFrame            # one row per paper, incl. community, centrality, outlier
    embeddings: np.ndarray
    scan: List[Dict[str, Any]]
    best: Optional[Dict[str, Any]]
    config: S3Config
    citation_pairs: List[tuple[int, int]] = field(default_factory=list)
    labels: Dict[int, str] = field(default_factory=dict)  # human / LLM community labels
    removed: pd.DataFrame = field(default_factory=pd.DataFrame)  # dropped records and why
    citation_source: str = "references"  # "references" (title matching) or "scopus_api"

    # ---- graph & guided questions -----------------------------------------
    def s3(self, directed: bool = True) -> np.ndarray:
        """The S3 matrix (recomputed from the stored embeddings)."""
        s = s3_matrix(self.embeddings)
        if directed and self.config.temporal_mask:
            s = apply_temporal_mask(s, self.papers["year"].to_numpy(dtype=float))
        return s

    def graph(self) -> nx.Graph:
        """The S3 similarity graph at the chosen threshold (cached)."""
        if getattr(self, "_graph", None) is None:
            t = self.best["threshold"] if self.best else 1.01
            self._graph = build_graph(self.s3(), t)
        return self._graph

    def questions(self, available_only: bool = False) -> pd.DataFrame:
        """The ranked research questions S3 can answer, and whether this data supports each."""
        from .questions import question_table
        return question_table(self, available_only)

    def answer(self, qid: str):
        """Answer one question, e.g. res.answer("Q7"). Returns an Answer
        with .findings (sentences), .tables (DataFrames) and .chart."""
        from .questions import answer
        set_seed(self.config.seed)
        return answer(self, qid)

    def report(self, path: Union[str, Path], title: Optional[str] = None) -> Path:
        """Answer every available question and write one self-contained HTML report."""
        from .report import build_report
        return build_report(self, path, title)

    # ---- views -----------------------------------------------------------
    def communities(self, min_size: int = 2) -> pd.DataFrame:
        """One row per community (size >= min_size), largest first."""
        d = self.papers.dropna(subset=["community"])
        g = d.groupby("community").agg(
            n_papers=("title", "size"),
            total_citations=("cited_by", "sum"),
            mean_citations=("cited_by", "mean"),
            year_min=("year", "min"),
            year_median=("year", "median"),
            year_max=("year", "max"),
        ).reset_index()
        g["community"] = g["community"].astype(int)
        g = g[g["n_papers"] >= min_size].sort_values("n_papers", ascending=False)
        g["label"] = [self.labels.get(int(c), "") for c in g["community"]]
        cm = self.citation_metrics()
        if not cm.empty:
            g = g.merge(cm[["community", "qi", "mi", "within_citations",
                            "inward_citations", "outward_citations"]],
                        on="community", how="left")
        return g.reset_index(drop=True)

    def community_papers(self, community: int, n: Optional[int] = 10) -> pd.DataFrame:
        sub = self.papers[self.papers["community"] == community]
        sub = sub.sort_values(["eigen_centrality", "cited_by"], ascending=False)
        return sub if n is None else sub.head(n)

    def outliers(self) -> pd.DataFrame:
        return self.papers[self.papers["is_outlier"]].sort_values("cited_by", ascending=False)

    def top_keywords(self, community: int, n: int = 15) -> List[tuple[str, int]]:
        kws: List[str] = []
        for v in self.papers.loc[self.papers["community"] == community, "keywords"].dropna():
            kws += [k.strip().lower() for k in str(v).split(";")
                    if len(k.strip()) > 2 and k.strip().lower() != "nan"]
        return Counter(kws).most_common(n)

    def top_terms(self, community: int, n: int = 15) -> List[tuple[str, float]]:
        """Most distinctive terms of a community by c-TF-IDF (paper Eq. 6): each community
        is one document; W[x, c] = tf[x, c] × log(1 + A / f[x]), where f[x] is the term's
        frequency across all communities and A the average word count per community."""
        if getattr(self, "_ctfidf", None) is None:
            from sklearn.feature_extraction.text import CountVectorizer
            P = self.papers.dropna(subset=["community"])
            ids = sorted(int(c) for c in P["community"].unique())
            docs = [" ".join(P.loc[P["community"] == c, "text"]) for c in ids]
            if not docs:
                self._ctfidf = ({}, None, [])
            else:
                vec = CountVectorizer(stop_words="english", token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z-]+\b")
                tf = vec.fit_transform(docs).toarray().astype(float)
                f = tf.sum(axis=0)
                A = tf.sum() / len(ids)
                W = tf * np.log(1 + A / np.maximum(f, 1))
                self._ctfidf = ({c: k for k, c in enumerate(ids)}, W, vec.get_feature_names_out())
        pos, W, terms = self._ctfidf
        if community not in pos:
            return []
        w = W[pos[community]]
        order = np.argsort(w)[::-1][:n]
        return [(str(terms[i]), round(float(w[i]), 3)) for i in order if w[i] > 0]

    def citation_matrix(self) -> pd.DataFrame:
        """Citations between the top-N communities (rows cite columns)."""
        ids = self._top_ids()
        C = pd.DataFrame(0, index=ids, columns=ids)
        comm = self.papers["community"]
        for i, j in self.citation_pairs:
            ci, cj = comm.iat[i], comm.iat[j]
            if pd.notna(ci) and pd.notna(cj) and int(ci) in C.index and int(cj) in C.index:
                C.loc[int(ci), int(cj)] += 1
        return C

    def mi_denominator(self) -> int:
        """Total field citations used for MI: in-corpus citation links made by retained
        papers (default), or all in-corpus links (mi_denominator='all')."""
        if not self.citation_pairs:
            return 0
        if self.config.mi_denominator == "all":
            return len(self.citation_pairs)
        out = self.papers["is_outlier"].to_numpy()
        return int(sum(1 for i, _ in self.citation_pairs if not out[i]))

    def citation_metrics(self) -> pd.DataFrame:
        """Table 1 of the S3 paper for the top-N communities.

        within  = citations from papers in c to other papers in c
        outward = citations from papers in c to any other paper in the corpus
        inward  = citations to papers in c from any other paper in the corpus
        QI = within / size;  MI = (within + inward) / field citations × 100;
        local = within + inward;  global = sum of Scopus 'Cited by'."""
        if not self.citation_pairs:
            return pd.DataFrame()
        comm = self.papers["community"].astype("float").to_numpy()
        cp = np.asarray(self.citation_pairs)
        ci, cj = comm[cp[:, 0]], comm[cp[:, 1]]
        denom = self.mi_denominator()
        rows = []
        for cid in self._top_ids():
            in_c = self.papers["community"] == cid
            size = int(in_c.sum())
            within = int(((ci == cid) & (cj == cid)).sum())
            outward = int(((ci == cid) & (cj != cid)).sum())
            inward = int(((ci != cid) & (cj == cid)).sum())
            rows.append({
                "community": int(cid), "size": size,
                "within_citations": within, "outward_citations": outward,
                "inward_citations": inward,
                "qi": within / size if size else 0.0,
                "mi": (within + inward) / denom * 100 if denom else 0.0,
                "local_citations": within + inward,
                "global_citations": int(self.papers.loc[in_c.fillna(False), "cited_by"].sum()),
            })
        return pd.DataFrame(rows)

    def search(self, query: str, top_k: int = 10) -> pd.DataFrame:
        q = embed_texts([query], self.config.model_name)[0]
        scores = self.embeddings @ q
        order = np.argsort(scores)[::-1][:top_k]
        out = self.papers.iloc[order].copy()
        out.insert(0, "score", scores[order])
        return out

    def summary(self, n_communities: int = 15) -> Dict[str, Any]:
        comms = self.communities()
        return {
            "n_papers": int(len(self.papers)),
            "year_range": [_i(self.papers["year"].min()), _i(self.papers["year"].max())],
            "best_threshold": self.best,
            "n_communities": int(len(comms)),
            "n_outliers": int(self.papers["is_outlier"].sum()),
            "n_citation_links": len(self.citation_pairs),
            "communities": _records(comms.head(n_communities)),
        }

    # ---- persistence -----------------------------------------------------
    def save(self, directory: Union[str, Path]) -> Path:
        d = Path(directory).expanduser()
        d.mkdir(parents=True, exist_ok=True)
        self.papers.to_pickle(d / "papers.pkl")
        self.removed.to_pickle(d / "removed.pkl")
        np.save(d / "embeddings.npy", self.embeddings)
        meta = {"scan": self.scan, "best": self.best, "config": asdict(self.config),
                "citation_pairs": self.citation_pairs,
                "labels": {str(k): v for k, v in self.labels.items()},
                "citation_source": self.citation_source}
        (d / "run.json").write_text(json.dumps(meta, indent=2, default=_json_default))
        return d

    @classmethod
    def load(cls, directory: Union[str, Path]) -> "S3Result":
        d = Path(directory).expanduser()
        meta = json.loads((d / "run.json").read_text())
        c = dict(meta["config"])
        c.setdefault("community_method", "louvain")  # runs saved before 0.3.0 used Louvain
        c["thresholds"] = tuple(c["thresholds"])
        cfg = S3Config(**{k: v for k, v in c.items() if k in S3Config.__dataclass_fields__})
        return cls(papers=pd.read_pickle(d / "papers.pkl"),
                   embeddings=np.load(d / "embeddings.npy"),
                   scan=meta["scan"], best=meta["best"], config=cfg,
                   citation_pairs=[tuple(p) for p in meta["citation_pairs"]],
                   labels={int(k): v for k, v in meta.get("labels", {}).items()},
                   removed=pd.read_pickle(d / "removed.pkl") if (d / "removed.pkl").exists() else pd.DataFrame(),
                   citation_source=meta.get("citation_source", "references"))

    def export(self, directory: Union[str, Path]) -> Dict[str, str]:
        """Write human-friendly CSV/XLSX outputs."""
        d = Path(directory).expanduser()
        d.mkdir(parents=True, exist_ok=True)
        papers = self.papers.drop(columns=["text"]).copy()
        papers["community_label"] = papers["community"].map(
            lambda c: self.labels.get(int(c), "") if pd.notna(c) else "")
        out = {"papers_csv": d / "papers_with_communities.csv",
               "communities_csv": d / "communities.csv",
               "threshold_scan_csv": d / "threshold_scan.csv"}
        papers.to_csv(out["papers_csv"], index=False)
        self.communities().to_csv(out["communities_csv"], index=False)
        pd.DataFrame(self.scan).to_csv(out["threshold_scan_csv"], index=False)
        if self.citation_pairs:
            out["citation_matrix_csv"] = d / "citation_matrix.csv"
            self.citation_matrix().to_csv(out["citation_matrix_csv"])
        return {k: str(v) for k, v in out.items()}

    def _top_ids(self) -> List[int]:
        vc = self.papers["community"].value_counts()  # largest first
        return [int(c) for c in vc.index[: self.config.top_n_communities]]


# --------------------------------------------------------------------------
# Main entry point
# --------------------------------------------------------------------------
def run_s3(source: Union[str, Path, pd.DataFrame],
           config: Optional[S3Config] = None,
           progress: Optional[ProgressCallback] = None) -> S3Result:
    """Run the full S3 pipeline on a Scopus export and return an S3Result."""
    cfg = config or S3Config()
    say = progress or (lambda s, p: log.info("%s (%.0f%%)", s, p * 100))
    set_seed(cfg.seed)

    say("Loading papers", 0.0)
    df = load_papers(source, cfg.start_year, cfg.end_year, drop_empty=False)
    empty = (df["title"] == "") & (df["abstract"] == "")
    removed: List[Dict[str, Any]] = [_removed_row(df, i, "no title and no abstract") for i in df.index[empty]]
    df = df[~empty].reset_index(drop=True)

    # embed before de-duplicating, so the embedding cache is the same with and without it
    say(f"Embedding {len(df)} papers", 0.05)
    emb = embed_texts(df["text"].tolist(), cfg.model_name, cfg.batch_size, cfg.cache_dir,
                      progress=lambda s, p: say(s, 0.05 + 0.45 * p))
    if cfg.dedupe:
        drop, rem = exact_duplicates(df)
        removed += rem
        keep = np.setdiff1d(np.arange(len(df)), drop)
        df, emb = df.iloc[keep].reset_index(drop=True), emb[keep]
        drop, rem = near_duplicates(df, emb, cfg.dedupe_threshold)
        removed += rem
        keep = np.setdiff1d(np.arange(len(df)), drop)
        df, emb = df.iloc[keep].reset_index(drop=True), emb[keep]
    n = len(df)

    say("Computing S3 matrix", 0.5)
    s3 = s3_matrix(emb)
    years = df["year"].to_numpy(dtype=float)
    s3_dir = apply_temporal_mask(s3, years) if cfg.temporal_mask else s3

    if cfg.community_method not in ("leiden", "louvain"):
        raise ValueError("community_method must be 'leiden' or 'louvain'")
    scan, best, part = threshold_scan(s3_dir, s3, cfg.thresholds, cfg.seed,
                                      progress=lambda s, p: say(s, 0.5 + 0.35 * p),
                                      method=cfg.community_method)
    if cfg.threshold is not None:  # reproduce a fixed (e.g. published) threshold
        best, part = _partition_at(s3_dir, s3, float(cfg.threshold), cfg.seed, cfg.community_method)
        if best is not None:
            best["fixed"] = True
            if not any(abs(r["threshold"] - best["threshold"]) < 1e-9 for r in scan):
                scan = sorted(scan + [dict(best)], key=lambda r: r["threshold"])

    df["community"] = pd.array([None] * n, dtype="Int64")
    df["eigen_centrality"] = 0.0
    df["is_outlier"] = False
    df["nearest_similarity"] = np.nan
    df["suggested_community"] = pd.array([None] * n, dtype="Int64")

    pairs: List[tuple[int, int]] = []
    if best is not None:
        say("Building final community graph", 0.86)
        G = build_graph(s3_dir, best["threshold"])
        labels = np.array([part[i] for i in range(n)])
        sizes = Counter(labels)
        # Singletons are not communities: they are the S3 "irrelevant" papers.
        df["community"] = pd.array([int(l) if sizes[l] > 1 else None for l in labels], dtype="Int64")
        df["community"] = _renumber(df["community"])
        try:
            eig = nx.eigenvector_centrality_numpy(G, weight="weight")
        except Exception:
            eig = {i: 0.0 for i in range(n)}
        df["eigen_centrality"] = [float(eig.get(i, 0.0)) for i in range(n)]

        deg = np.array([G.degree(i) for i in range(n)])
        df["is_outlier"] = deg == 0
        for i in np.where(deg == 0)[0]:
            row = s3_dir[i].copy()
            row[i] = -1.0
            j = int(row.argmax())
            df.at[i, "nearest_similarity"] = float(row[j])
            df.at[i, "suggested_community"] = df.at[j, "community"]

        say("Matching in-corpus citations", 0.9)
        pairs = citation_edges(df)
        if pairs:
            # local citations = citations a paper receives from other papers in the corpus
            received = Counter(j for _, j in pairs)
            df["local_citations"] = [received.get(i, 0) for i in range(n)]

    say("Done", 1.0)
    return S3Result(papers=df, embeddings=emb, scan=scan, best=best,
                    config=cfg, citation_pairs=pairs, removed=pd.DataFrame(removed))


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _renumber(s: pd.Series) -> pd.Series:
    """Renumber communities 0..k-1 by size (0 = largest)."""
    order = s.value_counts().index.tolist()
    mapping = {old: new for new, old in enumerate(order)}
    return s.map(mapping).astype("Int64")


def _i(x):
    return None if pd.isna(x) else int(x)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, tuple):
        return list(o)
    raise TypeError(type(o))


def _records(df: pd.DataFrame) -> List[Dict[str, Any]]:
    return json.loads(df.to_json(orient="records"))
