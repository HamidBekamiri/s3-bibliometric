"""Choosing the map before the full analysis (an extension of the S3 paper).

The published S3 method builds the map from text similarity alone; s3bib can mix in a
citation relation (s3bib.hybrid). This module compares the candidate maps on measures that
do not each favour one kind of map, and recommends one with a rule fixed in advance:

  held-out citations  20% of the in-corpus citation links are hidden while the maps are built
                      (direct citation and co-citation use only the other 80%). The score is
                      the share of hidden links that fall inside one community, minus the share
                      for random communities of the same sizes. Without hiding them, a map
                      built from citations would be judged on its own input.
  TCI gain            topic coherence of the communities (1 - Jensen-Shannon divergence of
                      TF-IDF words, as in the S3 paper's Table 5) minus that of random
                      communities of the same sizes.
  stability           adjusted Rand index between the partition and (a) the partitions with two
                      other random seeds and (b) the partitions of two 90% samples of the papers,
                      on the papers clustered in both. A map whose communities change when the
                      data change slightly is not one to build a review on.
  coverage            share of the papers placed in a community (the others are outliers).
  agreement with text adjusted Rand index with the text map: how different the result is.
                      Reported, not used for the choice.

Each map is built as in run_s3: the threshold scan picks the threshold by SII, and hybrid
maps scan thresholds that give the same numbers of links as the text scan.

Rule (RULE below): keep the text map unless a hybrid recovers more held-out citations
(by at least `min_heldout_gain`), keeps the TCI gain (at most `tci_tolerance` lower), and
loses at most `max_stability_loss` stability and `max_coverage_loss` coverage. Among the
hybrids that pass, the one with the highest held-out gain is recommended.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

ProgressCallback = Callable[[str, float], None]

MARGINS = {"min_heldout_gain": 0.01, "tci_tolerance": 0.005,
           "max_stability_loss": 0.05, "max_coverage_loss": 0.05}
RULE = ("Keep the text map (the published S3 method) unless a hybrid map places a share of the "
        "held-out citations inside communities (beyond random) at least {min_heldout_gain:g} higher than "
        "the text map does, has a TCI gain over random no more than {tci_tolerance:g} below the text map's, "
        "and loses no more than {max_stability_loss:g} stability (ARI) and {max_coverage_loss:g} coverage "
        "(share of papers). Among the hybrids that pass, choose the one with the highest held-out gain.")
MIN_SUPPORT = 0.05  # a citation relation must give at least one tie to this share of the papers


def candidates(extended: bool = False) -> List[Tuple[str, Optional[Dict[str, float]], bool]]:
    """(name, weights, significant ties only). Default: text and the three 75/25 hybrids.
    extended: also 10% and 50% citation weight and the significant-ties variants."""
    rel = (("coupling", "coupling"), ("cocitation", "cocitation"), ("direct", "direct"))
    out: List[Tuple[str, Optional[Dict[str, float]], bool]] = [("text", None, False)]
    shares = (0.1, 0.25, 0.5) if extended else (0.25,)
    for key, short in rel:
        for w in shares:
            name = f"text+{short}" if w == 0.25 else f"text+{short} {w:.0%}"
            out.append((name, {"s3": round(1 - w, 2), key: w}, False))
        if extended and key != "direct":
            out.append((f"text+{short} significant", {"s3": 0.75, key: 0.25}, True))
    return out


def split_pairs(pairs: Sequence[Tuple[int, int]], holdout: float, seed: int):
    """Hide a share of the citation links. Links are split as unordered pairs, so a pair that
    cites both ways is hidden in full."""
    groups: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}
    for a, b in pairs:
        groups.setdefault((min(a, b), max(a, b)), []).append((int(a), int(b)))
    keys = sorted(groups)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(keys))
    k = int(round(holdout * len(keys)))
    test = [keys[i] for i in order[:k]]
    train = [p for i in order[k:] for p in groups[keys[i]]]
    return sorted(train), test


def support(df: pd.DataFrame, pairs: Sequence[Tuple[int, int]]) -> Dict[str, float]:
    """Share of papers with at least one tie in each citation relation."""
    from .backbone import _binary
    from .hybrid import reference_matrix
    n = len(df)
    out = {"coupling": 0.0, "cocitation": 0.0, "direct": 0.0}
    if "references" in df and df["references"].notna().any():
        R = _binary(reference_matrix(df)).tocsc()
        shared = np.asarray(R.sum(axis=0)).ravel() >= 2
        out["coupling"] = float((np.asarray(R[:, shared].sum(axis=1)).ravel() > 0).mean()) if shared.any() else 0.0
    if pairs:
        outdeg = pd.Series([a for a, _ in pairs]).value_counts()
        cociting = {a for a, d in outdeg.items() if d >= 2}
        out["cocitation"] = len({b for a, b in pairs if a in cociting}) / n
        out["direct"] = len({x for p in pairs for x in p}) / n
    return out


def _labels(part: Dict[int, int], n: int) -> np.ndarray:
    """Community per paper, -1 for papers outside a community (singletons)."""
    lab = np.array([part[i] for i in range(n)])
    uniq, counts = np.unique(lab, return_counts=True)
    single = set(uniq[counts < 2].tolist())
    return np.array([-1 if x in single else x for x in lab])


def hits(test: Sequence[Tuple[int, int]], labels: np.ndarray) -> np.ndarray:
    """For each hidden link: True if both papers are in the same community."""
    a = np.array([p[0] for p in test], dtype=int)
    b = np.array([p[1] for p in test], dtype=int)
    return (labels[a] == labels[b]) & (labels[a] >= 0)


def recovered(test: Sequence[Tuple[int, int]], labels: np.ndarray) -> float:
    return float(hits(test, labels).mean()) if test else float("nan")


def paired_ci(h: np.ndarray, h0: np.ndarray, seed: int, n_boot: int = 2000) -> Tuple[float, float]:
    """95% bootstrap interval (over the hidden links) of the difference in recovered links."""
    d = h.astype(float) - h0.astype(float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    m = d[idx].mean(axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def recovered_random(test, labels: np.ndarray, seed: int, repeats: int = 20) -> float:
    """Random communities of the same sizes, over all papers (so coverage is the same)."""
    if not test:
        return float("nan")
    rng = np.random.default_rng(seed)
    return float(np.mean([recovered(test, rng.permutation(labels)) for _ in range(repeats)]))


def _ari(a: np.ndarray, b: np.ndarray) -> float:
    from sklearn.metrics import adjusted_rand_score
    m = (a >= 0) & (b >= 0)
    if m.sum() < 3:
        return float("nan")
    return float(adjusted_rand_score(a[m], b[m]))


def _stability(H_dir, H, t: float, labels: np.ndarray, seed: int, method: str, sample: float = 0.9):
    from .pipeline import _partition_at
    n = len(labels)
    seeds, samples = [], []
    for k in (1, 2):
        _, part = _partition_at(H_dir, H, t, seed + k, method)
        if part is not None:
            seeds.append(_ari(labels, _labels(part, n)))
    rng = np.random.default_rng(seed)
    for _ in range(2):
        idx = np.sort(rng.choice(n, int(round(sample * n)), replace=False))
        _, part = _partition_at(H_dir[np.ix_(idx, idx)], H[np.ix_(idx, idx)], t, seed, method)
        if part is not None:
            sub = np.full(n, -1)
            sub[idx] = _labels(part, len(idx))
            samples.append(_ari(labels, sub))
    return (float(np.nanmean(seeds)) if seeds else float("nan"),
            float(np.nanmean(samples)) if samples else float("nan"))


def compare_prepared(df: pd.DataFrame, emb: np.ndarray, pairs: Sequence[Tuple[int, int]], cfg,
                     extended: bool = False, holdout: float = 0.2,
                     margins: Optional[Dict[str, float]] = None,
                     progress: Optional[ProgressCallback] = None) -> Dict[str, Any]:
    """Compare the candidate maps on prepared papers (after de-duplication) and their embeddings."""
    from .evaluate import random_partition, tci, tfidf_matrix
    from .hybrid import density_thresholds, hybrid_matrix
    from .models import default_thresholds
    from .pipeline import _link_counts, apply_temporal_mask, s3_matrix, threshold_scan
    say = progress or (lambda s, p: None)
    m = {**MARGINS, **(margins or {})}
    n, seed, method = len(df), cfg.seed, cfg.community_method
    years = df["year"].to_numpy(dtype=float)
    mask = (lambda M: apply_temporal_mask(M, years)) if cfg.temporal_mask else (lambda M: M)
    train, test = split_pairs(pairs, holdout, seed) if pairs else ([], [])
    sup = support(df, train)
    P = tfidf_matrix(df)

    plain = s3_matrix(emb)
    plain_dir = mask(plain)
    counts = _link_counts(plain_dir, cfg.thresholds)
    default_scan = tuple(cfg.thresholds) == tuple(default_thresholds(cfg.model_name))

    cands = candidates(extended)
    rows, skipped, parts, hit = [], [], {}, {}
    for k, (name, w, sig) in enumerate(cands):
        say(f"Map {k + 1}/{len(cands)}: {name}", k / len(cands))
        rel = [c for c in (w or {}) if c != "s3"]
        low = [r for r in rel if sup[r] < MIN_SUPPORT]
        if low:
            r = low[0]
            skipped.append({"map": name, "reason": f"only {sup[r]:.0%} of the papers have a {r} tie "
                                                   f"(needs {MIN_SUPPORT:.0%})"})
            continue
        if w is None:
            H, H_dir, ths = plain, plain_dir, cfg.thresholds
        else:
            H = hybrid_matrix(emb, df, train, w, cfg.backbone_alpha if sig else None)
            H_dir = mask(H)
            ths = density_thresholds(H_dir, counts) if default_scan else cfg.thresholds
        _, best, part = threshold_scan(H_dir, H, ths, seed, method=method)
        if best is None:
            skipped.append({"map": name, "reason": "no community structure at any threshold"})
            continue
        lab = _labels(part, n)
        rnd = [random_partition(lab, seed + j) for j in range(5)]
        t = tci(P, lab)
        t0 = float(np.nanmean([tci(P, r) for r in rnd]))
        rec, rec0 = recovered(test, lab), recovered_random(test, lab, seed)
        st_seed, st_sample = _stability(H_dir, H, best["threshold"], lab, seed, method)
        parts[name] = lab
        if test:
            hit[name] = hits(test, lab)
        rows.append({"map": name, "weights": w, "significant ties": sig,
                     "held-out citations gain": rec - rec0, "held-out citations recovered": rec,
                     "held-out random (same sizes)": rec0, "TCI gain over random": t - t0,
                     "stability (ARI)": float(np.nanmean([st_seed, st_sample])),
                     "stability across seeds": st_seed, "stability on 90% samples": st_sample,
                     "coverage": float((lab >= 0).mean()), "papers clustered": int((lab >= 0).sum()),
                     "communities": int(len(set(lab[lab >= 0]))),
                     "papers with a tie": min([sup[r] for r in rel], default=float("nan")),
                     "threshold": float(best["threshold"]), "silhouette (own map)": float(best["silhouette"])})
        if w is not None:
            del H, H_dir
    say("Comparing the maps", 1.0)
    t = pd.DataFrame(rows)
    if len(t) and "text" in parts:
        t["agreement with text (ARI)"] = [_ari(parts["text"], parts[r]) for r in t["map"]]
    if len(t) and "text" in hit:
        ci = [paired_ci(hit[r], hit["text"], seed) for r in t["map"]]
        t["held-out vs text, CI low"] = [c[0] for c in ci]
        t["held-out vs text, CI high"] = [c[1] for c in ci]
    rec_name, reasons = _decide(t, m, bool(test))
    pick = t[t["map"] == rec_name].iloc[0] if rec_name in set(t.get("map", [])) else None
    return {"table": t, "recommended": rec_name, "extended": extended,
            "weights": None if pick is None else pick["weights"],
            "backbone": bool(pick is not None and pick["significant ties"]),
            "reasons": reasons, "rule": RULE.format(**m), "margins": m, "skipped": skipped,
            "support": sup, "holdout": holdout, "n_heldout": len(test), "n_links": len(pairs)}


def _decide(t: pd.DataFrame, m: Dict[str, float], have_test: bool) -> Tuple[str, List[str]]:
    if not len(t) or "text" not in set(t["map"]):
        return ("text", ["The text map could not be built; nothing to compare."])
    base = t[t["map"] == "text"].iloc[0]
    if not have_test:
        return ("text", ["No in-corpus citation links, so held-out citations cannot be measured; "
                         "the rule keeps the text map."])
    reasons, passed = [], []
    for _, r in t[t["map"] != "text"].iterrows():
        why = []
        if not r["held-out citations gain"] >= base["held-out citations gain"] + m["min_heldout_gain"]:
            why.append(f"held-out gain {r['held-out citations gain']:.3f} vs text {base['held-out citations gain']:.3f}")
        if not r["TCI gain over random"] >= base["TCI gain over random"] - m["tci_tolerance"]:
            why.append(f"TCI gain {r['TCI gain over random']:.3f} vs text {base['TCI gain over random']:.3f}")
        if not r["stability (ARI)"] >= base["stability (ARI)"] - m["max_stability_loss"]:
            why.append(f"stability {r['stability (ARI)']:.2f} vs text {base['stability (ARI)']:.2f}")
        if not r["coverage"] >= base["coverage"] - m["max_coverage_loss"]:
            why.append(f"coverage {r['coverage']:.0%} vs text {base['coverage']:.0%}")
        if why:
            reasons.append(f"{r['map']}: fails on " + "; ".join(why) + ".")
        else:
            passed.append(r)
    if not passed:
        return ("text", ["No hybrid map passes the rule, so the text map (the published method) is kept."] + reasons)
    best = max(passed, key=lambda r: r["held-out citations gain"])
    others = [r["map"] for r in passed if r["map"] != best["map"]]
    note = ""
    if "held-out vs text, CI low" in best and pd.notna(best["held-out vs text, CI low"]):
        lo, hi = best["held-out vs text, CI low"], best["held-out vs text, CI high"]
        note = (f" 95% interval of the difference in recovered hidden links: {lo:+.3f} to {hi:+.3f}"
                + (", so the advantage could be sampling noise." if lo <= 0 else "."))
    return (best["map"], [f"{best['map']} passes the rule with the highest held-out gain "
                          f"({best['held-out citations gain']:.3f} vs text {base['held-out citations gain']:.3f})."
                          + note + (f" Also passing: {', '.join(others)}." if others else "")] + reasons)


def compare_maps(source, config=None, extended: bool = False, holdout: float = 0.2,
                 margins: Optional[Dict[str, float]] = None,
                 progress: Optional[ProgressCallback] = None) -> Dict[str, Any]:
    """Load and embed a Scopus export as run_s3 does, then compare the candidate maps.
    Embeddings are cached (S3Config.cache_dir), so the full run afterwards does not embed again."""
    from .pipeline import S3Config, citation_edges, prepare, set_seed
    cfg = config or S3Config()
    say = progress or (lambda s, p: None)
    set_seed(cfg.seed)
    df, emb, _ = prepare(source, cfg, lambda s, p: say(s, 0.4 * p))
    say("Matching in-corpus citations", 0.4)
    pairs = citation_edges(df, strict=cfg.citation_match == "strict")
    return compare_prepared(df, emb, pairs, cfg, extended, holdout, margins,
                            progress=lambda s, p: say(s, 0.42 + 0.58 * p))


def to_jsonable(choice: Dict[str, Any]) -> Dict[str, Any]:
    import json
    out = {k: v for k, v in choice.items() if k != "table"}
    out["table"] = json.loads(choice["table"].to_json(orient="records"))
    return out


def from_jsonable(d: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(d)
    out["table"] = pd.DataFrame(d.get("table", []))
    return out
