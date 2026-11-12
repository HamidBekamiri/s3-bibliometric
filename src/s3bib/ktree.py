"""Knowledge tree: how communities develop and feed each other across time windows.

Columns are time windows (default 5 years, the last window starting at the latest
publication year). A node is one community's papers published in one window (N = papers).
A link runs from an earlier node to a later node and has two weights:

  citation   = citations from papers in the later node to papers in the earlier node
  similarity = S3 links (S3 >= the chosen threshold) between papers of the two nodes

so the tree shows both explicit (citation) and implicit (semantic) inheritance.
"""
from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


def time_windows(years, size: int = 5) -> List[tuple]:
    """Windows of `size` years; the last one starts at the latest year (e.g. 2023-2027)."""
    y = pd.Series(years).dropna().astype(int)
    if y.empty:
        return []
    lo, hi = int(y.min()), int(y.max())
    starts = []
    s = hi
    while s + size - 1 >= lo:
        starts.append(s)
        s -= size
    return [(s, s + size - 1) for s in sorted(starts)]


def knowledge_tree(res, window_years: int = 5, n_communities: int = 15) -> Dict[str, Any]:
    """Nodes (community × window) and links (earlier -> later) with citation and similarity
    weights. Uses only papers in the n largest communities."""
    P = res.papers
    comms = [int(c) for c in res.communities()["community"].head(n_communities)]
    comm = P["community"].astype("float").to_numpy()
    shown = np.isin(comm, comms)
    years = P["year"].to_numpy(dtype=float)
    wins = time_windows(years[shown & ~np.isnan(years)], window_years)
    if not wins:
        return {"windows": [], "nodes": pd.DataFrame(), "links": pd.DataFrame()}
    def win_of(y):
        if np.isnan(y):
            return -1
        for k, (a, b) in enumerate(wins):
            if a <= y <= b:
                return k
        return -1
    w = np.array([win_of(y) for y in years])
    ok = shown & (w >= 0)

    nodes = Counter((int(comm[i]), int(w[i])) for i in np.where(ok)[0])
    sim = Counter()
    for u, v in res.graph().edges():
        if ok[u] and ok[v] and w[u] != w[v]:
            a, b = (u, v) if w[u] < w[v] else (v, u)  # earlier -> later
            sim[(int(comm[a]), int(w[a]), int(comm[b]), int(w[b]))] += 1
    cit = Counter()
    for citing, cited in res.citation_pairs:
        if ok[citing] and ok[cited] and w[cited] < w[citing]:
            cit[(int(comm[cited]), int(w[cited]), int(comm[citing]), int(w[citing]))] += 1

    label = lambda c: res.labels.get(c, "")
    node_df = pd.DataFrame([{"community": c, "label": label(c), "window": k,
                             "period": f"{wins[k][0]}-{wins[k][1]}", "papers": n}
                            for (c, k), n in sorted(nodes.items(), key=lambda x: (x[0][1], x[0][0]))])
    keys = set(sim) | set(cit)
    link_df = pd.DataFrame([{"from_community": a, "from_period": f"{wins[wa][0]}-{wins[wa][1]}",
                             "to_community": b, "to_period": f"{wins[wb][0]}-{wins[wb][1]}",
                             "from_window": wa, "to_window": wb,
                             "citations": cit.get(k, 0), "similarity_links": sim.get(k, 0)}
                            for k in sorted(keys) for (a, wa, b, wb) in [k]])
    return {"windows": [f"{a}-{b}" for a, b in wins], "communities": comms,
            "nodes": node_df, "links": link_df, "citation_source": res.citation_source}


def ktree_spec(res, window_years: int = 5, n_communities: int = 15) -> Optional[Dict[str, Any]]:
    kt = knowledge_tree(res, window_years, n_communities)
    if kt["nodes"].empty:
        return None
    nodes = [[int(r.community), int(r.window), int(r.papers)] for r in kt["nodes"].itertuples()]
    links = [[int(r.from_community), int(r.from_window), int(r.to_community), int(r.to_window),
              int(r.citations), int(r.similarity_links)] for r in kt["links"].itertuples()] \
        if not kt["links"].empty else []
    from .questions import _display_name
    names = {int(c): _display_name(res, c) for c in kt["communities"]}
    return {"type": "ktree",
            "title": f"Knowledge tree: communities per {window_years}-year window, linked by citations "
                     f"and {'S3 similarity' if not res.config.map_weights else 'map links'} (earlier → later)",
            "windows": kt["windows"], "communities": kt["communities"], "names": names,
            "nodes": nodes, "links": links,
            "has_citations": bool(res.citation_pairs)}
