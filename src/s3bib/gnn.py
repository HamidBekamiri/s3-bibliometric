"""Graph learning on the S3 networks (Q30-Q34), in plain PyTorch.

  Q30 node classification   GraphSAGE: which community does a new or outlier paper belong to?
  Q31 node regression       GraphSAGE: which recent papers will be highly cited, and is impact
                            driven by content or by network position?
  Q32 heterogeneous graph   R-GCN on the knowledge graph: embeddings of authors, institutions,
                            funders, keywords and journals for expertise search and partner
                            matching
  Q33 temporal graph        GCN + GRU over yearly community states: which communities will
                            grow or decline next?
  Q34 KG embeddings         TransE and RotatE: ranked missing links in the knowledge graph

Every model is small, trained on CPU with a fixed seed, and compared with a simple baseline
on held-out data, so the report says whether the graph model actually adds anything.
PyTorch comes with sentence-transformers; nothing else is needed.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


def _torch():
    import torch
    return torch


def _seed(seed: int) -> None:
    import random
    torch = _torch()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))


def _mean_adj(n: int, src: Sequence[int], dst: Sequence[int]):
    """Sparse row-normalised adjacency: row dst averages the features of its src nodes."""
    torch = _torch()
    if len(src) == 0:
        return torch.sparse_coo_tensor(torch.zeros((2, 0), dtype=torch.long), torch.zeros(0), (n, n), check_invariants=False).coalesce()
    src, dst = np.asarray(src), np.asarray(dst)
    deg = np.bincount(dst, minlength=n).astype(np.float32)
    val = 1.0 / deg[dst]
    idx = torch.tensor(np.vstack([dst, src]), dtype=torch.long)
    return torch.sparse_coo_tensor(idx, torch.tensor(val, dtype=torch.float32), (n, n), check_invariants=False).coalesce()


def _undirected(pairs: Sequence[Tuple[int, int]]) -> Tuple[List[int], List[int]]:
    s = [a for a, b in pairs] + [b for a, b in pairs]
    d = [b for a, b in pairs] + [a for a, b in pairs]
    return s, d


class _SAGE:
    """Two-layer GraphSAGE (mean aggregator)."""

    def __new__(cls, d_in: int, d_hid: int, d_out: int, dropout: float = 0.3):
        torch = _torch()
        nn = torch.nn

        class SAGE(nn.Module):
            def __init__(self):
                super().__init__()
                self.s1, self.n1 = nn.Linear(d_in, d_hid), nn.Linear(d_in, d_hid, bias=False)
                self.s2, self.n2 = nn.Linear(d_hid, d_out), nn.Linear(d_hid, d_out, bias=False)
                self.drop = nn.Dropout(dropout)

            def forward(self, x, A):
                h = torch.relu(self.s1(x) + self.n1(torch.sparse.mm(A, x)))
                h = self.drop(h)
                return self.s2(h) + self.n2(torch.sparse.mm(A, h))
        return SAGE()


def _train(model, loss_fn, epochs: int = 200, lr: float = 0.01, wd: float = 5e-4):
    torch = _torch()
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        loss = loss_fn()
        loss.backward()
        opt.step()
    model.eval()
    return model


# ===========================================================================
# Q30 node classification
# ===========================================================================
def node_classification(res, n_comms: int = 15, test_size: float = 0.2) -> Optional[Dict]:
    """Predict the community of papers that have no S3 links (like new or outlier papers)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.model_selection import train_test_split
    torch = _torch()
    seed = res.config.seed
    _seed(seed)
    P, E = res.papers, res.embeddings.astype(np.float32)
    n = len(P)
    comms = [int(c) for c in res.communities()["community"].head(n_comms)]
    comm = P["community"]
    lab = np.array([int(c) if pd.notna(c) and int(c) in comms else -1 for c in comm])
    idx = np.where(lab >= 0)[0]
    if len(idx) < 50 or len(set(lab[idx])) < 3:
        return None
    cls = {c: k for k, c in enumerate(sorted(set(lab[idx])))}
    y_all = np.array([cls.get(v, -1) for v in lab])
    tr, te = train_test_split(idx, test_size=test_size, random_state=seed, stratify=lab[idx])
    te_set = set(te.tolist())
    map_edges = [(u, v) for u, v in res.graph().edges()]
    cite = list(res.citation_pairs)

    # test papers lose their S3 links (like outliers or brand-new papers); citations stay
    def adj(drop: set):
        pairs = [(u, v) for u, v in map_edges if u not in drop and v not in drop] + cite
        return _mean_adj(n, *_undirected(pairs))

    def fit_sage(train_idx, A):
        model = _SAGE(E.shape[1], 128, len(cls))
        X = torch.tensor(E)
        yt = torch.tensor(y_all[train_idx], dtype=torch.long)
        ti = torch.tensor(train_idx, dtype=torch.long)
        _train(model, lambda: torch.nn.functional.cross_entropy(model(X, A)[ti], yt), epochs=150)
        with torch.no_grad():
            return torch.softmax(model(X, A), dim=1).numpy()

    S = E[te] @ E[tr].T
    # like a real outlier, a test paper has no neighbour at or above the threshold
    t = res.best["threshold"] if res.best else 1.0
    if not res.config.map_weights:
        S = np.where(S >= t, -np.inf, S)
    nn_pred = y_all[tr][S.argmax(axis=1)]                     # today's rule: most similar paper
    cen = np.vstack([E[tr][y_all[tr] == k].mean(axis=0) for k in range(len(cls))])
    cen /= np.linalg.norm(cen, axis=1, keepdims=True)
    cen_pred = (E[te] @ cen.T).argmax(axis=1)
    lr = LogisticRegression(max_iter=2000, C=1.0).fit(E[tr], y_all[tr])
    lr_pred = lr.predict(E[te])
    prob = fit_sage(tr, adj(te_set))
    sage_pred = prob[te].argmax(axis=1)
    yt = y_all[te]
    rows = []
    for name, pred in [("Nearest paper by S3 (current rule)", nn_pred), ("Nearest community centroid", cen_pred),
                       ("Text classifier (logistic regression)", lr_pred), ("GraphSAGE (text + citations + S3 links)", sage_pred)]:
        rows.append({"method": name, "accuracy": round(accuracy_score(yt, pred), 3),
                     "macro F1": round(f1_score(yt, pred, average="macro"), 3)})
    has_cite = np.array([any(a == i or b == i for a, b in cite) for i in te]) if cite else np.zeros(len(te), bool)
    by_cite = None
    if has_cite.any() and (~has_cite).any():
        by_cite = {"with citations": round(accuracy_score(yt[has_cite], sage_pred[has_cite]), 3),
                   "without citations": round(accuracy_score(yt[~has_cite], sage_pred[~has_cite]), 3),
                   "share with citations": round(float(has_cite.mean()), 3)}

    # assign the real outliers: train on all labelled papers
    prob_all = fit_sage(idx, adj(set()))
    inv = {k: c for c, k in cls.items()}
    out = P.index[P["is_outlier"].fillna(False).to_numpy()]
    a = pd.DataFrame({"title": P.loc[out, "title"].str[:110], "year": P.loc[out, "year"],
                      "cited_by": P.loc[out, "cited_by"],
                      "GNN community": [inv[int(prob_all[i].argmax())] for i in out],
                      "confidence": [round(float(prob_all[i].max()), 3) for i in out],
                      "nearest-paper community": P.loc[out, "suggested_community"]})
    agree = float((a["GNN community"] == a["nearest-paper community"]).mean()) if len(a) else np.nan
    return {"table": pd.DataFrame(rows), "n_train": len(tr), "n_test": len(te), "classes": len(cls),
            "by_citations": by_cite, "outliers": a.sort_values("cited_by", ascending=False),
            "agreement": agree, "confident": float((a["confidence"] >= 0.5).mean()) if len(a) else np.nan}


# ===========================================================================
# Q31 node regression: future impact
# ===========================================================================
def impact_regression(res, test_years: int = 5, horizon: int = 3) -> Optional[Dict]:
    """Predict citations per year of recent papers from content, position, or both."""
    from scipy.stats import spearmanr
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.linear_model import Ridge
    from sklearn.metrics import r2_score
    torch = _torch()
    seed = res.config.seed
    _seed(seed)
    P, E = res.papers, res.embeddings.astype(np.float32)
    yr = P["year"].to_numpy(dtype=float)
    ymax = int(np.nanmax(yr))
    split = ymax - test_years
    n = len(P)
    ok = ~np.isnan(yr)
    target = np.log1p(P["cited_by"].to_numpy(dtype=float) / np.maximum(1.0, (ymax + 1) - np.nan_to_num(yr, nan=ymax)))
    train = np.where(ok & (yr <= split))[0]
    test = np.where(ok & (yr > split) & (yr <= split + horizon))[0]
    if len(train) < 100 or len(test) < 30:
        return None
    # only links to earlier (or same-year) papers: no information from the future
    cite = [(a, b) for a, b in res.citation_pairs if ok[a] and ok[b] and yr[b] <= yr[a]]
    simlinks = []
    for u, v in res.graph().edges():
        if ok[u] and ok[v]:
            a, b = (u, v) if yr[u] >= yr[v] else (v, u)  # a = later paper
            simlinks.append((a, b))
    src = [b for a, b in cite] + [b for a, b in simlinks]
    dst = [a for a, b in cite] + [a for a, b in simlinks]
    A = _mean_adj(n, src, dst)
    pop_pre = Counter(b for a, b in res.citation_pairs if ok[a] and yr[a] <= split)
    out_c = Counter(a for a, b in cite)
    n_sim = Counter(a for a, b in simlinks)
    comm = P["community"]
    size_pre = Counter(int(c) for c, y in zip(comm, yr) if pd.notna(c) and y <= split)
    cited_by_paper = defaultdict(list)
    for a, b in cite:
        cited_by_paper[a].append(pop_pre.get(b, 0))
    struct = np.c_[[out_c.get(i, 0) for i in range(n)],
                   [n_sim.get(i, 0) for i in range(n)],
                   [size_pre.get(int(c), 0) if pd.notna(c) else 0 for c in comm],
                   [np.mean(cited_by_paper[i]) if cited_by_paper[i] else 0 for i in range(n)],
                   [np.max(cited_by_paper[i]) if cited_by_paper[i] else 0 for i in range(n)],
                   np.nan_to_num(yr - split, nan=0.0)].astype(np.float32)
    names = ["in-corpus references", "S3 links to earlier papers", "community size (up to split)",
             "mean popularity of cited papers", "max popularity of cited papers", "year"]
    yt = target[test]
    preds = {}
    content = Ridge(alpha=10.0).fit(np.c_[E[train], struct[train, -1:]], target[train])
    preds["Content: text embedding (ridge)"] = content.predict(np.c_[E[test], struct[test, -1:]])
    pos = HistGradientBoostingRegressor(random_state=seed, max_iter=200).fit(struct[train], target[train])
    preds["Position: network features (boosted trees)"] = pos.predict(struct[test])
    from sklearn.decomposition import PCA
    pca = PCA(n_components=32, random_state=seed).fit(E[train])
    both = HistGradientBoostingRegressor(random_state=seed, max_iter=200).fit(
        np.c_[pca.transform(E[train]), struct[train]], target[train])
    preds["Content + position (boosted trees)"] = both.predict(np.c_[pca.transform(E[test]), struct[test]])
    X = torch.tensor(np.c_[E, struct[:, -1:] / max(1, test_years)])
    model = _SAGE(X.shape[1], 128, 1, dropout=0.2)
    ti = torch.tensor(train, dtype=torch.long)
    yy = torch.tensor(target[train], dtype=torch.float32)
    _train(model, lambda: torch.nn.functional.mse_loss(model(X, A)[ti, 0], yy), epochs=200, lr=0.005)
    with torch.no_grad():
        g = model(X, A)[:, 0].numpy()
    preds["GraphSAGE (text + earlier links)"] = g[test]
    preds["Year only (baseline)"] = np.full(len(test), target[train].mean()) + 1e-9 * np.arange(len(test))
    rows = []
    for name, p in preds.items():
        rho = spearmanr(yt, p).correlation
        rows.append({"method": name, "Spearman ρ": round(float(rho), 3), "R²": round(float(r2_score(yt, p)), 3)})
    table = pd.DataFrame(rows)
    imp = pd.Series(np.abs(np.corrcoef(np.c_[struct[train], target[train]].T)[-1, :-1]), index=names).round(3)
    # forecast: the newest papers (after the test window), ranked by the GNN
    newest = np.where(ok & (yr > split + horizon))[0]
    f = pd.DataFrame({"title": P.loc[newest, "title"].str[:110], "year": yr[newest].astype(int),
                      "cited_by so far": P.loc[newest, "cited_by"].to_numpy(),
                      "predicted citations / year": np.round(np.expm1(g[newest]), 1)}
                     ).sort_values("predicted citations / year", ascending=False).head(15)
    return {"table": table, "train": f"≤{split}", "test": f"{split + 1}-{split + horizon}", "n_test": len(test),
            "feature_corr": imp, "forecast": f}


# ===========================================================================
# Q32 heterogeneous graph: R-GCN on the knowledge graph
# ===========================================================================
_ENTITY_KINDS = ("author", "institution", "funder", "keyword", "journal")
_REL = {"author": "AUTHORED_BY", "institution": "AFFILIATED_WITH", "funder": "FUNDED_BY",
        "keyword": "HAS_KEYWORD", "journal": "PUBLISHED_IN"}


def _kg_index(res, min_papers: int = 2):
    """Paper rows + entities with >= min_papers papers; typed edge lists paper->entity."""
    from .networks import _memberships
    n = len(res.papers)
    nodes = [("paper", i) for i in range(n)]
    kind_of = ["paper"] * n
    ent_index: Dict[Tuple[str, str], int] = {}
    edges: Dict[str, List[Tuple[int, int]]] = {}
    for kind in _ENTITY_KINDS:
        try:
            mem = _memberships(res, kind)
        except Exception:
            continue
        cnt = Counter(name for _, name, _ in mem)
        keep = {k for k, v in cnt.items() if v >= min_papers}
        if not keep:
            continue
        for name in sorted(keep):
            ent_index[(kind, name)] = len(nodes)
            nodes.append((kind, name))
            kind_of.append(kind)
        edges[kind] = sorted({(int(i), ent_index[(kind, name)]) for i, name, _ in mem if name in keep})
    return nodes, kind_of, ent_index, edges


def hetero_embeddings(res, dim: int = 128, epochs: int = 150, holdout: float = 0.1) -> Optional[Dict]:
    torch = _torch()
    nn = torch.nn
    seed = res.config.seed
    _seed(seed)
    E = res.embeddings.astype(np.float32)
    nodes, kind_of, ent_index, edges = _kg_index(res)
    if not edges:
        return None
    N = len(nodes)
    n = len(res.papers)
    rng = np.random.default_rng(seed)
    test, train = {}, {}
    for kind, ed in edges.items():
        ed = np.array(ed)
        perm = rng.permutation(len(ed))
        k = int(len(ed) * holdout) if kind in ("author", "institution", "funder") else 0
        test[kind], train[kind] = ed[perm[:k]], ed[perm[k:]]
    # entity features: mean embedding of their training papers
    X = np.zeros((N, E.shape[1]), dtype=np.float32)
    X[:n] = E
    acc = defaultdict(list)
    for kind, ed in train.items():
        for p, e in ed:
            acc[e].append(p)
    for e, ps in acc.items():
        v = E[ps].mean(axis=0)
        X[e] = v / (np.linalg.norm(v) or 1.0)
    rels = []
    for kind, ed in train.items():
        if len(ed):
            rels.append((f"{_REL[kind]}", ed[:, 0], ed[:, 1]))      # paper -> entity
            rels.append((f"{_REL[kind]}^-1", ed[:, 1], ed[:, 0]))   # entity -> paper
    if res.citation_pairs:
        c = np.array(res.citation_pairs)
        rels.append(("CITES", c[:, 0], c[:, 1]))
        rels.append(("CITES^-1", c[:, 1], c[:, 0]))
    g = np.array(list(res.graph().edges())) if res.graph().number_of_edges() else np.zeros((0, 2), int)
    if len(g):
        rels.append(("SIMILAR_TO", np.r_[g[:, 0], g[:, 1]], np.r_[g[:, 1], g[:, 0]]))
    A = [_mean_adj(N, s, d) for _, s, d in rels]   # row d aggregates from s

    class RGCN(nn.Module):
        def __init__(self):
            super().__init__()
            self.inp = nn.Linear(X.shape[1], dim)
            self.self1, self.self2 = nn.Linear(dim, dim), nn.Linear(dim, dim)
            self.W1 = nn.ModuleList([nn.Linear(dim, dim, bias=False) for _ in rels])
            self.W2 = nn.ModuleList([nn.Linear(dim, dim, bias=False) for _ in rels])
            self.rel = nn.ParameterDict({k: nn.Parameter(torch.ones(dim)) for k in edges})  # DistMult

        def forward(self, x):
            h = torch.relu(self.inp(x))
            h1 = self.self1(h) + sum(W(torch.sparse.mm(a, h)) for W, a in zip(self.W1, A))
            h1 = torch.relu(h1)
            return self.self2(h1) + sum(W(torch.sparse.mm(a, h1)) for W, a in zip(self.W2, A))

        def encode_isolated(self, x):  # a query or new paper without links
            return self.self2(torch.relu(self.self1(torch.relu(self.inp(x)))))

    model = RGCN()
    Xt = torch.tensor(X)
    by_kind = {k: np.array([i for i, t in enumerate(kind_of) if t == k]) for k in edges}

    def loss_fn():
        H = model(Xt)
        loss = 0.0
        for kind, ed in train.items():
            if not len(ed):
                continue
            b = ed[rng.integers(0, len(ed), min(len(ed), 2048))]
            p, e = torch.tensor(b[:, 0]), torch.tensor(b[:, 1])
            neg = torch.tensor(by_kind[kind][rng.integers(0, len(by_kind[kind]), len(b))])
            r = model.rel[kind]
            pos_s = (H[p] * r * H[e]).sum(1)
            neg_s = (H[p] * r * H[neg]).sum(1)
            loss = loss + nn.functional.softplus(-pos_s).mean() + nn.functional.softplus(neg_s).mean()
        return loss

    _train(model, loss_fn, epochs=epochs, lr=0.005, wd=0.0)
    with torch.no_grad():
        H = model(Xt).numpy()
        R = {k: v.detach().numpy() for k, v in model.rel.items()}
    from sklearn.metrics import roc_auc_score
    rows = []
    for kind, ed in test.items():
        if len(ed) < 20:
            continue
        neg = by_kind[kind][rng.integers(0, len(by_kind[kind]), len(ed))]
        y = np.r_[np.ones(len(ed)), np.zeros(len(ed))]
        pp = np.r_[ed[:, 0], ed[:, 0]]
        ee = np.r_[ed[:, 1], neg]
        s_gnn = (H[pp] * R[kind] * H[ee]).sum(1)
        s_cos = (X[pp] * X[ee]).sum(1)  # paper text vs entity's other papers
        rows.append({"link": f"paper → {kind}", "held-out links": len(ed),
                     "R-GCN AUC": round(float(roc_auc_score(y, s_gnn)), 3),
                     "S3 baseline AUC": round(float(roc_auc_score(y, s_cos)), 3)})
    names = {i: nodes[i][1] for i in range(n, N)}
    out = {"table": pd.DataFrame(rows), "H": H, "R": R, "kind_of": np.array(kind_of), "names": names,
           "model": model, "n_entities": N - n, "relations": len(rels)}
    res._hetero = out
    return out


def find_experts(res, text_or_vector, kind: str = "author", n: int = 10) -> pd.DataFrame:
    """Entities (author, institution, funder, keyword, journal) that best fit a text, via the
    R-GCN: the text is encoded like a new paper without links and scored with the same
    decoder as the paper→entity links."""
    torch = _torch()
    het = getattr(res, "_hetero", None) or hetero_embeddings(res)
    if het is None or kind not in het["R"]:
        raise ValueError(f"No {kind} embeddings for this run")
    if isinstance(text_or_vector, str):
        from .pipeline import embed_texts
        v = embed_texts([text_or_vector], res.config.model_name)[0]
    else:
        v = np.asarray(text_or_vector, dtype=np.float32)
    with torch.no_grad():
        q = het["model"].encode_isolated(torch.tensor(v[None, :], dtype=torch.float32)).numpy()[0]
    idx = np.where(het["kind_of"] == kind)[0]
    s = (het["H"][idx] * het["R"][kind] * q).sum(1)
    top = idx[np.argsort(-s)[:n]]
    return pd.DataFrame({kind: [het["names"][i] for i in top],
                         "score": np.round(np.sort(s)[::-1][:n], 3)})


# ===========================================================================
# Q33 temporal graph learning: which communities grow or decline next?
# ===========================================================================
def community_forecast(res, n_comms: int = 15, window: int = 3, test_years: int = 3) -> Optional[Dict]:
    from scipy.stats import spearmanr
    from sklearn.linear_model import Ridge
    torch = _torch()
    nn = torch.nn
    seed = res.config.seed
    _seed(seed)
    P = res.papers
    yr = P["year"].to_numpy(dtype=float)
    comms = [int(c) for c in res.communities()["community"].head(n_comms)]
    if len(comms) < 4:
        return None
    K = len(comms)
    ci = {c: k for k, c in enumerate(comms)}
    cc = np.array([ci.get(int(c), -1) if pd.notna(c) else -1 for c in P["community"]])
    per_year = pd.Series(yr).value_counts()
    years = sorted(int(y) for y in per_year.index if per_year[y] >= 10)
    if len(years) < 3 * window + 4:
        return None
    y0, ymax = years[0], int(np.nanmax(yr))
    T = list(range(y0, ymax + 1))
    cnt = np.zeros((len(T), K))
    tot = np.zeros(len(T))
    for i, y in enumerate(yr):
        if np.isnan(y) or y < y0:
            continue
        t = int(y) - y0
        tot[t] += 1
        if cc[i] >= 0:
            cnt[t, cc[i]] += 1
    # citations received per community per year (from any paper of that year)
    inflow = np.zeros((len(T), K))
    for a, b in res.citation_pairs:
        if not np.isnan(yr[a]) and yr[a] >= y0 and cc[b] >= 0:
            inflow[int(yr[a]) - y0, cc[b]] += 1

    def share(t_end):  # share of papers in [t_end-window+1, t_end]
        s = slice(max(0, t_end - window + 1), t_end + 1)
        return (cnt[s].sum(0) + 0.5) / (tot[s].sum() + 0.5 * K)

    def feats(t):
        s_now, s_prev = share(t), share(t - window)
        s = slice(max(0, t - window + 1), t + 1)
        inf = (inflow[s].sum(0) + 0.5) / (inflow[s].sum() + 0.5 * K)
        return np.c_[np.log(s_now), np.log(s_now) - np.log(s_prev), np.log(inf),
                     np.log1p(cnt[: t + 1].sum(0))]

    def target(t):
        return np.log(share(t + window)) - np.log(share(t))

    E = res.embeddings
    cen = np.vstack([E[cc == k].mean(0) for k in range(K)])
    cen /= np.linalg.norm(cen, axis=1, keepdims=True)
    S = cen @ cen.T
    np.fill_diagonal(S, -1)
    nb = np.argsort(-S, axis=1)[:, :3]
    Adj = np.zeros((K, K), dtype=np.float32)
    for k in range(K):
        Adj[k, nb[k]] = 1 / 3
    At = torch.tensor(Adj)
    t_test = len(T) - 1 - window                 # predict the last full window
    t_first = 2 * window
    train_t = [t for t in range(t_first, t_test - window + 1)]
    if len(train_t) < 4:
        return None
    seq = 3

    def X_seq(t):
        return torch.tensor(np.stack([feats(t - window * (seq - 1 - j)) for j in range(seq)]), dtype=torch.float32)

    class TGNN(nn.Module):
        def __init__(self, d_in=4, d=32):
            super().__init__()
            self.s, self.n = nn.Linear(d_in, d), nn.Linear(d_in, d, bias=False)
            self.gru = nn.GRU(d, d, batch_first=True)
            self.out = nn.Linear(d, 1)

        def forward(self, xs):  # xs: (seq, K, d_in)
            h = torch.stack([torch.relu(self.s(x) + self.n(At @ x)) for x in xs], dim=1)  # (K, seq, d)
            o, _ = self.gru(h)
            return self.out(o[:, -1])[:, 0]

    ok_t = [t for t in train_t if t - window * (seq - 1) >= window]
    Xs = [X_seq(t) for t in ok_t]
    Ys = [torch.tensor(target(t), dtype=torch.float32) for t in ok_t]
    model = TGNN()
    _train(model, lambda: sum(nn.functional.mse_loss(model(x), y) for x, y in zip(Xs, Ys)) / len(Xs),
           epochs=300, lr=0.01, wd=1e-3)
    with torch.no_grad():
        p_gnn = model(X_seq(t_test)).numpy()
    yt = target(t_test)
    flat_X = np.vstack([feats(t) for t in ok_t])
    flat_y = np.concatenate([target(t) for t in ok_t])
    p_lin = Ridge(alpha=1.0).fit(flat_X, flat_y).predict(feats(t_test))
    p_pers = feats(t_test)[:, 1]  # momentum continues
    rows = []
    for name, p in [("Temporal GNN (GCN + GRU over communities)", p_gnn), ("Linear model, no graph", p_lin),
                    ("Momentum continues (baseline)", p_pers), ("No change (baseline)", np.zeros(K) + 1e-9 * np.arange(K))]:
        rows.append({"method": name, "Spearman ρ": round(float(spearmanr(yt, p).correlation), 3),
                     "direction correct": round(float(np.mean(np.sign(p) == np.sign(yt))), 3),
                     "mean abs. error": round(float(np.mean(np.abs(p - yt))), 3)})
    # forecast the next window from the latest year
    t_now = len(T) - 1
    with torch.no_grad():
        f = model(X_seq(t_now)).numpy()
    first = [int(np.min(yr[cc == k])) for k in range(K)]

    def status(k, v):
        if first[k] >= ymax - 2 * window and v > 0.1:
            return "emerging"
        return "growing" if v > 0.15 else "declining" if v < -0.15 else "stable"
    fc = pd.DataFrame({"community": comms, "share now": np.round(share(t_now), 4),
                       "predicted change (log share)": np.round(f, 3),
                       "outlook": [status(k, v) for k, v in enumerate(f)]}).sort_values(
        "predicted change (log share)", ascending=False)
    # converging pairs: S3 links between communities rising faster than either community
    G = res.graph()
    links = defaultdict(lambda: np.zeros(len(T)))
    for u, v in G.edges():
        a, b = cc[u], cc[v]
        if a >= 0 and b >= 0 and a != b and not np.isnan(yr[u]) and not np.isnan(yr[v]):
            t = int(max(yr[u], yr[v])) - y0
            if t >= 0:
                links[tuple(sorted((a, b)))][t] += 1
    conv = []
    for (a, b), series in links.items():
        recent, before = series[-2 * window:].sum(), series[: -2 * window].sum()
        if recent >= 3:
            conv.append({"community A": comms[a], "community B": comms[b], f"S3 links last {2 * window} years": int(recent),
                         "S3 links before": int(before), "S3 similarity": round(float(S[a, b]), 3)})
    conv = pd.DataFrame(conv)
    if len(conv):
        conv = conv.sort_values([f"S3 links last {2 * window} years", "S3 similarity"], ascending=False).head(10)
    return {"table": pd.DataFrame(rows), "test": f"{T[t_test] + 1}-{T[t_test] + window}",
            "forecast": fc, "converging": conv, "n_train": len(ok_t) * K, "window": window}


# ===========================================================================
# Q34 knowledge-graph embeddings: TransE and RotatE
# ===========================================================================
def kg_embeddings(res, dim: int = 64, epochs: int = 60, holdout: float = 0.05, n_test: int = 1500) -> Optional[Dict]:
    """Train TransE and RotatE on paper-level triples; evaluate on held-out triples
    (filtered MRR, Hits@10); rank missing entity -> community links by composing relations
    (paper -AUTHORED_BY-> author and paper -IN_COMMUNITY-> community)."""
    torch = _torch()
    nn = torch.nn
    seed = res.config.seed
    _seed(seed)
    rng = np.random.default_rng(seed)
    nodes, kind_of, ent_index, edges = _kg_index(res)
    n = len(res.papers)
    comms = [int(c) for c in res.communities()["community"]]
    comm_idx = {}
    for c in comms:
        comm_idx[c] = len(nodes)
        nodes.append(("community", str(c)))
        kind_of.append("community")
    trip = []
    rel_names = []
    def add(rel, pairs):
        r = len(rel_names)
        rel_names.append(rel)
        trip.extend((h, r, t) for h, t in pairs)
    for kind, ed in edges.items():
        add(_REL[kind], ed)
    add("IN_COMMUNITY", [(i, comm_idx[int(c)]) for i, c in res.papers["community"].items() if pd.notna(c)])
    if res.citation_pairs:
        add("CITES", res.citation_pairs)
    T = np.array(trip, dtype=np.int64)
    N, Rn = len(nodes), len(rel_names)
    kind_arr = np.array(kind_of)
    tail_kind = {r: kind_arr[T[T[:, 1] == r][:, 2]][0] for r in range(Rn) if (T[:, 1] == r).any()}
    cand_by_kind = {k: np.where(kind_arr == k)[0] for k in set(kind_arr)}
    perm = rng.permutation(len(T))
    k_test = min(n_test, int(len(T) * holdout))
    test, train = T[perm[:k_test]], T[perm[k_test:]]
    known = defaultdict(set)
    for h, r, t in T:
        known[(h, r)].add(t)

    def train_model(kind: str):
        _seed(seed)
        if kind == "TransE":
            ent = nn.Embedding(N, dim)
            rel = nn.Embedding(Rn, dim)
            nn.init.uniform_(ent.weight, -6 / dim ** 0.5, 6 / dim ** 0.5)
            nn.init.uniform_(rel.weight, -6 / dim ** 0.5, 6 / dim ** 0.5)
            params = list(ent.parameters()) + list(rel.parameters())

            def score(h, r, t):
                return -torch.norm(ent(h) + rel(r) - ent(t), p=1, dim=-1)
        else:  # RotatE: complex entity embeddings, relations are rotations
            half = dim // 2
            ent = nn.Embedding(N, 2 * half)
            rel = nn.Embedding(Rn, half)
            nn.init.uniform_(ent.weight, -1 / half ** 0.5, 1 / half ** 0.5)
            nn.init.uniform_(rel.weight, -np.pi, np.pi)
            params = list(ent.parameters()) + list(rel.parameters())

            def score(h, r, t):
                eh, et = ent(h), ent(t)
                re_h, im_h = eh[..., :half], eh[..., half:]
                re_t, im_t = et[..., :half], et[..., half:]
                ph = rel(r)
                c, s = torch.cos(ph), torch.sin(ph)
                re = re_h * c - im_h * s - re_t
                im = re_h * s + im_h * c - im_t
                return -torch.sqrt(re ** 2 + im ** 2 + 1e-9).sum(-1)
        opt = torch.optim.Adam(params, lr=0.01)
        tt = torch.tensor(train)
        for _ in range(epochs):
            order = torch.randperm(len(tt))
            for s0 in range(0, len(tt), 4096):
                b = tt[order[s0: s0 + 4096]]
                h, r, t = b[:, 0], b[:, 1], b[:, 2]
                neg_t = torch.tensor(np.array([cand_by_kind[tail_kind[int(x)]][rng.integers(0, len(cand_by_kind[tail_kind[int(x)]]))]
                                               for x in r.numpy()]))
                pos, neg = score(h, r, t), score(h, r, neg_t)
                loss = nn.functional.softplus(-(pos + 6)).mean() + nn.functional.softplus(neg + 6).mean() \
                    if kind == "TransE" else nn.functional.softplus(-(pos + 9)).mean() + nn.functional.softplus(neg + 9).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
        return score, ent, rel

    def evaluate(score):
        ranks = defaultdict(list)
        with torch.no_grad():
            for h, r, t in test:
                cands = cand_by_kind[tail_kind[r]]
                s = score(torch.full((len(cands),), int(h)), torch.full((len(cands),), int(r)),
                          torch.tensor(cands)).numpy()
                true = s[np.searchsorted(cands, t)]
                mask = np.isin(cands, list(known[(h, r)] - {t}))
                ranks[rel_names[r]].append(1 + int(((s > true) & ~mask).sum()))
        return ranks

    def popularity_ranks():
        deg = Counter(int(t) for _, _, t in train)
        ranks = defaultdict(list)
        for h, r, t in test:
            cands = cand_by_kind[tail_kind[r]]
            s = np.array([deg.get(int(c), 0) for c in cands], dtype=float) + rng.random(len(cands)) * 1e-6
            true = s[np.searchsorted(cands, t)]
            mask = np.isin(cands, list(known[(h, r)] - {t}))
            ranks[rel_names[r]].append(1 + int(((s > true) & ~mask).sum()))
        return ranks

    results, models = {}, {}
    for name in ("TransE", "RotatE"):
        sc, ent, rel = train_model(name)
        results[name] = evaluate(sc)
        models[name] = (sc, ent, rel)
    results["Popularity (baseline)"] = popularity_ranks()
    rows = []
    for name, rk in results.items():
        allr = np.concatenate([np.array(v) for v in rk.values()])
        row = {"model": name, "MRR": round(float(np.mean(1 / allr)), 3), "Hits@10": round(float(np.mean(allr <= 10)), 3)}
        for rel_name, v in rk.items():
            row[f"Hits@10 {rel_name}"] = round(float(np.mean(np.array(v) <= 10)), 3)
        rows.append(row)
    table = pd.DataFrame(rows)
    best = max(("TransE", "RotatE"), key=lambda m: table.set_index("model").loc[m, "MRR"])
    sc, ent, rel = models[best]
    # missing entity -> community links by composition: e <-r_e- paper -r_c-> community
    r_c = rel_names.index("IN_COMMUNITY")
    top_comms = [c for c in comms][:15]
    existing = defaultdict(set)
    pc = {i: int(c) for i, c in res.papers["community"].items() if pd.notna(c)}
    for kind, ed in edges.items():
        for p, e in ed:
            if p in pc:
                existing[e].add(pc[p])
    missing = {}
    with torch.no_grad():
        W = ent.weight.detach()
        R = rel.weight.detach()
        for kind in ("author", "institution", "funder"):
            if kind not in edges:
                continue
            r_e = rel_names.index(_REL[kind])
            ids = np.array(sorted({e for _, e in edges[kind]}))
            ids = ids[[len({p for p, e in edges[kind] if e == x}) >= 3 for x in ids]] if len(ids) < 4000 else ids
            cidx = np.array([comm_idx[c] for c in top_comms])
            if best == "TransE":
                # paper ≈ e - r_e, community ≈ paper + r_c  ->  community ≈ e - r_e + r_c
                q = W[ids] - R[r_e] + R[r_c]
                S = -torch.cdist(q, W[cidx], p=1).numpy()
            else:
                half = W.shape[1] // 2
                def rot(x, ph, inv=False):
                    re, im = x[..., :half], x[..., half:]
                    c, s = torch.cos(ph), torch.sin(ph) * (-1 if inv else 1)
                    return torch.cat([re * c - im * s, re * s + im * c], -1)
                q = rot(rot(W[ids], R[r_e], inv=True), R[r_c])
                qe, ce = q, W[cidx]
                d = (qe[:, None, :half] - ce[None, :, :half]) ** 2 + (qe[:, None, half:] - ce[None, :, half:]) ** 2
                S = -torch.sqrt(d + 1e-9).sum(-1).numpy()
            rows_m = []
            for a, e in enumerate(ids):
                for b, c in enumerate(top_comms):
                    if c not in existing[e]:
                        rows_m.append({kind: nodes[e][1], "community": c, "score": float(S[a, b])})
            m = pd.DataFrame(rows_m)
            if len(m):
                m["score"] = m["score"].round(3)
                missing[kind] = m.sort_values("score", ascending=False).head(15)
    return {"table": table, "best": best, "missing": missing, "triples": len(T), "test": len(test),
            "relations": rel_names, "entities": N}
