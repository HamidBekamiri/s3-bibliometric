"""Answers for the prediction questions Q25-Q34 (link prediction and graph learning).

These questions are off by default: enable them with S3Config(prediction=True),
res.report(..., prediction=True), `s3bib report --prediction` or, in Claude,
build_report(prediction=True). A single one: res.answer("Q30", prediction=True).
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

from .questions import Answer, _name


def _comm_names(res, df: pd.DataFrame, cols=("community", "community A", "community B", "next community",
                                              "GNN community", "nearest-paper community")) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        if c in df.columns:
            df[c] = [(_name(res, v) if pd.notna(v) else "") for v in df[c]]
    return df


def _auc_chart(table: pd.DataFrame, title: str, col: str = "ROC AUC") -> Dict:
    return {"type": "bar", "title": title, "labels": [m.split(" (")[0] for m in table["method"]],
            "values": table[col].tolist()}


def _verdict(table: pd.DataFrame, s3_method: str) -> str:
    t = table.set_index("method")
    best = t["ROC AUC"].idxmax()
    rnd = t.loc["Random", "ROC AUC"]
    s3 = t.loc[s3_method, "ROC AUC"]
    lo = float(str(t.loc[s3_method, "AUC 95% CI"]).split("-")[0])
    if lo <= 0.5:
        q = "not clearly better than chance (its 95% interval includes 0.5)"
    elif s3 >= 0.8:
        q = "strong"
    elif s3 >= 0.65:
        q = "useful"
    else:
        q = "weak but above chance"
    base = [m for m in t.index if "baseline" in m.lower() or m.startswith(("Adamic", "Common"))]
    beat = [m for m in base if t.loc[m, "ROC AUC"] > s3 + 0.01]
    note = (f" A simple baseline does better on its own: {beat[0]} (AUC {t.loc[beat[0], 'ROC AUC']:.2f})."
            if beat else "")
    return (f"S3 alone reaches ROC AUC {s3:.2f} (95% CI {t.loc[s3_method, 'AUC 95% CI']}; random {rnd:.2f}): {q}."
            f"{note} Best method: {best} (AUC {t.loc[best, 'ROC AUC']:.2f}).")


def _lp_answer(qid: str, r: Optional[Dict], res, s3_method: str, what: str, forecast_title: str) -> Answer:
    a = Answer(qid, "")
    if r is None:
        a.findings.append("Too few cases in this corpus to test this prediction (needs enough candidates and "
                          "links that appear in the last 5 years).")
        return a
    a.findings.append(f"Test: {r['positives']} of {r['candidates']} candidate {what} appeared in {r['test']} "
                      f"(trained on information up to {r['split']}). {r['note']}")
    a.findings.append(_verdict(r["table"], s3_method))
    a.tables[f"Prediction accuracy ({r['test']})"] = r["table"]
    if len(r["forecast"]):
        a.tables[forecast_title] = _comm_names(res, r["forecast"])
    a.chart = _auc_chart(r["table"], "ROC AUC by method (0.5 = chance)")
    return a


# --------------------------------------------------------------------------
# Link prediction
# --------------------------------------------------------------------------
def q_gaps_close(res) -> Answer:
    from .linkpred import gap_closing
    r = gap_closing(res)
    a = _lp_answer("Q25", r, res, "S3 centroid similarity", "community pairs",
                   "Forecast: similar community pairs that still do not cite each other")
    if r is not None:
        a.findings.append("Communities are the final S3 communities of the whole run, so this test knows the "
                          "partition; the forecast lists the gaps most likely to close next.")
    return a


def q_concepts_enter(res) -> Answer:
    from .linkpred import concept_entry
    return _lp_answer("Q26", concept_entry(res), res, "S3 similarity (keyword ↔ community)",
                      "(keyword, community) entries", "Forecast: concepts most likely to enter each community next")


def q_authors_move(res) -> Answer:
    from .linkpred import entity_entry
    a = _lp_answer("Q27", entity_entry(res, "author"), res, "S3 similarity (author ↔ community)",
                   "(author, community) moves", "Forecast: the community each author is most likely to enter next")
    a.findings.append("Scopus abbreviates author names, so different people can share a name.")
    return a


def q_coauthors(res) -> Answer:
    from .linkpred import coauthorship
    r = coauthorship(res)
    a = Answer("Q28", "")
    if r is None:
        a.findings.append("Too few repeat authors to test co-authorship prediction.")
        return a
    a.findings.append(f"Test: {r['positives']} of {r['candidates']} candidate author pairs co-authored in "
                      f"{r['test']}. {r['note']}")
    a.findings.append(_verdict(r["table"], "S3 similarity (author centroids)"))
    t = r["table"].set_index("method")
    a.findings.append(f"Classic network score (Adamic-Adar): AUC {t.loc['Adamic-Adar (co-author network)', 'ROC AUC']:.2f}; "
                      "S3 finds pairs who work on the same content but share no co-authors yet.")
    a.tables[f"Prediction accuracy ({r['test']})"] = r["table"]
    if len(r["forecast"]):
        a.tables["Forecast: author pairs with the most similar work who never co-authored"] = r["forecast"]
    a.chart = _auc_chart(r["table"], "ROC AUC by method (0.5 = chance)")
    return a


def q_orgs_enter(res) -> Answer:
    from .linkpred import entity_entry
    a = Answer("Q29", "")
    for kind in ("institution", "funder"):
        r = entity_entry(res, kind)
        if r is None:
            a.findings.append(f"{kind.capitalize()}s: too few cases to test.")
            continue
        a.findings.append(f"{kind.capitalize()}s: {r['positives']} of {r['candidates']} candidate entries appeared "
                          f"in {r['test']}. " + _verdict(r["table"], f"S3 similarity ({kind} ↔ community)"))
        a.tables[f"{kind.capitalize()}s: prediction accuracy ({r['test']})"] = r["table"]
        if len(r["forecast"]):
            a.tables[f"Forecast: the community each {kind} is most likely to enter next"] = _comm_names(res, r["forecast"])
        if a.chart is None:
            a.chart = _auc_chart(r["table"], f"{kind.capitalize()}s: ROC AUC by method (0.5 = chance)")
    a.findings.append("Institution names come from the affiliation text; spelling variants and department "
                      "names (e.g. 'College of Business') are not merged.")
    return a


# --------------------------------------------------------------------------
# Graph learning
# --------------------------------------------------------------------------
def q_gnn_classify(res) -> Answer:
    from .gnn import node_classification
    a = Answer("Q30", "")
    r = node_classification(res)
    if r is None:
        a.findings.append("Too few clustered papers to train a community classifier.")
        return a
    t = r["table"].set_index("method")
    best = t["accuracy"].idxmax()
    sage = "GraphSAGE (text + citations + S3 links)"
    a.findings.append(f"Test: {r['n_test']} clustered papers in {r['classes']} communities were treated like outliers "
                      f"(their S3 links above the threshold hidden) and their community predicted.")
    a.findings.append(f"Best: {best} (accuracy {t.loc[best, 'accuracy']:.0%}). GraphSAGE: {t.loc[sage, 'accuracy']:.0%}; "
                      f"the current nearest-paper rule: {t.loc['Nearest paper by S3 (current rule)', 'accuracy']:.0%}.")
    if r["by_citations"]:
        b = r["by_citations"]
        a.findings.append(f"GraphSAGE is right for {b['with citations']:.0%} of papers with in-corpus citations and "
                          f"{b['without citations']:.0%} of papers without: citations help place a paper.")
    a.findings.append(f"Applied to the {len(r['outliers'])} real outliers: {r['confident']:.0%} get a community with "
                      f"confidence ≥ 0.5; GraphSAGE and the nearest-paper rule agree for {r['agreement']:.0%}. Low "
                      "confidence means the paper does not fit any of the main communities.")
    a.tables["Classifier accuracy (papers treated as outliers)"] = r["table"]
    a.tables["Most-cited outliers and their predicted community"] = _comm_names(res, r["outliers"].head(25))
    a.chart = {"type": "bar", "title": "Accuracy by method", "labels": [m.split(" (")[0] for m in r["table"]["method"]],
               "values": r["table"]["accuracy"].tolist()}
    return a


def q_gnn_impact(res) -> Answer:
    from .gnn import impact_regression
    a = Answer("Q31", "")
    r = impact_regression(res)
    if r is None:
        a.findings.append("Too few papers with a year to test impact prediction.")
        return a
    t = r["table"].set_index("method")
    c = t.loc["Content: text embedding (ridge)", "Spearman ρ"]
    p = t.loc["Position: network features (boosted trees)", "Spearman ρ"]
    g = t.loc["GraphSAGE (text + earlier links)", "Spearman ρ"]
    a.findings.append(f"Trained on papers up to {r['train'][1:]}, tested on the {r['n_test']} papers of {r['test']}: "
                      f"predicting citations per year from information available at publication (text, and links to "
                      f"earlier papers only).")
    a.findings.append(f"Content predicts impact with Spearman ρ = {c:.2f}, network position with ρ = {p:.2f}, the "
                      f"GNN (both) with ρ = {g:.2f}: impact here is driven "
                      + ("more by content than by position." if c > p + 0.05 else
                         "more by position than by content." if p > c + 0.05 else "by content and position about equally.")
                      + " R² stays low: most variation in citations is not predictable at publication.")
    a.tables[f"Impact prediction ({r['test']})"] = r["table"]
    a.tables["Network features: correlation with impact (training papers)"] = r["feature_corr"].rename("|r|").reset_index().rename(columns={"index": "feature"})
    a.tables["Forecast: newest papers with the highest predicted impact"] = r["forecast"]
    a.chart = {"type": "bar", "title": "Spearman ρ between predicted and real citations per year",
               "labels": [m.split(" (")[0].split(":")[0] if ":" in m else m.split(" (")[0] for m in r["table"]["method"]],
               "values": r["table"]["Spearman ρ"].tolist()}
    return a


def q_gnn_hetero(res) -> Answer:
    from .gnn import find_experts, hetero_embeddings
    a = Answer("Q32", "")
    r = hetero_embeddings(res)
    if r is None or r["table"].empty:
        a.findings.append("Not enough authors, institutions or funders with at least 2 papers.")
        return a
    a.findings.append(f"An R-GCN learns embeddings for {r['n_entities']} authors, institutions, funders, keywords and "
                      f"journals from the knowledge graph ({r['relations']} relation types, including citations "
                      "and S3 links).")
    t = r["table"]
    better = (t["R-GCN AUC"] > t["S3 baseline AUC"]).sum()
    a.findings.append(f"On held-out links the R-GCN ranks the true author/institution/funder of a paper above a random "
                      f"one with AUC {t['R-GCN AUC'].min():.2f}-{t['R-GCN AUC'].max():.2f}, better than the text-only "
                      f"baseline in {better} of {len(t)} link types.")
    a.tables["Held-out link accuracy"] = t
    rows = []
    E = res.embeddings
    from .questions import _main_comms
    for c in _main_comms(res, 6):
        v = E[(res.papers["community"] == c).fillna(False).to_numpy()].mean(axis=0)
        v = v / (np.linalg.norm(v) or 1.0)
        for kind in ("author", "institution", "funder"):
            if kind in r["R"]:
                top = find_experts(res, v, kind, 3)
                rows.append({"community": _name(res, c), "kind": kind, "best fits": "; ".join(top[kind])})
    if rows:
        a.tables["Who fits each community best (expertise search with the community as query)"] = pd.DataFrame(rows)
    a.findings.append("Use res.find_experts('your topic', kind='author') for expertise search or reviewer finding; "
                      "kind can be author, institution, funder, keyword or journal.")
    a.chart = {"type": "bar", "title": "Held-out link AUC: R-GCN", "labels": t["link"].tolist(),
               "values": t["R-GCN AUC"].tolist()}
    return a


def q_gnn_temporal(res) -> Answer:
    from .gnn import community_forecast
    a = Answer("Q33", "")
    r = community_forecast(res)
    if r is None:
        a.findings.append("Too few years with enough papers to learn community dynamics.")
        return a
    t = r["table"].set_index("method")
    g = t.loc["Temporal GNN (GCN + GRU over communities)"]
    nc = t.loc["No change (baseline)"]
    a.findings.append(f"Each community's share of papers in {r['window']}-year windows is predicted from its recent "
                      f"history and that of its most similar communities (a GCN over the community graph and a GRU over "
                      f"time). Test: the change in {r['test']}.")
    a.findings.append(f"Temporal GNN: Spearman ρ {g['Spearman ρ']:.2f}, direction right for {g['direction correct']:.0%} "
                      f"of communities (no-change baseline: ρ {nc['Spearman ρ']:.2f}). With only "
                      f"{len(r['forecast'])} communities this is a rough signal, not a precise forecast.")
    fc = _comm_names(res, r["forecast"])
    grow = fc[fc["outlook"].isin(["growing", "emerging"])]["community"].tolist()[:4]
    dec = fc[fc["outlook"] == "declining"]["community"].tolist()[:4]
    if grow:
        a.findings.append("Forecast, next window: growing " + ", ".join(grow) + "."
                          + (" Declining: " + ", ".join(dec) + "." if dec else ""))
    a.tables[f"Forecast accuracy ({r['test']})"] = r["table"]
    a.tables["Forecast: change in share over the next window"] = fc
    if len(r["converging"]):
        a.tables["Converging community pairs (S3 links between them rising): possible merges"] = _comm_names(res, r["converging"])
    a.chart = {"type": "bar", "title": "Predicted change in share, next window (log ratio)",
               "labels": fc["community"].tolist(), "values": fc["predicted change (log share)"].tolist()}
    return a


def q_gnn_kg(res) -> Answer:
    from .gnn import kg_embeddings
    a = Answer("Q34", "")
    r = kg_embeddings(res)
    if r is None:
        a.findings.append("The knowledge graph is too small to train embeddings.")
        return a
    t = r["table"].set_index("model")
    a.findings.append(f"TransE and RotatE were trained on {r['triples']} knowledge-graph triples "
                      f"({', '.join(r['relations'])}); {r['test']} held-out triples test them (filtered ranking).")
    a.findings.append(f"MRR: TransE {t.loc['TransE', 'MRR']:.3f}, RotatE {t.loc['RotatE', 'MRR']:.3f}, popularity "
                      f"baseline {t.loc['Popularity (baseline)', 'MRR']:.3f}. Hits@10: {t.loc[r['best'], 'Hits@10']:.0%} "
                      f"for {r['best']} vs {t.loc['Popularity (baseline)', 'Hits@10']:.0%} for the baseline.")
    a.findings.append(f"Missing links are ranked with {r['best']} by composing relations (entity ← paper → community): "
                      "authors, institutions and funders that fit a community they have not published or funded in yet.")
    a.tables["Link prediction accuracy (held-out triples)"] = r["table"]
    for kind, m in r["missing"].items():
        a.tables[f"Missing links: {kind} → community"] = _comm_names(res, m)
    a.chart = {"type": "bar", "title": "MRR by model", "labels": r["table"]["model"].tolist(),
               "values": r["table"]["MRR"].tolist()}
    return a
