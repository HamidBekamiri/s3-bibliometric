"""Tests on real data: 109 OpenAlex records (CC0) on absorptive capacity and work group
diversity, with their real all-mpnet-base-v2 embeddings precomputed, so no model download
is needed. Rebuild the data with tests/data/build_openalex_sample.py."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import s3bib
from s3bib import pipeline
from s3bib.pipeline import _norm_text

DATA = Path(__file__).parent / "data"
CL = "W2108795964"       # Cohen & Levinthal (1990), Absorptive capacity
CL_REPRINT = "W3125646472"  # the same paper reprinted as a book chapter
DIV = "W2159012497"      # van Knippenberg et al. (2004), Work group diversity
QUIET = dict(progress=lambda s, p: None)


@pytest.fixture(scope="session")
def lookup():
    z = np.load(DATA / "openalex_sample_mpnet.npz")
    return dict(zip(z["text"].tolist(), z["emb"]))


@pytest.fixture(autouse=True)
def real_embeddings(monkeypatch, lookup):
    """Serve the precomputed mpnet embeddings instead of running the model."""
    monkeypatch.setattr(pipeline, "embed_texts", lambda texts, *a, **k: np.array([lookup[t] for t in texts]))


@pytest.fixture(scope="session")
def df():
    return pd.read_csv(DATA / "openalex_sample.csv")


@pytest.fixture(scope="session")
def res(df):
    mp = pytest.MonkeyPatch()
    z = np.load(DATA / "openalex_sample_mpnet.npz")
    lut = dict(zip(z["text"].tolist(), z["emb"]))
    mp.setattr(pipeline, "embed_texts", lambda texts, *a, **k: np.array([lut[t] for t in texts]))
    r = s3bib.run_s3(df, **QUIET)
    mp.undo()
    return r


def _topic(df):
    """Which seminal paper each record cites (from its real OpenAlex reference list)."""
    refs = df.set_index("EID")["Referenced IDs"].fillna("")
    return {e: ("AC" if CL in r else "") + ("DIV" if DIV in r else "") for e, r in refs.items()}


def test_communities_follow_the_two_literatures(res, df, tmp_path):
    comms = res.communities()
    assert len(comms) >= 2
    assert res.best["sii"] == max(r["sii"] for r in res.scan)
    # papers in a community cite the same seminal paper: absorptive capacity or diversity
    topic = _topic(df)
    P = res.papers.dropna(subset=["community"])
    mixed = sum(1 for _, g in P.groupby("community")
                if {topic[e] for e in g["eid"]} - {""} >= {"AC", "DIV"})
    assert mixed <= 1
    assert len(res.outliers()) == len(res.papers) - len(P)

    res.labels = {0: "A"}
    res.save(tmp_path / "run")
    again = s3bib.S3Result.load(tmp_path / "run")
    assert again.labels == {0: "A"}
    assert again.papers["community"].equals(res.papers["community"])
    assert "papers_csv" in res.export(tmp_path / "out")
    res.labels = {}


def test_title_matching_recovers_real_citations(res, df):
    """Citation links found from the References text agree with OpenAlex reference lists."""
    eids = res.papers["eid"].tolist()
    ids = df.set_index("EID")["Referenced IDs"].fillna("")
    real = {(i, eids.index(r)) for i, e in enumerate(eids) for r in ids[e].split("; ") if r in eids}
    found = set(map(tuple, res.citation_pairs))
    assert len(real & found) / len(real) >= 0.95
    assert found <= real  # same-title papers are told apart by the year in the reference
    # every link found cites a paper whose title really is in the citing paper's references
    titles = res.papers["title"].map(_norm_text).tolist()
    real_titles = {(i, titles[j]) for i, j in real}
    assert all((i, titles[j]) in real_titles for i, j in found)


def test_requires_title_and_abstract(df):
    with pytest.raises(ValueError):
        s3bib.load_papers(df[["Title"]])


def test_every_question_answers(res, tmp_path):
    qt = res.questions()
    assert list(qt["id"]) == [q.qid for q in s3bib.QUESTIONS if not q.prediction]  # prediction hidden
    assert list(res.questions(prediction=True)["id"]) == [q.qid for q in s3bib.QUESTIONS]
    assert qt["available"].all()
    for qid in qt["id"]:
        a = res.answer(qid)
        assert a.findings, qid
        a.to_dict()
    html = res.report(tmp_path / "r.html").read_text()
    assert "Q1" in html and "Q23" in html and "Could not compute" not in html
    assert 'class="ktree"' in html  # knowledge tree under Q13
    assert "force-graph@1.51.0" in html and 's3FG("' in html  # network drawn as in the S3 web app
    assert 'id="Q24"' not in html and 'class="tv"' in html  # prediction off; tables open as charts


def test_missing_columns_are_skipped(df):
    r = s3bib.run_s3(df.drop(columns=["Funding Details"]), **QUIET)
    assert r.answer("Q20").findings[0].startswith("Not available")


def test_split_refs_handles_both_scopus_formats():
    from s3bib.questions import _split_refs
    new = ("van Knippenberg D.; De Dreu C.K.W.; Homan A.C., Work group diversity and group performance: "
           "An integrative model and research agenda, Journal of Applied Psychology, 89, 6, pp. 1008-1022, (2004); "
           "Cohen W.M.; Levinthal D.A., Absorptive capacity: A new perspective on learning and innovation, "
           "Administrative Science Quarterly, 35, 1, pp. 128-152, (1990)")
    old = ("Cohen W.M., Levinthal D.A., Absorptive capacity: A new perspective on learning and innovation, "
           "Administrative Science Quarterly, 35, 1, pp. 128-152, (1990); Zahra S.A., George G., Absorptive "
           "capacity: A review, reconceptualization, and extension, Academy of Management Review, 27, 2, "
           "pp. 185-203, (2002)")
    parts = _split_refs(new)
    assert len(parts) == 2 and parts[0].startswith("van Knippenberg D.; De Dreu C.K.W.; Homan A.C.")
    parts = _split_refs(old)
    assert len(parts) == 2 and parts[1].startswith("Zahra S.A., George G.") and parts[0].endswith("(1990)")


def test_time_questions_skipped_for_single_year(df):
    r = s3bib.run_s3(df.assign(Year=2023), **QUIET)
    for qid in ("Q13", "Q23"):
        assert "fewer than 3 publication years" in r.answer(qid).findings[0]


def test_concept_trends_leave_out_near_universal_keywords(res):
    a = res.answer("Q14")
    t = a.tables["Concept trends (RQ5)"]
    common = set(t.loc[t["share of papers"] >= 0.10, "keyword"])
    text = " ".join(a.findings)
    assert not any(f"{k} (" in text or f"“{k}”" in text for k in common)


def test_fixed_threshold_and_methods(df, res):
    fixed = s3bib.run_s3(df, s3bib.S3Config(threshold=0.8), **QUIET)
    assert fixed.best["threshold"] == 0.8 and fixed.best["fixed"]
    assert "fixed" in fixed.answer("Q1").findings[0]
    scan = {r["threshold"]: r for r in res.scan}
    assert fixed.best["n_papers"] == scan[0.8]["n_papers"]
    lv = s3bib.run_s3(df, s3bib.S3Config(community_method="louvain"), **QUIET)
    assert lv.config.community_method == "louvain" and len(lv.communities()) >= 2


def test_citation_metrics_count_whole_corpus(res):
    m = res.citation_metrics()
    assert {"inward_citations", "outward_citations", "global_citations"} <= set(m.columns)
    assert m["local_citations"].equals(m["within_citations"] + m["inward_citations"])
    assert abs(m["mi"].sum() - m["local_citations"].sum() / res.mi_denominator() * 100) < 1e-9
    assert res.top_terms(0, 3)  # c-TF-IDF terms


def test_runs_are_reproducible(df, res):
    b = s3bib.run_s3(df, **QUIET)
    assert res.papers["community"].equals(b.papers["community"]) and res.scan == b.scan
    for qid, table in [("Q7", "Most central papers per community"), ("Q12", "Bridging papers")]:
        ta, tb = res.answer(qid).tables.get(table), b.answer(qid).tables.get(table)
        assert (ta is None and tb is None) or ta.equals(tb)


def test_empty_and_duplicate_records_removed(df):
    # a record exported twice (same EID), the same paper under a second id, and an empty record
    extra = pd.concat([df.iloc[[5]], df.iloc[[7]].assign(EID="second-id", **{"Cited by": 0}),
                       df.iloc[[9]].assign(Title="", Abstract="", EID="empty")], ignore_index=True)
    r = s3bib.run_s3(pd.concat([df, extra], ignore_index=True), **QUIET)
    reasons = r.removed["reason"].tolist()
    assert len(r.papers) == len(df)
    assert "no title and no abstract" in reasons
    assert sum(x.startswith("exact duplicate") for x in reasons) == 2
    assert "second-id" in r.removed["eid"].tolist()  # the less-cited copy goes
    kept = s3bib.run_s3(pd.concat([df, extra], ignore_index=True), s3bib.S3Config(dedupe=False), **QUIET)
    assert len(kept.papers) == len(df) + 2  # the empty record is always dropped


def test_near_duplicates_by_s3(df, lookup):
    """The 1990 article and its book-chapter reprint are one paper indexed twice."""
    papers = s3bib.load_papers(df)
    pair = papers[papers["eid"].isin([CL, CL_REPRINT])].reset_index(drop=True)
    emb = np.array([lookup[t] for t in pair["text"]])
    s3 = float(emb[0] @ emb[1])
    drop, removed = pipeline.near_duplicates(pair, emb, threshold=s3 - 0.01)
    assert len(drop) == 1 and removed[0]["eid"] == CL_REPRINT and removed[0]["kept_eid"] == CL
    assert pipeline.near_duplicates(pair, emb, threshold=0.98)[0] == []  # default keeps both


def test_graphs_show_community_names(res):
    from s3bib.questions import _display_name, _network_spec
    from s3bib.ktree import ktree_spec
    terms = [t for t, _ in res.top_terms(0, 3)]
    assert _display_name(res, 0) == "C0 " + " · ".join(terms)  # unlabelled: top c-TF-IDF terms
    res.labels = {0: "Absorptive capacity"}
    try:
        assert _network_spec(res)["legend"][0]["name"] == "C0 Absorptive capacity"
        assert ktree_spec(res)["names"][0] == "C0 Absorptive capacity"
    finally:
        res.labels = {}


def test_knowledge_tree(res):
    kt = res.knowledge_tree(window_years=5)
    shown = res.papers[res.papers["community"].isin(kt["communities"]) & res.papers["year"].notna()]
    assert kt["windows"][-1].startswith(str(int(shown["year"].max())))  # last window starts at the latest year
    assert kt["nodes"]["papers"].sum() == len(shown)
    L = kt["links"]
    assert (L["from_window"] < L["to_window"]).all()
    assert 0 < L["citations"].sum() <= len(res.citation_pairs)
    assert 0 < L["similarity_links"].sum() <= res.graph().number_of_edges()


def test_model_presets():
    from s3bib.models import MPNET_THRESHOLDS, WIDE_THRESHOLDS
    assert s3bib.S3Config().model_name == s3bib.DEFAULT_MODEL
    assert s3bib.S3Config().thresholds == MPNET_THRESHOLDS
    c = s3bib.S3Config(model_name="specter")
    assert c.model_name == "sentence-transformers/allenai-specter" and c.thresholds == WIDE_THRESHOLDS
    assert s3bib.S3Config(model_name="clip").model_name.endswith("clip-ViT-B-32")


def test_scopus_enrichment_with_mocked_api(df, tmp_path):
    """The mocked Scopus API answers with each paper's real OpenAlex reference list."""
    r = s3bib.run_s3(df, **QUIET)
    eids = r.papers["eid"].tolist()
    refs = df.set_index("EID")["Referenced IDs"].fillna("")

    class Resp:
        def __init__(self, payload):
            self.payload, self.status_code, self.headers = payload, 200, {}
        def json(self):
            return self.payload
        def raise_for_status(self):
            pass

    class Session:
        calls = 0
        def get(self, url, headers, params, timeout):
            Session.calls += 1
            assert headers["X-ELS-APIKey"] == "test-key" and params["view"] == "REF"
            cited = [x for x in refs[url.rsplit("/", 1)[-1]].split("; ") if x]
            return Resp({"abstracts-retrieval-response": {"references": {
                "@total-references": str(len(cited)), "reference": [{"scopus-eid": x} for x in cited]}}})

    stats = s3bib.enrich_citations(r, api_key="test-key", session=Session(), cache_dir=tmp_path, pause=0)
    real = {(i, eids.index(x)) for i, e in enumerate(eids) for x in refs[e].split("; ") if x in eids}
    assert r.citation_source == "scopus_api" and stats["failed"] == 0
    assert set(map(tuple, r.citation_pairs)) == real
    assert "Scopus API (exact)" in r.answer("Q10").findings[0]
    n = Session.calls
    s3bib.enrich_citations(r, api_key="test-key", session=Session(), cache_dir=tmp_path, pause=0)
    assert Session.calls == n  # second run comes from the cache


def _mock_scopus(df):
    """A mocked Scopus API that answers with each paper's real OpenAlex reference list."""
    refs = df.set_index("EID")["Referenced IDs"].fillna("")

    class Resp:
        def __init__(self, payload):
            self.payload, self.status_code, self.headers = payload, 200, {}
        def json(self):
            return self.payload
        def raise_for_status(self):
            pass

    class Session:
        def get(self, url, headers, params, timeout):
            cited = [x for x in refs[url.rsplit("/", 1)[-1]].split("; ") if x]
            return Resp({"abstracts-retrieval-response": {"references": {
                "@total-references": str(len(cited)), "reference": [{"scopus-eid": x} for x in cited]}}})
    return Session(), refs


def test_coupling_from_scopus_reference_ids(df, tmp_path):
    """After enrich_citations, coupling uses each paper's cited IDs (complete, exact), the map is
    rebuilt when it uses citations, and the report says where the references came from."""
    from s3bib.hybrid import reference_matrix, reference_source
    r = s3bib.run_s3(df, s3bib.S3Config(map_weights="text+coupling"), **QUIET)
    r.labels = {0: "a label"}
    session, refs = _mock_scopus(df)
    stats = s3bib.enrich_citations(r, api_key="test-key", session=session, cache_dir=tmp_path, pause=0)
    real = {e: [x for x in refs[e].split("; ") if x] for e in r.papers["eid"]}
    with_ids = sum(1 for v in real.values() if v)
    assert stats["papers_with_reference_ids"] == with_ids and stats["share_of_references_with_scopus_id"] == 1.0
    assert stats["coupling_reference_source"]["scopus_ids"] == with_ids and stats["map_rebuilt"]
    assert r.labels == {} and r.best is not None and r.config.map_weights == {"s3": 0.75, "coupling": 0.25}
    # shared references counted on the real cited IDs
    R = reference_matrix(r.papers).tocsr()
    shared = (R @ R.T).toarray()
    eids = r.papers["eid"].tolist()
    for i in range(0, len(eids), 7):
        for j in range(i + 1, len(eids), 11):
            if real[eids[i]] and real[eids[j]]:
                assert shared[i, j] == len(set(real[eids[i]]) & set(real[eids[j]]))
    assert reference_source(r.papers)["scopus_ids"] == with_ids
    assert any("cited Scopus IDs" in f for f in r.answer("Q4").findings)
    back = s3bib.S3Result.load(r.save(tmp_path / "run"))
    assert back.papers["reference_eids"].tolist() == r.papers["reference_eids"].tolist()
    assert "retrieved from the Scopus API as Scopus IDs" in r.report(tmp_path / "r.html").read_text()

    # a map chosen with "auto" is chosen again on the exact links
    auto = s3bib.run_s3(df, s3bib.S3Config(map_weights="auto"), **QUIET)
    s3bib.enrich_citations(auto, api_key="test-key", session=session, cache_dir=tmp_path, pause=0)
    assert auto.map_choice["auto"] and auto.citation_source == "scopus_api"
    assert auto.map_choice["n_links"] == len(auto.citation_pairs)
    # caches written by earlier versions (a plain list of EIDs) still load
    import json
    from s3bib.scopus import fetch_references
    (tmp_path / "old").mkdir()
    (tmp_path / "old" / "X.json").write_text(json.dumps(["2-s2.0-1"]))
    assert fetch_references("X", "k", cache_dir=tmp_path / "old") == (["2-s2.0-1"], None)


def test_scopus_access_error(res, monkeypatch):
    class Denied:
        def get(self, *a, **k):
            class R:
                status_code, headers = 403, {}
            return R()
    monkeypatch.delenv("SCOPUS_API_KEY", raising=False)
    with pytest.raises(s3bib.ScopusAccessError):
        s3bib.enrich_citations(res)  # no key at all
    with pytest.raises(s3bib.ScopusAccessError):
        s3bib.enrich_citations(res, api_key="x", session=Denied(), cache_dir=None, pause=0)


def test_network_levels(res, df, tmp_path):
    from s3bib import networks as N
    # one-mode: the S3 graph and the in-corpus citations, with typed paper ids
    nodes, edges = N.one_mode(res)
    assert set(nodes["type"]) == {"paper"} and len(nodes) == len(res.papers)
    assert (edges["type"] == "SIMILAR_TO").sum() == res.graph().number_of_edges()
    assert (edges["type"] == "CITES").sum() == len(res.citation_pairs)
    cites = edges[edges["type"] == "CITES"]
    assert f"paper:{CL}" in set(cites["target"])  # Cohen & Levinthal is cited inside the corpus
    sim = edges[edges["type"] == "SIMILAR_TO"]
    year = dict(zip(nodes["id"], nodes["year"]))
    assert all(year[a] <= year[b] for a, b in zip(sim["source"], sim["target"]))  # earlier -> later

    # two-mode: paper-author ties match the Authors column
    n2, e2 = N.two_mode(res, "author")
    n_auth = sum(len(set(a.split("; "))) for a in df["Authors"].dropna())
    assert len(e2) == n_auth and set(n2["type"]) == {"paper", "author"}
    assert set(e2["type"]) == {"AUTHORED_BY"}
    _, e3 = N.two_mode(res, "reference")
    assert len(e3) > 0

    # knowledge graph: communities and entity types together
    kn, ke = N.knowledge_graph(res)
    assert {"paper", "community", "author", "keyword", "journal", "institution", "country", "funder"} <= set(kn["type"])
    assert {"SIMILAR_TO", "CITES", "IN_COMMUNITY", "AUTHORED_BY", "LOCATED_IN", "FUNDED_BY"} <= set(ke["type"])
    assert (ke["type"] == "IN_COMMUNITY").sum() == res.papers["community"].notna().sum()
    assert set(ke["source"]) | set(ke["target"]) <= set(kn["id"])  # no dangling ties

    G = res.network("paper-journal")
    assert G.number_of_edges() == res.papers["source"].notna().sum()
    with pytest.raises(ValueError):
        res.network("paper-unicorn")
    assert list(res.networks()["level"])[0] == "one-mode"

    files = res.export_networks(tmp_path / "net")
    import networkx as nx
    kg = nx.read_graphml(files["knowledge_graph_graphml"])
    assert kg.number_of_nodes() == len(kn)
    assert "two_mode_paper_author_graphml" in files and not any("reference" in k for k in files)
    files = res.export_networks(tmp_path / "ref", formats=("csv",), two_mode_kinds=["reference"])
    assert "two_mode_paper_reference_edges_csv" in files


def test_citation_prediction(res):
    """Q24 on real data: S3 similarity ranks the earlier papers that new papers cite."""
    from s3bib.predict import citation_prediction
    r = citation_prediction(res)
    t = r["table"].set_index("method")
    assert r["test"] == "2019-2023" and r["new_papers"] >= 10
    s3 = t.loc["S3 similarity (text only)"]
    assert s3["ROC AUC"] > 0.7 > t.loc["Random", "ROC AUC"]
    assert {"forecast", "reference point", "baseline"} == set(t["type"])
    assert r["table"].equals(citation_prediction(res)["table"])  # seeded: same result every run
    a = res.answer("Q24", prediction=True)
    assert "ROC AUC" in a.findings[1] and a.chart["type"] == "line"



def test_prediction_questions(res, tmp_path):
    """Q25-Q34 run on real data (or say the corpus is too small) and appear only when enabled."""
    from s3bib.questions import QUESTIONS
    pred = [q.qid for q in QUESTIONS if q.prediction]
    assert pred == [f"Q{i}" for i in range(24, 35)]
    assert res.config.prediction is False  # off by default
    for qid in pred:
        off = res.answer(qid)
        assert off.findings[0].startswith("Prediction questions (Q24-Q34) are off") and not off.tables
    with pytest.raises(RuntimeError):
        res.find_experts("absorptive capacity")
    assert not any(q in res.report(tmp_path / "d.html").read_text() for q in ('id="Q24"', 'id="Q30"'))
    for qid in pred[1:]:
        a = res.answer(qid, prediction=True)  # switched on for one call
        assert a.findings and not a.findings[0].startswith("Prediction questions"), qid
        a.to_dict()
    html = res.report(tmp_path / "p.html", prediction=True).read_text()
    assert 'id="Q30"' in html and "Could not compute" not in html


def test_prediction_switched_on_in_config(df):
    r = s3bib.run_s3(df, s3bib.S3Config(prediction=True), **QUIET)
    assert "Q30" in set(r.questions()["id"])
    assert not r.answer("Q24").findings[0].startswith("Prediction questions")


def test_cli_lists_prediction_only_when_asked(capsys):
    from s3bib.cli import main
    main(["questions"])
    plain = capsys.readouterr().out
    main(["questions", "--prediction"])
    full = capsys.readouterr().out
    assert "Q23" in plain and "Q24" not in plain and "Q34" in full


def test_link_prediction_evaluation():
    from s3bib.linkpred import evaluate
    y = np.array([1, 1, 0, 0, 0, 1, 0, 0])
    t = evaluate({"perfect": y.astype(float), "reversed": 1.0 - y}, y, k=3).set_index("method")
    assert t.loc["perfect", "ROC AUC"] == 1.0 and t.loc["reversed", "ROC AUC"] == 0.0
    assert t.loc["perfect", "precision@3"] == 1.0


def test_gnn_models_on_real_data(res):
    from s3bib import gnn
    k = gnn.kg_embeddings(res, epochs=5)
    t = k["table"].set_index("model")
    assert {"TransE", "RotatE", "Popularity (baseline)"} <= set(t.index) and (t["MRR"] > 0).all()
    h = gnn.hetero_embeddings(res, epochs=20)
    assert h is not None and len(h["names"]) == h["n_entities"]
    top = gnn.find_experts(res, res.embeddings[0], "author", 5)
    assert len(top) == 5


def test_four_map_options(df, tmp_path):
    from s3bib.hybrid import MAP_OPTIONS, map_label, resolve_weights
    assert list(MAP_OPTIONS) == ["text", "text+coupling", "text+cocitation", "text+direct"]
    assert resolve_weights("text") is None and resolve_weights(None) is None and resolve_weights("s3") is None
    assert resolve_weights("text+coupling") == {"s3": 0.75, "coupling": 0.25}
    assert resolve_weights("coc") == {"s3": 0.75, "cocitation": 0.25}
    assert resolve_weights("text+direct") == {"s3": 0.75, "direct": 0.25}
    assert resolve_weights("s3=1,coupling=1") == {"s3": 0.5, "coupling": 0.5}
    assert resolve_weights("hybrid") == {"s3": 0.25, "coupling": 0.25, "direct": 0.25, "cocitation": 0.25}
    with pytest.raises(ValueError):
        resolve_weights({"unicorn": 1})
    with pytest.raises(ValueError):
        resolve_weights("unicorn")
    assert map_label(resolve_weights("text+cocitation")) == "75% text (S3) + 25% co-citation"
    plain = s3bib.run_s3(df, **QUIET)
    for opt in ("text+coupling", "text+cocitation", "text+direct"):
        r = s3bib.run_s3(df, s3bib.S3Config(map_weights=opt), **QUIET)
        # each threshold links about as many pairs as the matching S3 threshold
        assert [x["n_edges"] for x in r.scan][:3] == pytest.approx([x["n_edges"] for x in plain.scan][:3], rel=0.15)
        assert r.graph().number_of_edges() > 0
    # every result follows the chosen map: header, Q1, network map
    html = r.report(tmp_path / "m.html").read_text()
    assert "map: 75% text (S3) + 25% direct citation" in html
    assert "the map score (75% text (S3) + 25% direct citation)" in r.answer("Q1").findings[0]
    assert "map: text (S3)" in plain.report(tmp_path / "p.html").read_text()


def test_sections_closed_and_word_cloud(res, tmp_path):
    html = res.report(tmp_path / "w.html").read_text()
    assert "<details open" not in html and "<details>" in html  # everything opens on click
    wg = res.answer("Q6").chart  # one word cloud per main community, 30 words by default
    assert wg["type"] == "wordgrid" and wg["n"] == 30
    assert len(wg["clouds"]) == min(10, len(res.communities())) and all(len(c["words"]) <= 50 for c in wg["clouds"])
    assert 's3WG("' in html and html.count('class="wg-cell"') == len(wg["clouds"])
    assert '<option value="30" selected>30 words</option>' in html


def test_strict_citation_matching(df):
    """A short title only counts when the reference also has its year and first author."""
    from s3bib.pipeline import citation_edges, load_papers
    P = load_papers(df)
    i = int(P.index[P["eid"] == DIV][0])
    P.loc[len(P)] = P.loc[i]  # a real paper cited under a generic short title
    P.at[len(P) - 1, "title"], P.at[len(P) - 1, "eid"] = "Introduction", "intro"
    P.at[0, "references"] = "Smith J., Introduction to econometrics, (2017)"
    lenient = {(a, b) for a, b in citation_edges(P) if b == len(P) - 1}
    strict = {(a, b) for a, b in citation_edges(P, strict=True) if b == len(P) - 1}
    assert (0, len(P) - 1) in lenient and (0, len(P) - 1) not in strict


def test_layer_overlap_homophily_direction(res, tmp_path):
    """Q35-Q37 on real data: multiplex overlap, homophily and citation direction."""
    from s3bib.layers import assortativity, flow_direction, layers, overlap, pagerank
    L = layers(res)
    assert {"text", "coupling", "co-citation", "direct citation", "co-authorship"} <= set(L)
    n = len(res.papers)
    assert L["text"].size == res.graph().number_of_edges()  # text layer = the map's links
    assert L["direct citation"].size == len({tuple(sorted(p)) for p in res.citation_pairs})
    J, lift, summ = overlap(res)
    assert np.allclose(J.values, J.values.T) and (np.diag(J.values) == 1).all()
    assert lift.loc["direct citation", "text"] > 1  # citation ties are text links more often than chance
    # assortativity: perfect within-group ties give r = 1, a pure cross-group pattern r < 0
    vals = pd.Series({0: "a", 1: "a", 2: "b", 3: "b"})
    assert assortativity(np.array([0 * 4 + 1, 2 * 4 + 3]), 4, vals)["r"] == pytest.approx(1.0)
    assert assortativity(np.array([0 * 4 + 2, 1 * 4 + 3]), 4, vals)["r"] < 0
    pr = pagerank(res)
    assert pr.sum() == pytest.approx(1.0) and len(pr) == n
    assert res.papers.at[pr.idxmax(), "title"].startswith(("Work Group Diversity", "Absorptive Capacity"))
    comm, pairs, tot = flow_direction(res)
    assert 0 <= tot["reciprocity"] <= 1 and tot["reciprocity"] < tot["expected"]  # citations point back in time
    assert comm["source score"].between(-1, 1).all()
    for qid in ("Q35", "Q36", "Q37"):
        a = res.answer(qid)
        assert a.findings and a.tables and a.chart, qid
    assert [q for q in res.questions()["id"] if q in ("Q35", "Q36", "Q37")] == ["Q35", "Q37", "Q36"]  # by stage
    html = res.report(tmp_path / "r.html").read_text()
    assert all(f'id="{q}"' in html for q in ("Q35", "Q36", "Q37")) and "Could not compute" not in html


def test_projection_backbone_and_four_cycles(res, df, tmp_path):
    """Significant ties and four-cycle clustering of the two-mode networks behind coupling,
    co-citation and co-authorship (Q35 projection check, S3Config(backbone=True))."""
    import scipy.sparse as sp
    from s3bib.backbone import backbone, bipartite_clustering, projection_clustering
    from s3bib.layers import layers, projection_check, two_mode
    # four-cycles: two papers sharing two items close every 3-path; a path has no square
    assert bipartite_clustering(sp.csr_matrix(np.ones((2, 2)))) == pytest.approx(1.0)
    assert bipartite_clustering(sp.csr_matrix(np.array([[1, 1, 0], [0, 1, 1]]))) == 0.0
    assert projection_clustering(np.array([0 * 3 + 1, 0 * 3 + 2, 1 * 3 + 2]), 3) == pytest.approx(1.0)
    rng = np.random.default_rng(0)
    for name, B in two_mode(res).items():
        B = sp.csr_matrix(B).tocoo()
        # degree-preserving shuffle of the real ties: no structure left, so nothing is significant
        R = sp.csr_matrix((np.ones(B.nnz), (B.row, rng.permutation(B.col))), shape=B.shape)
        assert backbone(R)["keys"].size == 0, name
        assert backbone(B)["tested"] == layers(res, backbone=False)[name].size, name  # every tie is tested
    raw, sig = layers(res, backbone=False), layers(res, backbone=True)
    assert all(np.isin(sig[k], raw[k]).all() for k in sig)  # significant ties are a subset
    pc = projection_check(res)
    assert set(pc["layer"]) == {"coupling", "co-citation", "co-authorship"}
    assert (pc["clustering of the projection"] > pc["four-cycle clustering (two-mode)"]).all()  # projections inflate
    r = s3bib.run_s3(df, s3bib.S3Config(map_weights="text+coupling", backbone=True), **QUIET)
    assert r.config.backbone and r.best
    a = r.answer("Q35")
    assert "Projection check: significant ties and four-cycle clustering" in a.tables
    assert any("significant ties here" in f for f in a.findings)


def test_levels(res):
    """Level 1 = the S3 paper (+ extensions on the same map), 2 = + two-mode/KG, 3 = + prediction."""
    from s3bib.questions import LEVEL2, PUBLISHED, QUESTIONS
    l1, l2, l3 = (set(res.questions(level=k)["id"]) for k in (1, 2, 3))
    assert l1 < l2 < l3 and l3 == {q.qid for q in QUESTIONS}
    assert not (l1 & LEVEL2) and LEVEL2 <= l2 and PUBLISHED <= l1
    assert all(q.prediction == (q.level == 3) for q in QUESTIONS)
    assert set(res.questions()["id"]) == l2  # default level 2: prediction stays off
    t = res.questions(level=1).set_index("id")
    assert t.at["Q10", "source"] == "S3 paper" and t.at["Q11", "source"] == "extension"
    with pytest.raises(ValueError):
        s3bib.S3Config(level=4)


def test_paper_evaluation(res):
    """Q38 reproduces the S3 paper's Tables 5-7 on real data: S3 vs coupling vs random."""
    from s3bib.evaluate import evaluate, tci
    t = evaluate(res, bertopic=False)["table"].set_index("method")
    s3 = t.loc["S3"]
    assert "Bibliographic coupling" in t.index
    # better than random clusters of the same sizes
    assert s3["TCI gain over random"] > 0 and s3["silhouette"] > s3["silhouette random (same sizes)"]
    assert s3["QI (top communities)"] > s3["QI random (same sizes)"]
    assert 0 < s3["TCI"] <= 1 and s3["silhouette"] == pytest.approx(res.best["silhouette"], abs=0.05)
    import scipy.sparse as sp
    same = sp.csr_matrix(np.array([[.5, .5, 0], [.5, .5, 0], [.5, .5, 0]]))
    assert tci(same, np.array([0, 0, 0])) == pytest.approx(1.0)  # identical papers: perfect coherence
    a = res.answer("Q38")
    assert a.tables and a.chart and any("TCI" in f for f in a.findings)


def test_review_report_and_exports(res, tmp_path):
    """The literature-review layout: brief, theme cards, every question of the level once, downloads."""
    from s3bib.questions import QUESTIONS
    from s3bib.review import SECTIONS
    placed = [q for _, _, _, qs in SECTIONS for q in qs]
    assert sorted(placed) == sorted(q.qid for q in QUESTIONS) and len(placed) == len(set(placed))
    for lv in (1, 2):
        html = res.report(tmp_path / f"r{lv}.html", level=lv).read_text()
        ids = {q.qid for q in QUESTIONS if q.level <= lv}
        assert all(f'id="{q}"' in html for q in ids)
        assert not any(f'id="{q.qid}"' in html for q in QUESTIONS if q.level > lv)
        assert 'id="brief"' in html and html.count('class="theme"') == len(res.communities().head(10))
        assert "Could not compute" not in html and "For your review" in html
        assert 'download="reading_list.ris"' in html and "Methods paragraph" in html
        assert "https://github.com/AI-Growth-Lab/S3" in html  # the S3 project, on every report
        assert "https://doi.org/10.1016/j.jbusres.2026.116582" in html  # and the paper
    assert "Who works on it" in html  # level 2 adds actors to the theme cards
    old = res.report(tmp_path / "q.html", layout="questions").read_text()
    assert 'id="brief"' not in old and 'id="Q1"' in old
    files = res.export_review(tmp_path / "exp")
    ris = files["reading_list.ris"].read_text()
    assert ris.count("TY  - JOUR") == ris.count("ER  - ") > 0 and "S3 theme:" in ris
    bib = files["reading_list.bib"].read_text()
    assert bib.count("@article{") == ris.count("TY  - JOUR")
    keys = [l.split("{", 1)[1].rstrip(",") for l in bib.splitlines() if l.startswith("@article{")]
    assert len(keys) == len(set(keys))  # unique citation keys
    m = files["methods.txt"].read_text()
    assert "Bekamiri" in m and f"{res.best['threshold']:.2f}" in m
    pr = pd.read_csv(files["prisma.csv"])
    assert pr.iloc[0]["papers"] == len(res.papers) + len(res.removed)


def test_map_choice(res, df, tmp_path):
    """Q0: the maps are compared on held-out citations, TCI, stability and coverage, and a fixed
    rule recommends one; map_weights="auto" runs on it."""
    from s3bib.mapchoice import split_pairs
    pairs = res.citation_pairs
    train, test = split_pairs(pairs, 0.2, 42)
    hidden = {tuple(p) for p in test}
    assert not any((min(a, b), max(a, b)) in hidden for a, b in train)  # hidden links are not used
    assert len(train) + sum(1 for a, b in pairs if (min(a, b), max(a, b)) in hidden) == len(pairs)

    ch = res.compare_maps()
    t = ch["table"].set_index("map")
    assert "text" in t.index and ch["recommended"] in t.index and ch["n_heldout"] == len(test)
    assert len(t) + len(ch["skipped"]) == 1 + 3 * 8  # text + three relations at 5%, 10%, ... 40%
    assert (t["coverage"].between(0, 1)).all() and (t["stability (ARI)"] <= 1).all()
    assert t.loc["text", "agreement with text (ARI)"] == pytest.approx(1.0)
    base = t.loc["text"]
    # the rule: with a very high bar the text map is kept; with no bar the best held-out gain wins
    from s3bib.mapchoice import _decide
    strict = dict(ch["margins"], min_heldout_gain=1.0)
    assert _decide(ch["table"], strict, True)[0] == "text"
    loose = {"min_heldout_gain": -1, "tci_tolerance": 1, "max_stability_loss": 2, "max_coverage_loss": 1}
    hybrids = t.drop(index="text")
    if len(hybrids):
        assert _decide(ch["table"], loose, True)[0] == hybrids["held-out citations gain"].idxmax()
    assert _decide(ch["table"], ch["margins"], False)[0] == "text"  # no citations: keep text
    if ch["recommended"] != "text":
        r = t.loc[ch["recommended"]]
        assert r["held-out citations gain"] >= base["held-out citations gain"] + ch["margins"]["min_heldout_gain"]

    a = res.answer("Q0")
    assert a.tables and a.chart and any("Recommended map" in f for f in a.findings)

    # the text floor: citations add nothing to pairs below it, and something above it
    from s3bib.hybrid import hybrid_matrix, resolve_text_floor, coupling_matrix
    floor = resolve_text_floor(None, res.config.thresholds)
    assert floor == pytest.approx(min(res.config.thresholds) - 0.05) and resolve_text_floor(False, (0.8,)) is None
    S = np.clip(res.embeddings @ res.embeddings.T, 0, None)
    H = hybrid_matrix(res.embeddings, res.papers, res.citation_pairs, {"s3": 0.8, "coupling": 0.2}, None, floor)
    C = coupling_matrix(res.papers)
    extra = H - 0.8 * S
    np.fill_diagonal(extra, 0)
    off = ~np.eye(len(S), dtype=bool)
    assert np.allclose(extra[(S < floor) & off], 0, atol=1e-5)
    assert (extra[(S >= floor) & (C > 0) & off] > 0).all() and ((S >= floor) & (C > 0) & off).any()

    auto = s3bib.run_s3(df, s3bib.S3Config(map_weights="auto"), **QUIET)
    assert auto.map_choice["auto"] and auto.map_choice["recommended"] == ch["recommended"]
    assert auto.config.map_weights == ch["weights"]
    back = s3bib.S3Result.load(auto.save(tmp_path / "run"))
    assert back.map_choice["recommended"] == ch["recommended"] and len(back.map_choice["table"]) == len(t)
    assert "Before the analysis we compared" in auto.report(tmp_path / "auto.html").read_text()


def test_r_package_version_matches_python():
    """The R package (r/) runs the Python library of its own version, so the two
    version numbers must always be the same: bump both in every release."""
    import re
    root = Path(__file__).resolve().parents[1]
    py = re.search(r'^version = "([^"]+)"', (root / "pyproject.toml").read_text(), re.M).group(1)
    r = re.search(r"^Version:\s*(\S+)", (root / "r" / "DESCRIPTION").read_text(), re.M).group(1)
    assert r == py == s3bib.__version__
