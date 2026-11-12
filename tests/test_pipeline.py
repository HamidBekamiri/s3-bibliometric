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
    assert list(qt["id"]) == [q.qid for q in s3bib.QUESTIONS]
    assert qt["available"].all()
    for qid in qt["id"]:
        a = res.answer(qid)
        assert a.findings, qid
        a.to_dict()
    html = res.report(tmp_path / "r.html").read_text()
    assert "Q1" in html and "Q23" in html and "Could not compute" not in html
    assert 'class="ktree"' in html  # knowledge tree under Q13
    assert "force-graph@1.51.0" in html and 's3FG("' in html  # network drawn as in the S3 web app


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
