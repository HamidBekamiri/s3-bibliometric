"""Fast tests: embeddings are stubbed, so no model download is needed."""
import numpy as np
import pandas as pd
import pytest

import s3bib
from s3bib import pipeline


def _fake_embed(texts, *a, **k):
    # 3 well-separated topics + noise; topic is encoded in the title prefix
    rng = np.random.default_rng(0)
    centers = np.eye(64)[:3]
    out = []
    for t in texts:
        c = centers[int(t[1])] if t.startswith("T") and t[1].isdigit() else rng.normal(size=64)
        out.append(c + 0.045 * rng.normal(size=64))
    e = np.array(out, dtype=np.float32)
    return e / np.linalg.norm(e, axis=1, keepdims=True)


@pytest.fixture
def df():
    rows = []
    for topic in range(3):
        for k in range(10):
            rows.append({"Title": f"T{topic} paper {k}", "Abstract": "abc",
                         "Year": 2010 + k, "Cited by": k, "EID": f"e{topic}{k}",
                         "References": f"x, T{topic} paper {(k + 1) % 10}, y"})
    rows.append({"Title": "Unrelated", "Abstract": "zzz", "Year": 2015,
                 "Cited by": 1, "EID": "ex", "References": ""})
    return pd.DataFrame(rows)


def test_run_finds_three_communities(df, monkeypatch, tmp_path):
    monkeypatch.setattr(pipeline, "embed_texts", _fake_embed)
    res = s3bib.run_s3(df, progress=lambda s, p: None)
    comms = res.communities()
    assert len(comms) == 3
    assert sorted(comms["n_papers"]) == [10, 10, 10]
    assert res.outliers()["title"].tolist() == ["Unrelated"]
    assert res.best["sii"] == max(r["sii"] for r in res.scan)
    # citations stay within each topic
    C = res.citation_matrix().to_numpy()
    assert C.trace() == C.sum() > 0

    res.labels = {0: "A"}
    res.save(tmp_path / "run")
    again = s3bib.S3Result.load(tmp_path / "run")
    assert again.labels == {0: "A"}
    assert again.papers["community"].equals(res.papers["community"])
    files = res.export(tmp_path / "out")
    assert "papers_csv" in files


def test_requires_title_and_abstract():
    with pytest.raises(ValueError):
        s3bib.load_papers(pd.DataFrame({"Title": ["a", "b", "c"]}))


def test_every_question_answers(df, monkeypatch, tmp_path):
    monkeypatch.setattr(pipeline, "embed_texts", _fake_embed)
    df = df.assign(Authors=["A1; A2"] * 15 + ["A2; A3"] * 16,
                   Affiliations=["Aalborg University, Aalborg, Denmark"] * 31,
                   **{"Source title": ["J1", "J2"] * 15 + ["J1"], "Author Keywords": ["k1; k2"] * 31})
    res = s3bib.run_s3(df, progress=lambda s, p: None)
    qt = res.questions()
    assert list(qt["id"]) == [q.qid for q in s3bib.QUESTIONS]
    for qid in qt.loc[qt["available"], "id"]:
        a = res.answer(qid)
        assert a.findings, qid
        a.to_dict()
    assert res.answer("Q20").findings[0].startswith("Not available")  # no funding column
    out = res.report(tmp_path / "r.html")
    html = out.read_text()
    assert "Q1" in html and "Q23" in html and "Could not compute" not in html


def test_split_refs_handles_both_scopus_formats():
    from s3bib.questions import _split_refs
    new = "Gielis J.; Shi P.; Caratelli D., Universal Equations, Growth Form, 3, (2022); Koza J.R., Genetic programming, Stat. Comput., 4, (1994)"
    old = "Smith J., Doe A., A study, Journal, 3, (2001); Lee K., Another study, J2, (1999a)"
    assert len(_split_refs(new)) == 2 and _split_refs(new)[0].startswith("Gielis J.; Shi P.")
    assert _split_refs(old) == ["Smith J., Doe A., A study, Journal, 3, (2001)", "Lee K., Another study, J2, (1999a)"]


def test_time_questions_skipped_for_single_year(df, monkeypatch):
    monkeypatch.setattr(pipeline, "embed_texts", _fake_embed)
    res = s3bib.run_s3(df.assign(Year=2027), progress=lambda s, p: None)
    for qid in ("Q13", "Q23"):
        assert "fewer than 3 publication years" in res.answer(qid).findings[0]


def test_concept_trends_ignore_search_terms(df, monkeypatch):
    monkeypatch.setattr(pipeline, "embed_texts", _fake_embed)
    years = list(range(2010, 2023))
    rows = []
    for k, y in enumerate(years * 30):
        kws = ["innovation"] + ((["green innovation"] if y >= 2018 else ["networks"]) if k % 12 == 0 else [])
        rows.append({"Title": f"T{k % 3} p{k}", "Abstract": "x", "Year": y, "Cited by": 1,
                     "EID": f"e{k}", "Author Keywords": "; ".join(kws)})
    res = s3bib.run_s3(pd.DataFrame(rows), progress=lambda s, p: None)
    f = " ".join(res.answer("Q14").findings)
    assert "green innovation" in f.split("Fading")[0] and "networks" in f.split("Fading")[1]
    assert "innovation (" not in f.replace("green innovation (", "")


def test_fixed_threshold_and_methods(df, monkeypatch):
    monkeypatch.setattr(pipeline, "embed_texts", _fake_embed)
    fixed = s3bib.run_s3(df, s3bib.S3Config(threshold=0.8), progress=lambda s, p: None)
    assert fixed.best["threshold"] == 0.8 and fixed.best["fixed"]
    assert "fixed" in fixed.answer("Q1").findings[0]
    lv = s3bib.run_s3(df, s3bib.S3Config(community_method="louvain"), progress=lambda s, p: None)
    assert lv.config.community_method == "louvain" and len(lv.communities()) >= 3
    assert len(fixed.communities()) == 3


def test_citation_metrics_count_whole_corpus(df, monkeypatch):
    monkeypatch.setattr(pipeline, "embed_texts", _fake_embed)
    res = s3bib.run_s3(df, progress=lambda s, p: None)
    m = res.citation_metrics()
    assert {"inward_citations", "outward_citations", "global_citations"} <= set(m.columns)
    assert m["local_citations"].equals(m["within_citations"] + m["inward_citations"])
    assert abs(m["mi"].sum() - m["local_citations"].sum() / res.mi_denominator() * 100) < 1e-9
    assert res.top_terms(0, 3)  # c-TF-IDF terms


def test_runs_are_reproducible(df, monkeypatch):
    monkeypatch.setattr(pipeline, "embed_texts", _fake_embed)
    a = s3bib.run_s3(df, progress=lambda s, p: None)
    b = s3bib.run_s3(df, progress=lambda s, p: None)
    assert a.papers["community"].equals(b.papers["community"]) and a.scan == b.scan
    for qid, table in [("Q7", "Most central papers per community"), ("Q12", "Bridging papers")]:
        ta, tb = a.answer(qid).tables.get(table), b.answer(qid).tables.get(table)
        assert (ta is None and tb is None) or ta.equals(tb)


def test_empty_and_duplicate_records_removed(df, monkeypatch):
    monkeypatch.setattr(pipeline, "embed_texts", _fake_embed)
    extra = pd.DataFrame([
        {"Title": "", "Abstract": "", "Year": 2015, "Cited by": 0, "EID": "empty1"},
        {"Title": "T0 paper 1", "Abstract": "abc", "Year": 2011, "Cited by": 0, "EID": "dup-text"},
        dict(df.iloc[2]) | {"Title": "T0 paper 2 (reprint)"},  # same EID as row 2
    ])
    res = s3bib.run_s3(pd.concat([df, extra], ignore_index=True), progress=lambda s, p: None)
    reasons = res.removed["reason"].tolist()
    assert len(res.papers) == len(df)
    assert "no title and no abstract" in reasons
    assert sum(r.startswith("exact duplicate") for r in reasons) == 2
    kept = s3bib.run_s3(pd.concat([df, extra], ignore_index=True), s3bib.S3Config(dedupe=False),
                        progress=lambda s, p: None)
    assert len(kept.papers) == len(df) + 2  # empty record is always dropped


def test_near_duplicates_by_s3(df, monkeypatch):
    def emb(texts, *a, **k):
        e = _fake_embed(texts)
        e[-1] = e[0]  # last paper is an exact semantic copy of the first
        return e
    monkeypatch.setattr(pipeline, "embed_texts", emb)
    d = pd.concat([df, df.iloc[[0]].assign(Title="T0 paper 0 reprinted", EID="other")], ignore_index=True)
    res = s3bib.run_s3(d, progress=lambda s, p: None)
    assert res.removed["reason"].str.startswith("near duplicate").sum() == 1


def test_model_presets():
    from s3bib.models import MPNET_THRESHOLDS, WIDE_THRESHOLDS
    assert s3bib.S3Config().model_name == s3bib.DEFAULT_MODEL
    assert s3bib.S3Config().thresholds == MPNET_THRESHOLDS
    c = s3bib.S3Config(model_name="specter")
    assert c.model_name == "sentence-transformers/allenai-specter" and c.thresholds == WIDE_THRESHOLDS
    assert s3bib.S3Config(model_name="clip").model_name.endswith("clip-ViT-B-32")


def test_scopus_enrichment_with_mocked_api(df, monkeypatch, tmp_path):
    monkeypatch.setattr(pipeline, "embed_texts", _fake_embed)
    res = s3bib.run_s3(df, progress=lambda s, p: None)
    eids = res.papers["eid"].tolist()

    class Resp:
        def __init__(self, payload, code=200):
            self.payload, self.status_code, self.headers = payload, code, {}
        def json(self):
            return self.payload
        def raise_for_status(self):
            pass

    class Session:
        calls = 0
        def get(self, url, headers, params, timeout):
            Session.calls += 1
            assert headers["X-ELS-APIKey"] == "test-key" and params["view"] == "REF"
            eid = url.rsplit("/", 1)[-1]
            i = eids.index(eid)
            refs = [{"scopus-eid": eids[(i + 1) % len(eids)]}, {"scopus-id": "999"}, {"@id": "123"}]
            return Resp({"abstracts-retrieval-response": {"references": {
                "@total-references": "3", "reference": refs}}})

    stats = s3bib.enrich_citations(res, api_key="test-key", session=Session(), cache_dir=tmp_path, pause=0)
    assert res.citation_source == "scopus_api" and stats["failed"] == 0
    assert len(res.citation_pairs) == len(eids)  # each paper cites the next one
    assert "Scopus API (exact)" in res.answer("Q10").findings[0]
    n = Session.calls
    s3bib.enrich_citations(res, api_key="test-key", session=Session(), cache_dir=tmp_path, pause=0)
    assert Session.calls == n  # second run comes from the cache


def test_scopus_access_error(df, monkeypatch):
    monkeypatch.setattr(pipeline, "embed_texts", _fake_embed)
    res = s3bib.run_s3(df, progress=lambda s, p: None)

    class Denied:
        def get(self, *a, **k):
            class R: status_code, headers = 403, {}
            return R()
    monkeypatch.delenv("SCOPUS_API_KEY", raising=False)
    with pytest.raises(s3bib.ScopusAccessError):
        s3bib.enrich_citations(res)  # no key at all
    with pytest.raises(s3bib.ScopusAccessError):
        s3bib.enrich_citations(res, api_key="x", session=Denied(), cache_dir=None, pause=0)
