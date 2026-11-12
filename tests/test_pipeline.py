"""Fast tests: embeddings are stubbed, so no model download is needed."""
import numpy as np
import pandas as pd
import pytest

import s3bib
from s3bib import pipeline


def _fake_embed(texts, *a, **k):
    # 3 well-separated topics + noise; topic is encoded in the title prefix
    rng = np.random.default_rng(0)
    centers = np.eye(8)[:3]
    out = []
    for t in texts:
        c = centers[int(t[1])] if t.startswith("T") and t[1].isdigit() else rng.normal(size=8)
        out.append(c + 0.05 * rng.normal(size=8))
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
