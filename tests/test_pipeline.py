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
