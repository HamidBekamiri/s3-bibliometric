"""Build the test corpus from real OpenAlex records (CC0) and their real mpnet embeddings.

The corpus is two seminal papers plus the most-cited papers that cite them in four periods:

  W2108795964  Cohen & Levinthal (1990) Absorptive capacity
  W3125646472  the same paper reprinted as a book chapter (a real near-duplicate record)
  W2159012497  van Knippenberg et al. (2004) Work group diversity and group performance

Columns follow a Scopus export. `References` lists only the references that are inside the
corpus (OpenAlex `referenced_works`), formatted like Scopus references; `Referenced IDs` keeps
the OpenAlex ids so tests can mock the Scopus reference API with the real reference lists.
Affiliation countries are OpenAlex ISO country codes.

Run from the repository root (needs network access and the mpnet model):

    python tests/data/build_openalex_sample.py
"""
from __future__ import annotations

import time
import urllib.parse
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
API = "https://api.openalex.org/works"
SEEDS = {"W2108795964": "absorptive capacity", "W2159012497": "work group diversity"}
EXTRA = ["W3125646472"]  # reprint of Cohen & Levinthal (1990)
PERIODS = [(1991, 2004), (2005, 2012), (2013, 2018), (2019, 2025)]
PER_PERIOD = 15
FIELDS = ("id,doi,display_name,publication_year,cited_by_count,abstract_inverted_index,"
          "authorships,primary_location,keywords,referenced_works,funders,type")


def get(url: str) -> dict:
    for attempt in range(5):
        try:
            import requests
            r = requests.get(url, timeout=60)
            r.raise_for_status()
            return r.json()
        except Exception:  # pragma: no cover - network retry
            time.sleep(2 ** attempt)
    raise RuntimeError(f"OpenAlex request failed: {url}")


def abstract(inv) -> str:
    if not inv:
        return ""
    pos = sorted((p, w) for w, ps in inv.items() for p in ps)
    return " ".join(w for _, w in pos)


def short(w: str) -> str:
    return w.rsplit("/", 1)[-1]


def fetch() -> list[dict]:
    works = {}
    for wid in list(SEEDS) + EXTRA:
        works[wid] = get(f"{API}/{wid}?select={FIELDS}")
    for seed in SEEDS:
        for a, b in PERIODS:
            flt = (f"cites:{seed},has_abstract:true,type:article,"
                   f"publication_year:{a}-{b},language:en")
            q = urllib.parse.urlencode({"filter": flt, "sort": "cited_by_count:desc",
                                        "per-page": PER_PERIOD, "select": FIELDS})
            for w in get(f"{API}?{q}")["results"]:
                works.setdefault(short(w["id"]), w)
    return list(works.values())


def to_rows(works: list[dict]) -> pd.DataFrame:
    by_id = {short(w["id"]): w for w in works}

    def authors(w):
        return "; ".join(a["author"]["display_name"] for a in w["authorships"])

    def ref(w):
        names = ", ".join(a["author"]["display_name"] for a in w["authorships"][:3])
        src = ((w.get("primary_location") or {}).get("source") or {}).get("display_name") or ""
        return f"{names}, {w['display_name']}, {src}, ({w['publication_year']})"

    rows = []
    for w in works:
        wid = short(w["id"])
        affs = []
        for a in w["authorships"]:
            for inst in a.get("institutions") or []:
                affs.append(f"{inst['display_name']}, {inst.get('country_code') or ''}".strip(", "))
        inside = [short(r) for r in w.get("referenced_works") or [] if short(r) in by_id]
        src = ((w.get("primary_location") or {}).get("source") or {}).get("display_name") or ""
        rows.append({
            "Title": w["display_name"], "Abstract": abstract(w["abstract_inverted_index"]),
            "Year": w["publication_year"], "Cited by": w["cited_by_count"], "EID": wid,
            "DOI": (w.get("doi") or "").replace("https://doi.org/", ""),
            "Authors": authors(w), "Affiliations": "; ".join(dict.fromkeys(affs)),
            "Author Keywords": "; ".join(k["display_name"] for k in w.get("keywords") or []),
            "Source title": src,
            "Funding Details": "; ".join(dict.fromkeys(
                g.get("display_name") or g.get("funder_display_name") for g in w.get("funders") or [] if g.get("display_name") or g.get("funder_display_name"))),
            "References": "; ".join(ref(by_id[r]) for r in inside),
            "Referenced IDs": "; ".join(inside),
            "Document Type": w.get("type") or "",
        })
    return pd.DataFrame(rows).sort_values(["Year", "EID"]).reset_index(drop=True)


def main() -> None:
    df = to_rows(fetch())
    df.to_csv(HERE / "openalex_sample.csv", index=False)
    import s3bib
    from s3bib.pipeline import embed_texts
    papers = s3bib.load_papers(df)
    emb = embed_texts(papers["text"].tolist(), s3bib.DEFAULT_MODEL)
    np.savez_compressed(HERE / "openalex_sample_mpnet.npz", text=np.array(papers["text"].tolist()),
                        emb=emb.astype(np.float32))
    print(f"{len(df)} works, {int((df['References'] != '').sum())} with in-corpus references")


if __name__ == "__main__":
    main()
