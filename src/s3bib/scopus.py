"""Exact in-corpus citations from the Scopus Abstract Retrieval API (REF view).

Without an API key, S3 finds citations by matching titles inside the References text,
which misses some links and can add false ones. With a Scopus API key, each paper's
reference list is retrieved as Scopus IDs, so a citation counts only when the cited
paper's EID is in the corpus.

The key is read from the SCOPUS_API_KEY environment variable (and an optional
SCOPUS_INST_TOKEN). The REF view needs Scopus entitlement: run it on your institution's
network, or use an institutional token.

    import s3bib
    res = s3bib.run_s3("scopus.csv")
    s3bib.enrich_citations(res)      # uses SCOPUS_API_KEY
"""
from __future__ import annotations

import json
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set

API = "https://api.elsevier.com/content/abstract/eid/{eid}"
PAGE = 200


class ScopusAccessError(RuntimeError):
    """The key is missing, wrong, or not entitled to reference lists."""


def _as_list(x) -> list:
    if x is None:
        return []
    return x if isinstance(x, list) else [x]


def parse_reference_eids(payload: Dict[str, Any]) -> tuple[List[str], int]:
    """(cited EIDs on this page, total number of references) from a REF-view response."""
    root = payload.get("abstracts-retrieval-response", payload)
    refs = root.get("references") or {}
    total = int(refs.get("@total-references") or 0)
    eids = []
    for r in _as_list(refs.get("reference")):
        eid = r.get("scopus-eid")
        sid = r.get("scopus-id") or r.get("@id")
        if not eid and sid:
            eid = f"2-s2.0-{str(sid).split(':')[-1]}"
        if eid:
            eids.append(str(eid).strip())
    return eids, total


def fetch_reference_eids(eid: str, api_key: str, inst_token: Optional[str] = None,
                         session=None, cache_dir: Optional[Path] = None,
                         max_retries: int = 5) -> List[str]:
    """All cited EIDs of one paper (paginated, cached on disk)."""
    if cache_dir is not None:
        f = Path(cache_dir) / f"{eid}.json"
        if f.exists():
            return json.loads(f.read_text())
    if session is None:
        import requests
        session = requests.Session()
    s = session
    headers = {"X-ELS-APIKey": api_key, "Accept": "application/json"}
    if inst_token:
        headers["X-ELS-Insttoken"] = inst_token
    out: List[str] = []
    start = 1
    while True:
        for attempt in range(max_retries):
            r = s.get(API.format(eid=eid), headers=headers,
                      params={"view": "REF", "startref": start, "refcount": PAGE}, timeout=30)
            if r.status_code == 429 or r.status_code >= 500:
                time.sleep(float(r.headers.get("Retry-After", 2 ** attempt)))
                continue
            break
        if r.status_code in (401, 403):
            raise ScopusAccessError(
                f"Scopus refused the request (HTTP {r.status_code}). Check the API key, and that it is "
                "entitled to reference lists: use your institution's network or an institutional token.")
        if r.status_code == 404:
            break
        r.raise_for_status()
        page, total = parse_reference_eids(r.json())
        out += page
        if not page or start + len(page) > total:
            break
        start += len(page)
    if cache_dir is not None:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        (Path(cache_dir) / f"{eid}.json").write_text(json.dumps(out))
    return out


def enrich_citations(res, api_key: Optional[str] = None, inst_token: Optional[str] = None,
                     cache_dir: Optional[str] = "~/s3bib_runs/_scopus",
                     progress: Optional[Callable[[str, float], None]] = None,
                     session=None, pause: float = 0.12) -> Dict[str, Any]:
    """Replace the run's citation links with exact links from the Scopus API.
    Updates res.citation_pairs, res.papers['local_citations'] and res.citation_source."""
    api_key = api_key or os.environ.get("SCOPUS_API_KEY")
    inst_token = inst_token or os.environ.get("SCOPUS_INST_TOKEN")
    if not api_key:
        raise ScopusAccessError("No Scopus API key. Set the SCOPUS_API_KEY environment variable "
                                "(get a key at https://dev.elsevier.com).")
    P = res.papers
    eid_to_idx = {str(e).strip(): i for i, e in P["eid"].items() if isinstance(e, str) and e.strip()}
    if not eid_to_idx:
        raise ValueError("The export has no EID column, which the Scopus API needs.")
    cdir = Path(cache_dir).expanduser() if cache_dir else None
    pairs: Set[tuple] = set()
    failed: List[Dict[str, str]] = []
    n = len(eid_to_idx)
    for k, (eid, i) in enumerate(eid_to_idx.items()):
        try:
            cached = cdir is not None and (cdir / f"{eid}.json").exists()
            for cited in fetch_reference_eids(eid, api_key, inst_token, session=session, cache_dir=cdir):
                j = eid_to_idx.get(cited)
                if j is not None and j != i:
                    pairs.add((i, j))
            if not cached and pause:
                time.sleep(pause)
        except ScopusAccessError:
            raise
        except Exception as e:  # keep going; report at the end
            failed.append({"eid": eid, "error": str(e)[:200]})
        if progress:
            progress(f"Scopus references {k + 1}/{n}", (k + 1) / n)
    old = len(res.citation_pairs)
    res.citation_pairs = sorted(pairs)
    received = Counter(j for _, j in res.citation_pairs)
    res.papers["local_citations"] = [received.get(i, 0) for i in range(len(P))]
    res.citation_source = "scopus_api"
    return {"papers_queried": n, "failed": len(failed), "failures": failed[:20],
            "citation_links_before": old, "citation_links_now": len(res.citation_pairs)}
