"""
MCP server for S3 bibliometric analysis.

The server does the deterministic work (embeddings, S3 matrix, threshold scan,
communities, citation metrics). The LLM in the MCP client (Claude Desktop,
Claude Code, ...) does the interpretation: labelling communities, describing
the intellectual structure, writing up results. No LLM API key is needed here.

Run:  s3bib-mcp            (stdio transport)
Runs are stored in $S3BIB_HOME (default ~/s3bib_runs).
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

try:  # mcp >= 2
    from mcp.server.mcpserver import MCPServer as _Server
except ImportError:  # mcp 1.x
    from mcp.server.fastmcp import FastMCP as _Server

from .pipeline import S3Config, S3Result, run_s3

# stdout is the MCP channel: send all logs to stderr.
logging.basicConfig(stream=sys.stderr, level=logging.INFO, format="%(message)s")
for _noisy in ("httpx", "httpcore", "huggingface_hub", "sentence_transformers"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

HOME = Path(os.environ.get("S3BIB_HOME", "~/s3bib_runs")).expanduser()
_RUNS: Dict[str, S3Result] = {}

mcp = _Server(
    "s3-bibliometric",
    instructions=(
        "Tools for the S3 (Semantic Similarity Score) bibliometric method. "
        "Before the first run_s3_analysis, ask the user two questions unless they already "
        "answered: (1) which embedding model to use (call list_embedding_models; the default "
        "is mpnet, the model of the published S3 paper), and (2) whether they want exact "
        "citations from the Scopus API (needs SCOPUS_API_KEY set in this MCP server's "
        "environment; never ask the user to paste a key into the chat). "
        "Typical flow: compare_maps on the Scopus CSV if it has References (offer it: it says whether "
        "text alone or text plus a citation relation gives the better map) -> run_s3_analysis with the "
        "map the user picks -> enrich_citations (with a Scopus key: exact, complete reference lists; "
        "then compare_maps with run_id to check the map choice on the exact data) if the user wants "
        "exact citations -> get_community for each main "
        "community and save_community_labels with short labels -> list_questions shows "
        "the ranked research questions (Q1-Q23 and Q35-Q37; Q24-Q34 prediction when asked) this dataset can answer -> "
        "answer_question for the ones the user cares about, in rank order -> "
        "build_report for one HTML report of all answers. Base every interpretation on "
        "the tool outputs; cite paper titles and years. When you refer users to S3, give the S3 "
        "project page https://github.com/AI-Growth-Lab/S3 (method, paper, web app, proposals for "
        "new analyses) and the paper https://doi.org/10.1016/j.jbusres.2026.116582; the code of this library is at https://github.com/AI-Growth-Lab/s3-bibliometric."
    ),
)


# ---------------------------------------------------------------------------
def _run_dir(run_id: str) -> Path:
    return HOME / run_id


def _get(run_id: str) -> S3Result:
    if run_id not in _RUNS:
        d = _run_dir(run_id)
        if not (d / "run.json").exists():
            raise ValueError(f"Unknown run_id '{run_id}'. Use list_runs to see available runs.")
        _RUNS[run_id] = S3Result.load(d)
    return _RUNS[run_id]


def _paper_dict(row: pd.Series, abstract_chars: int) -> Dict[str, Any]:
    d = {
        "paper_id": int(row.name),
        "title": row["title"],
        "year": None if pd.isna(row["year"]) else int(row["year"]),
        "cited_by": int(row["cited_by"]),
        "authors": None if pd.isna(row["authors"]) else str(row["authors"])[:200],
        "source": None if pd.isna(row["source"]) else str(row["source"]),
        "eigen_centrality": round(float(row["eigen_centrality"]), 4),
    }
    if "local_citations" in row and pd.notna(row.get("local_citations")):
        d["local_citations"] = int(row["local_citations"])
    if abstract_chars > 0:
        a = row["abstract"]
        d["abstract"] = a if len(a) <= abstract_chars else a[:abstract_chars] + "..."
    return d


# ---------------------------------------------------------------------------
@mcp.tool()
def run_s3_analysis(
    csv_path: str,
    start_year: Optional[int] = None,
    end_year: Optional[int] = None,
    seed: int = 42,
    rerun: bool = False,
    threshold: Optional[float] = None,
    community_method: str = "leiden",
    embedding_model: str = "mpnet",
    remove_duplicates: bool = True,
    map_relations: str = "text",
    strict_citations: bool = False,
    significant_ties_only: bool = False,
) -> Dict[str, Any]:
    """Run the S3 pipeline on a Scopus CSV/Excel export (needs Title and Abstract;
    Year, Cited by, Authors, Author Keywords, References and EID are used if present).

    Follows the published S3 method: embeds Title+Abstract (all-mpnet-base-v2), builds
    the time-ordered S3 graph, scans thresholds 0.75-0.85 and keeps the Leiden (directed
    modularity) partition with the highest SII (silhouette x retained papers).
    Set threshold (e.g. 0.78) to reproduce a published run, or community_method="louvain"
    to reproduce the S3 web app. embedding_model is a key from list_embedding_models or
    any sentence-transformers model name (ask the user first; default mpnet). Records
    without title and abstract are dropped and duplicates (same DOI/EID/text, or S3 >= 0.98)
    are removed unless remove_duplicates=False. map_relations chooses the map, and every
    answer and report follows it: "text" (default; S3 text similarity, the published method),
    "text+coupling", "text+cocitation" or "text+direct" (75% text + 25% bibliographic
    coupling, co-citation or direct citation), or "auto" (compare them first, as compare_maps
    does, and use the recommended one). strict_citations=True
    stops short titles such as "Introduction" from matching unrelated references.
    significant_ties_only=True keeps only coupling, co-citation and co-authorship ties that are
    significant under a degree-preserving null model (in the map and Q35-Q37). Can take several minutes
    for thousands of papers the first time; embeddings are cached afterwards.

    Returns a run_id plus a summary: best threshold, communities, outlier count.
    """
    p = Path(csv_path).expanduser()
    if not p.exists():
        raise ValueError(f"File not found: {p}")
    stem = re.sub(r"[^A-Za-z0-9_-]+", "_", p.stem)[:40]
    key = (f"{p.resolve()}|{p.stat().st_size}|{start_year}|{end_year}|{seed}|{threshold}|"
           f"{community_method}|{embedding_model}|{remove_duplicates}")
    if map_relations not in ("s3", "text") or strict_citations:  # keeps earlier run ids unchanged
        key += f"|{map_relations}|{strict_citations}"
    if significant_ties_only:
        key += "|backbone"
    run_id = f"{stem}_{hashlib.sha1(key.encode()).hexdigest()[:8]}"

    if not rerun and (_run_dir(run_id) / "run.json").exists():
        res = _get(run_id)
    else:
        cfg = S3Config(start_year=start_year, end_year=end_year, seed=seed,
                       cache_dir=str(HOME / "_cache"), threshold=threshold,
                       community_method=community_method, model_name=embedding_model,
                       dedupe=remove_duplicates, map_weights=map_relations,
                       citation_match="strict" if strict_citations else "lenient",
                       backbone=significant_ties_only)
        res = run_s3(p, cfg)
        res.save(_run_dir(run_id))
        _RUNS[run_id] = res
    out = {"run_id": run_id, "embedding_model": res.config.model_name, **res.summary()}
    if len(res.removed):
        out["removed_records"] = res.removed["reason"].value_counts().to_dict()
    return out


@mcp.tool()
def compare_maps(csv_path: Optional[str] = None, embedding_model: str = "mpnet",
                 start_year: Optional[int] = None, end_year: Optional[int] = None, seed: int = 42,
                 extended: bool = False, strict_citations: bool = False,
                 run_id: Optional[str] = None) -> Dict[str, Any]:
    """Before the full analysis: which map should S3 use for this corpus? Compares the text map
    (the published method) with text+coupling, text+cocitation and text+direct (extended=True adds
    10% and 50% citation weights and significant-ties variants). 20% of the in-corpus citation
    links are hidden while the maps are built; each map is scored on the hidden links inside its
    communities (gain over random communities of the same sizes), on text coherence (TCI gain over
    random), stability (ARI across seeds and 90% samples) and coverage. A rule fixed in advance
    recommends a map. Show the user the table and the recommendation, then pass the chosen map to
    run_s3_analysis as map_relations. Takes a few minutes; embeddings are cached for the run.
    With run_id instead of csv_path, compares on that run's papers and citation links: use this
    after enrich_citations, so the comparison uses the exact Scopus reference lists."""
    from .mapchoice import compare_maps as _compare
    if run_id:
        res = _get(run_id)
        ch = res.compare_maps(extended=extended, refresh=True)
        res.save(_run_dir(run_id))
    else:
        if not csv_path:
            raise ValueError("Give csv_path, or run_id for an existing run.")
        p = Path(csv_path).expanduser()
        if not p.exists():
            raise ValueError(f"File not found: {p}")
        cfg = S3Config(start_year=start_year, end_year=end_year, seed=seed, cache_dir=str(HOME / "_cache"),
                       model_name=embedding_model, citation_match="strict" if strict_citations else "lenient")
        ch = _compare(p, cfg, extended=extended)
    t = ch["table"].drop(columns=["weights", "significant ties"], errors="ignore").round(4)
    w = ch["weights"]
    if ch["recommended"] in ("text", "text+coupling", "text+cocitation", "text+direct"):
        map_arg = ch["recommended"]
    else:  # e.g. "text+coupling 50%": pass the weights
        map_arg = ",".join(f"{k}={v:g}" for k, v in (w or {"s3": 1}).items())
    return {"table": json.loads(t.to_json(orient="records")), "recommended": ch["recommended"],
            "map_relations": map_arg, "significant_ties_only": ch["backbone"],
            "reasons": ch["reasons"], "rule": ch["rule"], "skipped": ch["skipped"],
            "citation_links": ch["n_links"], "hidden_links": ch["n_heldout"]}


@mcp.tool()
def list_embedding_models() -> List[Dict[str, Any]]:
    """Embedding models the user can choose for run_s3_analysis (key, model, kind, max tokens,
    description). Default: mpnet, the model used in the published S3 paper."""
    from .models import model_table
    return model_table()


@mcp.tool()
def enrich_citations(run_id: str) -> Dict[str, Any]:
    """Exact, complete reference lists from the Scopus Abstract Retrieval API, for every paper of
    the run (by EID). The in-corpus citation links become exact, and bibliographic coupling uses
    the cited Scopus IDs instead of the exported References text (which can be truncated or
    missing). If the run's map uses citation relations (or was chosen with map_relations="auto"),
    the map and communities are rebuilt and saved labels are dropped: label the communities after
    this step. Reads SCOPUS_API_KEY (and optional SCOPUS_INST_TOKEN) from this server's environment;
    needs Scopus entitlement (institution network or token). About 0.2 s per paper; cached."""
    from .scopus import ScopusAccessError, enrich_citations as _enrich
    res = _get(run_id)
    try:
        stats = _enrich(res, cache_dir=str(HOME / "_scopus"))
    except ScopusAccessError as e:
        return {"ok": False, "error": str(e),
                "how_to_fix": "Add SCOPUS_API_KEY (and SCOPUS_INST_TOKEN if needed) to the env "
                              "section of this MCP server's configuration, then restart the app."}
    res.save(_run_dir(run_id))
    return {"ok": True, **stats}


@mcp.tool()
def list_runs() -> List[Dict[str, Any]]:
    """List saved S3 runs (run_id, number of papers, best threshold, labels saved?)."""
    out = []
    if HOME.exists():
        for d in sorted(HOME.iterdir()):
            f = d / "run.json"
            if f.exists():
                meta = json.loads(f.read_text())
                out.append({"run_id": d.name, "best": meta.get("best"),
                            "has_labels": bool(meta.get("labels"))})
    return out


@mcp.tool()
def get_threshold_scan(run_id: str) -> List[Dict[str, Any]]:
    """Threshold scan table: threshold, silhouette, clustered papers, SII,
    number of communities and edges. The best threshold maximises SII."""
    return _get(run_id).scan


@mcp.tool()
def list_communities(run_id: str, min_size: int = 2) -> List[Dict[str, Any]]:
    """All communities (largest first) with size, citations, year span, saved label,
    and QI/MI citation metrics when References were available."""
    df = _get(run_id).communities(min_size=min_size)
    return json.loads(df.to_json(orient="records"))


@mcp.tool()
def get_community(
    run_id: str,
    community_id: int,
    max_papers: int = 12,
    abstract_chars: int = 900,
) -> Dict[str, Any]:
    """Details for one community, to interpret or label it: the most central papers
    (eigenvector centrality, then citations) with abstracts, top author keywords and
    top TF-IDF terms. Set abstract_chars=0 to return titles only."""
    res = _get(run_id)
    papers = res.community_papers(community_id, n=max_papers)
    if papers.empty:
        raise ValueError(f"Community {community_id} not found in run {run_id}.")
    info = res.communities(min_size=1)
    info = info[info["community"] == community_id]
    return {
        "community": community_id,
        "label": res.labels.get(community_id, ""),
        "stats": json.loads(info.to_json(orient="records"))[0] if len(info) else {},
        "top_keywords": res.top_keywords(community_id),
        "top_terms": res.top_terms(community_id),
        "central_papers": [_paper_dict(r, abstract_chars) for _, r in papers.iterrows()],
    }


@mcp.tool()
def get_outliers(run_id: str, limit: int = 20) -> List[Dict[str, Any]]:
    """Papers with no S3 link above the chosen threshold (the "irrelevant" set),
    most cited first, with their nearest similarity and suggested community."""
    res = _get(run_id)
    out = []
    for _, r in res.outliers().head(limit).iterrows():
        d = _paper_dict(r, 300)
        d["nearest_similarity"] = None if pd.isna(r["nearest_similarity"]) else round(float(r["nearest_similarity"]), 3)
        d["suggested_community"] = None if pd.isna(r["suggested_community"]) else int(r["suggested_community"])
        out.append(d)
    return out


@mcp.tool()
def get_citation_structure(run_id: str) -> Dict[str, Any]:
    """In-corpus citation flows between the largest communities (rows cite columns),
    plus QI (within-community citations / size) and MI (share of field citations).
    Needs References and EID columns in the export."""
    res = _get(run_id)
    if not res.citation_pairs:
        return {"available": False,
                "reason": "No in-corpus citations found (export needs References and EID)."}
    C = res.citation_matrix()
    return {"available": True,
            "n_citation_links": len(res.citation_pairs),
            "community_ids": [int(c) for c in C.index],
            "matrix": C.values.tolist(),
            "metrics": json.loads(res.citation_metrics().to_json(orient="records"))}


@mcp.tool()
def get_knowledge_tree(run_id: str, window_years: int = 5, n_communities: int = 15,
                       min_links: int = 2) -> Dict[str, Any]:
    """Knowledge tree: papers per community in each time window, and links from earlier to
    later windows weighted by citations and by S3 similarity links. Use it to describe how
    communities emerged, which earlier work each one builds on, and where knowledge flowed."""
    kt = _get(run_id).knowledge_tree(window_years, n_communities)
    links = kt["links"]
    if not links.empty:
        links = links[(links["citations"] + links["similarity_links"]) >= min_links]
    return {"windows": kt["windows"], "citation_source": kt.get("citation_source"),
            "nodes": json.loads(kt["nodes"].to_json(orient="records")),
            "links": json.loads(links.sort_values(["from_window", "to_window"]).to_json(orient="records"))}


@mcp.tool()
def search_papers(run_id: str, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
    """Semantic search over the run's papers (same embedding model as S3)."""
    res = _get(run_id)
    hits = res.search(query, top_k)
    out = []
    for _, r in hits.iterrows():
        d = _paper_dict(r, 300)
        d["score"] = round(float(r["score"]), 4)
        d["community"] = None if pd.isna(r["community"]) else int(r["community"])
        out.append(d)
    return out


@mcp.tool()
def save_community_labels(run_id: str, labels: Dict[str, str]) -> Dict[str, Any]:
    """Store short topic labels for communities, e.g. {"0": "Open innovation", "1": "..."}.
    Labels are saved with the run and included in exports."""
    res = _get(run_id)
    for k, v in labels.items():
        res.labels[int(k)] = v.strip()
    res.save(_run_dir(run_id))
    return {"saved": len(labels), "labels": {str(k): v for k, v in sorted(res.labels.items())}}


@mcp.tool()
def export_results(run_id: str, output_dir: Optional[str] = None) -> Dict[str, str]:
    """Write CSV outputs (papers with community + label, community table, threshold
    scan, citation matrix) to output_dir (default: the run folder)."""
    res = _get(run_id)
    return res.export(output_dir or (_run_dir(run_id) / "export"))


@mcp.tool()
def export_networks(run_id: str, output_dir: Optional[str] = None) -> Dict[str, Any]:
    """Export the networks behind S3 at three levels: one-mode (papers tied by S3 similarity
    and/or citations), two-mode (papers + authors, keywords, journals, institutions,
    countries, funders or references) and a knowledge graph (all node and tie types together,
    with communities as nodes). Writes node/edge CSVs (Neo4j, igraph) and GraphML (Gephi)
    to output_dir (default: the run folder) and returns the files and a summary of what
    each network contains."""
    res = _get(run_id)
    files = res.export_networks(output_dir or (_run_dir(run_id) / "networks"))
    return {"summary": json.loads(res.networks().to_json(orient="records")), "files": files}


@mcp.tool()
def list_questions(run_id: str, include_prediction: bool = False, level: int = 2) -> List[Dict[str, Any]]:
    """The ranked research questions of the S3 method, in three levels:
    level 1 = the published S3 method (text similarity map; each question says whether it is
    in the S3 paper or an extension), level 2 (default) = + two-mode networks and knowledge
    graph (authors, institutions, journals, funders, layer overlap, homophily, citation
    direction), level 3 = + prediction (link prediction Q24-Q29, GNNs Q30-Q34; only when the
    user asks; include_prediction=True is the same as level 3). Shows priority and whether
    this dataset has the columns each question needs."""
    t = _get(run_id).questions(prediction=True if include_prediction else None,
                               level=3 if include_prediction else level)
    return json.loads(t.to_json(orient="records"))


@mcp.tool()
def answer_question(run_id: str, question_id: str, max_rows: int = 20,
                    prediction: bool = False) -> Dict[str, Any]:
    """Answer one ranked question, e.g. "Q8" (foundational papers) or "Q21"
    (interdisciplinarity). Returns findings (plain sentences) and result tables.
    The prediction questions Q24-Q34 are off unless prediction=True (only when the user
    asks for prediction)."""
    return _get(run_id).answer(question_id, prediction=prediction or None).to_dict(max_rows=max_rows)


@mcp.tool()
def build_report(run_id: str, output_path: Optional[str] = None,
                 title: Optional[str] = None, prediction: bool = False, level: int = 2,
                 layout: str = "review") -> Dict[str, str]:
    """Write one self-contained HTML report. layout="review" (default) is organised for a
    literature review: a one-page brief, scope (records in and out), one card per theme
    (what it is about, papers to read first, growth, links to other themes), how themes
    connect, evolution, research agenda (gaps) and a methods appendix, with RIS/BibTeX reading
    lists and a methods paragraph to download. layout="questions" lists every question.
    level: 1 = the published S3 method, 2 = + two-mode and knowledge graph (default),
    3 = + prediction (prediction=True is the same; a few minutes more). Default path: the run folder."""
    res = _get(run_id)
    p = res.report(output_path or (_run_dir(run_id) / "s3_report.html"), title,
                   prediction=True if prediction else None, level=3 if prediction else level, layout=layout)
    return {"report": str(p)}


@mcp.prompt()
def s3_guided(run_id: str) -> str:
    """Walk through the S3 research questions one stage at a time."""
    return (
        f"Guide me through the S3 analysis of run '{run_id}'. Call list_questions, then go "
        "stage by stage (1 Trust the map → 6 Interdisciplinarity). For each available "
        "question, call answer_question and explain the result in 2-4 sentences in plain "
        "language, quoting the key numbers and paper titles. Label communities first "
        "(get_community + save_community_labels) so later answers use the labels. After "
        "each stage, ask whether to continue. At the end, call build_report."
    )


@mcp.prompt()
def s3_report(run_id: str) -> str:
    """Write an S3 intellectual-structure report for a run."""
    return (
        f"Using the s3-bibliometric tools on run '{run_id}':\n"
        "1. Call list_communities and get_threshold_scan; report the chosen threshold, "
        "silhouette and SII, and how many papers were clustered vs. left as outliers.\n"
        "2. For each community with at least 2% of the clustered papers, call "
        "get_community and give it a 2-5 word label grounded in its central papers, "
        "keywords and terms. Save all labels with save_community_labels.\n"
        "3. Write a table: community, label, size, year span, total citations, QI/MI "
        "(if available), 2-3 defining papers (title, year).\n"
        "4. If get_citation_structure is available, describe which communities cite "
        "each other and which are isolated.\n"
        "5. Summarise the field's intellectual structure and research gaps in 3-5 "
        "paragraphs. Only state what the tool outputs support."
    )


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
