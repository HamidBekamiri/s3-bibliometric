"""Command line: run S3 and write the question-based report.

    s3bib report scopus.csv -o report.html
    s3bib report scopus.csv --model specter --scopus --threshold 0.78
    s3bib compare-maps scopus.csv          which map to use, before the full analysis
    s3bib report scopus.csv --map auto     compare the maps, then run on the recommended one
    s3bib models
    s3bib questions

When run in a terminal without --model or --scopus/--no-scopus, it asks.
"""
from __future__ import annotations

import argparse
import getpass
import os
import sys


def _ask_model() -> str:
    from .models import EMBEDDING_MODELS
    keys = list(EMBEDDING_MODELS)
    print("\nWhich embedding model should S3 use?", file=sys.stderr)
    for k, key in enumerate(keys, 1):
        model, kind, _, desc = EMBEDDING_MODELS[key]
        print(f"  {k}. {key:<12} [{kind}] {desc}", file=sys.stderr)
    ans = input("Number or model name [1 = mpnet, the default]: ").strip()
    if not ans:
        return "mpnet"
    return keys[int(ans) - 1] if ans.isdigit() and 1 <= int(ans) <= len(keys) else ans


def _ask_scopus() -> bool:
    print("\nCitations are found by matching titles in the References text (approximate).\n"
          "With a Scopus API key S3 can fetch exact reference lists instead "
          "(needs your institution's network or an institutional token).", file=sys.stderr)
    return input("Use the Scopus API for exact citations? [y/N]: ").strip().lower() in ("y", "yes")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="s3bib", description="S3 bibliometric analysis",
                                epilog="S3 project: https://github.com/AI-Growth-Lab/S3 · "
                                       "paper: https://doi.org/10.1016/j.jbusres.2026.116582 · "
                                       "library: https://github.com/AI-Growth-Lab/s3-bibliometric")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("report", help="Run S3 on a Scopus export and write an HTML report")
    r.add_argument("csv")
    r.add_argument("-o", "--output", default="s3_report.html")
    r.add_argument("--start-year", type=int)
    r.add_argument("--end-year", type=int)
    r.add_argument("--title")
    r.add_argument("--model", help="Embedding model: mpnet (default), specter, minilm, multilingual, "
                                   "qwen3, clip, or any sentence-transformers model name")
    r.add_argument("--threshold", type=float, help="Fix the S3 threshold (e.g. 0.78) instead of choosing it by SII")
    r.add_argument("--method", choices=["leiden", "louvain"], default="leiden",
                   help="leiden = published method (default); louvain = S3 web app")
    r.add_argument("--no-dedupe", action="store_true", help="Keep duplicate records")
    sc = r.add_mutually_exclusive_group()
    sc.add_argument("--scopus", action="store_true", help="Exact citations from the Scopus API (SCOPUS_API_KEY)")
    sc.add_argument("--no-scopus", action="store_true", help="Don't ask about the Scopus API")
    r.add_argument("--seed", type=int, default=42)
    r.add_argument("--map", default="text",
                   help="Which relations build the map: text (default: S3 text similarity, the published "
                        "method), text+coupling, text+cocitation or text+direct (75%% text + 25%% "
                        "bibliographic coupling, co-citation or direct citation), or auto: compare these "
                        "maps first (as in compare-maps) and use the recommended one. Every result in the "
                        "report follows this choice.")
    r.add_argument("--strict-citations", action="store_true",
                   help="Short titles (e.g. 'Introduction') only count as cited when year and first author match")
    r.add_argument("--backbone", action="store_true",
                   help="Keep only significant coupling, co-citation and co-authorship ties (in the map "
                        "and in Q35-Q37): sharing one popular reference or author is not enough")
    r.add_argument("--prediction", action="store_true",
                   help="Add link prediction and GNN questions (Q24-Q34) to the report (same as --level 3)")
    r.add_argument("--level", type=int, choices=(1, 2, 3), default=2,
                   help="1 = the published S3 method; 2 = + two-mode networks and knowledge graph "
                        "(default); 3 = + prediction")
    r.add_argument("--layout", choices=("review", "questions"), default="review",
                   help="review (default): brief, theme cards, research agenda, methods appendix; "
                        "questions: every question in order")
    r.add_argument("--save", help="Also save the run to this folder (reload with S3Result.load)")
    r.add_argument("--cache-dir", default="~/s3bib_runs/_cache",
                    help="Keep embeddings here, so compare-maps and report embed only once "
                         "(default ~/s3bib_runs/_cache; '' to switch off)")
    qp = sub.add_parser("questions", help="List the ranked research questions")
    qp.add_argument("--prediction", action="store_true", help="Also list the prediction questions (Q24-Q34)")
    qp.add_argument("--level", type=int, choices=(1, 2, 3), default=2,
                    help="1 = the published S3 method; 2 = + two-mode and knowledge graph; 3 = + prediction")
    cm = sub.add_parser("compare-maps", help="Compare the text map with text+coupling, text+cocitation and "
                                             "text+direct on this corpus and recommend one")
    cm.add_argument("csv")
    cm.add_argument("--model", default="mpnet", help="Embedding model (default mpnet)")
    cm.add_argument("--start-year", type=int)
    cm.add_argument("--end-year", type=int)
    cm.add_argument("--seed", type=int, default=42)
    cm.add_argument("--extended", action="store_true",
                    help="Also the 5-40%% grid without the text floor, to show the floor's effect")
    cm.add_argument("--holdout", type=float, default=0.2, help="Share of citation links hidden (default 0.2)")
    cm.add_argument("--strict-citations", action="store_true")
    cm.add_argument("-o", "--output", help="Also write the comparison table to this CSV file")
    cm.add_argument("--cache-dir", default="~/s3bib_runs/_cache",
                    help="Keep embeddings here, so compare-maps and report embed only once "
                         "(default ~/s3bib_runs/_cache; '' to switch off)")
    sub.add_parser("models", help="List the embedding models you can choose")
    a = p.parse_args(argv)

    if a.cmd == "questions":
        from .questions import LEVELS, QUESTIONS, STAGES
        lv = 3 if a.prediction else a.level
        print(f"Level {lv}: {LEVELS[lv]}")
        for k, q in enumerate([q for q in QUESTIONS if q.level <= lv], 1):
            print(f"{k:>2}. {q.qid:<4} [L{q.level} {q.source} · {STAGES[q.stage]} · {q.priority}] {q.text}")
        return
    if a.cmd == "models":
        from .models import model_table
        for m in model_table():
            print(f"{m['key']:<12} {m['model']:<62} [{m['kind']}]{'  (default)' if m['default'] else ''}\n"
                  f"{'':<12} {m['description']}")
        return

    if a.cmd == "compare-maps":
        from .mapchoice import compare_maps
        from .pipeline import S3Config
        cfg = S3Config(model_name=a.model, start_year=a.start_year, end_year=a.end_year, seed=a.seed,
                       citation_match="strict" if a.strict_citations else "lenient",
                       cache_dir=a.cache_dir or None)
        bar = lambda s, x: print(f"\r{x:4.0%}  {s:<50}", end="", file=sys.stderr)
        ch = compare_maps(a.csv, cfg, extended=a.extended, holdout=a.holdout, progress=bar)
        print(file=sys.stderr)
        t = ch["table"].drop(columns=["weights", "significant ties"], errors="ignore")
        cols = ["map", "held-out citations gain", "TCI gain over random", "stability (ARI)", "coverage",
                "agreement with text (ARI)", "communities"]
        print(t[[c for c in cols if c in t]].round(3).to_string(index=False))
        print(f"\n{ch['n_heldout']} citation links hidden of {ch['n_links']}.")
        for s in ch["skipped"]:
            print(f"Not compared: {s['map']}, {s['reason']}.")
        print(f"\nRule: {ch['rule']}")
        print(f"\nRecommended map: {ch['recommended']}")
        for r in ch["reasons"]:
            print(f"  - {r}")
        if a.output:
            t.to_csv(a.output, index=False)
            print(f"Table written to {a.output}")
        return

    interactive = sys.stdin.isatty()
    model = a.model or (_ask_model() if interactive else "mpnet")
    use_scopus = a.scopus or (not a.no_scopus and interactive and _ask_scopus())
    if use_scopus and not os.environ.get("SCOPUS_API_KEY"):
        if not interactive:
            sys.exit("Set SCOPUS_API_KEY to use --scopus.")
        os.environ["SCOPUS_API_KEY"] = getpass.getpass("Scopus API key (not shown, not saved): ").strip()

    from .pipeline import S3Config, run_s3
    cfg = S3Config(model_name=model, start_year=a.start_year, end_year=a.end_year, threshold=a.threshold,
                   community_method=a.method, dedupe=not a.no_dedupe, seed=a.seed,
                   map_weights=a.map, citation_match="strict" if a.strict_citations else "lenient",
                   prediction=a.prediction or a.level == 3, backbone=a.backbone,
                   level=3 if a.prediction else a.level, cache_dir=a.cache_dir or None)
    bar = lambda s, x: print(f"\r{x:4.0%}  {s:<50}", end="", file=sys.stderr)
    res = run_s3(a.csv, cfg, progress=bar)
    print(file=sys.stderr)
    if len(res.removed):
        print(f"Removed {len(res.removed)} records: "
              + ", ".join(f"{v} {k}" for k, v in res.removed["reason"].value_counts().items()), file=sys.stderr)
    if use_scopus:
        from .scopus import enrich_citations
        stats = enrich_citations(res, progress=bar)
        print(f"\nScopus: {stats['citation_links_now']} exact citation links "
              f"(title matching found {stats['citation_links_before']}); {stats['failed']} papers failed.",
              file=sys.stderr)
        src = stats["coupling_reference_source"]
        print(f"Bibliographic coupling uses the Scopus reference IDs for {src['scopus_ids']} papers"
              f" and the References text for {src['text']}." + (" The map was rebuilt on the exact data."
                                                                 if stats["map_rebuilt"] else ""), file=sys.stderr)
    if a.save:
        res.save(a.save)
    out = res.report(a.output, a.title, prediction=True if a.prediction else None,
                     level=3 if a.prediction else a.level, layout=a.layout)
    print(f"Report written to {out}")
    print("S3 project: https://github.com/AI-Growth-Lab/S3 · paper: https://doi.org/10.1016/j.jbusres.2026.116582", file=sys.stderr)


if __name__ == "__main__":
    main()
