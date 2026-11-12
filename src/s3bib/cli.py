"""Command line: run S3 and write the question-based report.

    s3bib report scopus.csv -o report.html [--start-year 2010 --end-year 2024]
    s3bib questions
"""
from __future__ import annotations

import argparse
import sys


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="s3bib", description="S3 bibliometric analysis")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("report", help="Run S3 on a Scopus export and write an HTML report")
    r.add_argument("csv")
    r.add_argument("-o", "--output", default="s3_report.html")
    r.add_argument("--start-year", type=int)
    r.add_argument("--end-year", type=int)
    r.add_argument("--title")
    r.add_argument("--threshold", type=float, help="Fix the S3 threshold (e.g. 0.78) instead of choosing it by SII")
    r.add_argument("--method", choices=["leiden", "louvain"], default="leiden",
                   help="leiden = published method (default); louvain = S3 web app")
    r.add_argument("--save", help="Also save the run to this folder (reload with S3Result.load)")
    sub.add_parser("questions", help="List the ranked research questions")
    a = p.parse_args(argv)

    from .questions import QUESTIONS, STAGES
    if a.cmd == "questions":
        for k, q in enumerate(QUESTIONS, 1):
            print(f"{k:>2}. {q.qid:<4} [{STAGES[q.stage]} · {q.priority}] {q.text}")
        return

    from .pipeline import S3Config, run_s3
    cfg = S3Config(start_year=a.start_year, end_year=a.end_year, threshold=a.threshold,
                   community_method=a.method)
    res = run_s3(a.csv, cfg, progress=lambda s, x: print(f"\r{x:4.0%}  {s:<50}", end="", file=sys.stderr))
    print(file=sys.stderr)
    if a.save:
        res.save(a.save)
    out = res.report(a.output, a.title)
    print(f"Report written to {out}")


if __name__ == "__main__":
    main()
