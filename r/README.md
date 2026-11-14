# s3bibliometric: S3 for R

The R package for **S3 (Semantic Similarity Score)**, the method of Bekamiri, Østergaard &
Timmermans (2027) for mapping the intellectual structure of a research field from what its papers
say. It runs the [s3-bibliometric](https://github.com/AI-Growth-Lab/s3-bibliometric) Python library
of the **same version**, so R and Python give identical results, and every improvement to the
Python library reaches R with the next release.

S3 project: [github.com/AI-Growth-Lab/S3](https://github.com/AI-Growth-Lab/S3) ·
Paper: [doi.org/10.1016/j.jbusres.2026.116582](https://doi.org/10.1016/j.jbusres.2026.116582) ·
Guide: [ai-growth-lab.github.io/S3](https://ai-growth-lab.github.io/S3/)

## Install

```r
install.packages("remotes")
remotes::install_github("AI-Growth-Lab/s3-bibliometric", subdir = "r")
```

The first time you use it, [reticulate](https://rstudio.github.io/reticulate/) sets up Python with
the matching `s3-bibliometric` library for you. The first analysis also downloads the language
model (about 420 MB). If you prefer your own Python environment, run `s3_install()` once and
restart R.

## Quick start

```r
library(s3bibliometric)

res <- s3_run("scopus.csv")          # your Scopus export (Title and Abstract required)
res                                  # threshold, communities, outliers
s3_communities(res)                  # the research themes, with citations, QI and MI
s3_answer(res, "Q8")                 # local vs global foundations of each theme
s3_report(res, "s3_report.html")     # the full report, organised for a literature review
```

Try it on the included real example, 109 OpenAlex records (CC0) on absorptive capacity and work
group diversity: `res <- s3_run(s3_example())`.

## Functions

| Function | What it does |
|---|---|
| `s3_run()` | Run S3 on a Scopus export (file or data frame): model, map, threshold, years, level |
| `s3_compare_maps()` | Before the analysis: text map vs text plus a citation relation, with a recommendation; or use `s3_run(map = "hybrid")` |
| `s3_communities()`, `s3_papers()`, `s3_scan()` | Communities, papers and the threshold scan as data frames |
| `s3_questions()`, `s3_answer()` | The ranked research questions, and findings plus tables for one question |
| `s3_report()` | The HTML report (literature-review layout, or every question in order) |
| `s3_search()` | Papers closest in meaning to a query |
| `s3_enrich_citations()` | Complete reference lists from the Scopus API (`SCOPUS_API_KEY` in `~/.Renviron`) |
| `s3_save()`, `s3_load()`, `s3_export()` | Save and reload a run (also readable from Python), or write CSV files |
| `s3_models()`, `s3_version()`, `s3_install()` | Embedding models, versions, and Python setup |

Answers come back as R objects: `s3_answer(res, "Q5")$tables[[1]]` is a data frame you can plot
or join with your own data.

## Scopus API key

Put the key in `~/.Renviron` (never in a script you share), restart R, then:

```r
s3_enrich_citations(res)
```

```
SCOPUS_API_KEY=your-key
```

Reference lists need your institution's Scopus access (campus network, VPN or an institutional
token in `SCOPUS_INST_TOKEN`).

## How R and Python stay in step

The R package version always equals the Python library version (a test in the repository fails if
they differ), and on load the package asks reticulate for exactly that Python version. Each
release updates both. `s3_version()` shows the two.

## Citation

```r
citation("s3bibliometric")
```

Bekamiri, H., Østergaard, C. R., & Timmermans, B. (2027). Semantic similarity score: A new method
for mapping intellectual structures in business research. *Journal of Business Research*, 219,
116582. https://doi.org/10.1016/j.jbusres.2026.116582
