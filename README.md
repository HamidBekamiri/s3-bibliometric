# S3 Bibliometric

**Semantic Similarity Score (S3): a bibliometric method for mapping intellectual
structures**, packaged as a free Python library and an MCP server that LLM assistants
(Claude, and any other MCP client) can call.

Give it a Scopus export. It finds the research communities of the field, their central
papers, outliers, and the citation flows between communities. Your AI assistant can then
label the communities and write up the intellectual structure.

## How S3 works

The library follows the published method: Bekamiri, Østergaard & Timmermans (2027),
*Semantic similarity score: A new method for mapping intellectual structures in business
research*, Journal of Business Research 219, 116582.

1. Embed each paper's `Title. Abstract` with `sentence-transformers/all-mpnet-base-v2`
2. **S3 matrix** = cosine similarity between all papers, time-masked so links run from
   earlier to later papers (year_i ≤ year_j)
3. Scan thresholds 0.75–0.85: papers with at least one link S3 ≥ t are retained;
   **Leiden with directed modularity** finds the communities
4. Keep the threshold with the highest **SII = mean silhouette of retained papers ×
   number retained**
5. Describe communities with eigenvector and degree centrality and **c-TF-IDF** terms
6. With `References` + `EID`: in-corpus citations, **QI** (within-community citations per
   paper) and **MI** (the community's within + inward citations as a share of the field's
   citation links)

Leiden uses a fixed seed (default 42), so results are reproducible.

```python
s3bib.S3Config(threshold=0.78)            # reproduce a published threshold
s3bib.S3Config(community_method="louvain") # reproduce the S3 web app instead
s3bib.S3Config(mi_denominator="all")       # MI over all in-corpus links
```

**Reproducing the paper.** On the paper's ‘Innovation and Diversity’ corpus, a fixed
threshold of 0.78 retains exactly the paper's 1,826 papers, and the ten largest Leiden
communities and their QI, local and global citations closely match Table 1.

## The guided pipeline: 23 research questions

Every analysis answers one research question. Go through them in order, or jump to the
ones you need. Questions whose data is missing from your export are skipped and say why.

| Stage | Questions |
|---|---|
| **1. Trust the map** | Q1 Best similarity threshold (SII) · Q2 Strength of the community structure (silhouette, modularity) · Q3 Papers that don't belong · Q4 Agreement with bibliographic coupling and co-citation |
| **2. Map the field** | Q5 Main communities · Q6 What each is about · Q7 Core papers (centrality) · Q8 Foundational papers, local vs global (Table 3) · Q9 How close communities are |
| **3. Knowledge flows** | Q10 Citation flows, QI and MI (Table 2) · Q11 Knowledge gaps (similar but not citing) · Q12 Papers that bridge communities |
| **4. Evolution** | Q13 Emerging, growing and declining communities · Q14 Rising and fading concepts |
| **5. Actors** | Q15 Leading authors (RFI) · Q16 Broker authors · Q17 Collaboration gaps · Q18 Countries and institutions · Q19 Journals · Q20 Funders |
| **6. Interdisciplinarity** | Q21 How interdisciplinary each community is (Rao-Stirling with S3 distances) · Q22 Are interdisciplinary papers cited more? · Q23 Is the field becoming more interdisciplinary? |

```python
res.questions()              # the ranked list, and which ones your data supports
res.answer("Q8")             # findings + tables for one question
res.report("report.html")    # every answer in one HTML report with charts
```

From the command line:

```bash
s3bib report scopus.csv -o report.html
```

In Claude, use the **`s3_guided`** prompt, or ask: *"Walk me through the S3 questions for
my Scopus file."* See `examples/sample_report.html` for a report on synthetic data.

## Use it from Claude (no install, no API key)

You need [uv](https://docs.astral.sh/uv/getting-started/installation/).

**Claude Code**

```bash
claude mcp add s3-bibliometric -- uvx s3-bibliometric
```

**Claude Desktop**: add this to `claude_desktop_config.json` (Settings → Developer → Edit Config):

```json
{
  "mcpServers": {
    "s3-bibliometric": {
      "command": "uvx",
      "args": ["s3-bibliometric"]
    }
  }
}
```

Then ask, for example:

> Run S3 on ~/Downloads/scopus_innovation.csv, label the communities, and write the
> intellectual-structure report.

You can also use the built-in **`s3_report`** prompt. The first run downloads the model
(~420 MB) and PyTorch, so it takes a few minutes. Later runs are fast.

| Tool | What it does |
|---|---|
| `run_s3_analysis` | Run S3 on a CSV/XLSX (optional year range); returns a `run_id` and a summary |
| `list_runs` | Saved runs |
| `get_threshold_scan` | Silhouette / SII / communities for each threshold |
| `list_communities` | Community table incl. QI/MI and saved labels |
| `get_community` | Central papers with abstracts, keywords, TF-IDF terms |
| `get_outliers` | Unconnected papers + nearest community |
| `get_citation_structure` | Community × community citation matrix + QI/MI |
| `search_papers` | Semantic search inside the corpus |
| `save_community_labels` | Store the labels the assistant proposes |
| `export_results` | Write CSV outputs |

Runs are saved in `~/s3bib_runs` (change it with the `S3BIB_HOME` environment variable).

## Use it from Python

```bash
pip install s3-bibliometric
```

```python
import s3bib

res = s3bib.run_s3("scopus.csv")       # or run_s3(df, s3bib.S3Config(start_year=2010))
res.scan                               # threshold scan
res.communities()                      # size, citations, years, QI/MI
res.community_papers(0, n=10)          # most central papers
res.top_keywords(0); res.top_terms(0)
res.outliers()
res.citation_matrix()
res.search("absorptive capacity")
res.labels = {0: "Open innovation", 1: "M&A"}
res.export("results/")                 # CSVs
res.save("runs/innovation")            # reload with s3bib.S3Result.load(...)
```

## Input

Export from Scopus as CSV with at least **Title** and **Abstract**. Also used if present:
Year, Cited by, Authors, Author Keywords, Source title, DOI, and **References + EID**
(needed for the citation analysis).

## Notes

- On a CPU, embedding 5,000 abstracts takes a few minutes. The silhouette step uses an
  N×N matrix (~100 MB at N = 5,000).
- A paper counts as citing another if the first 60 characters of the other paper's
  normalised title appear in its References. This step is O(N²).
- `examples/make_sample.py` creates a small synthetic dataset for testing.

## Citation

If you use this package, please cite the S3 paper: *Semantic Similarity Score: A New
Bibliometric Method for Mapping Intellectual Structures in Business Research.*

## License

MIT
