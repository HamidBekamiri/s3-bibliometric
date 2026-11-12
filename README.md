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

**Reproducible.** One seed (default 42) fixes every random step: Python, NumPy, PyTorch,
Leiden/Louvain, sampling and layouts. Two runs on the same file give identical results.

**Clean input.** Before the analysis S3 drops records with neither title nor abstract and
removes duplicates: the same EID, the same title + abstract, the same DOI with the same
title (book chapters can share a DOI), and near-identical texts (S3 ≥ 0.98, e.g. reprints).
The most-cited copy is kept; `res.removed` lists every dropped record and why.

**Choose the embedding model** (default: mpnet, the model of the published paper):

| key | model | use it for |
|---|---|---|
| `mpnet` | sentence-transformers/all-mpnet-base-v2 | default; the published S3 method |
| `specter` | sentence-transformers/allenai-specter | scientific papers (trained on citations) |
| `minilm` | sentence-transformers/all-MiniLM-L6-v2 | quick looks, about 5× faster |
| `multilingual` | paraphrase-multilingual-mpnet-base-v2 | corpora mixing languages |
| `qwen3` | Qwen/Qwen3-Embedding-0.6B | long abstracts (no truncation); slower |
| `clip` | sentence-transformers/clip-ViT-B-32 | multimodal (text + images); reads only 77 tokens of text |

Any other sentence-transformers model name also works. The threshold range 0.75–0.85 was
calibrated for mpnet; other models get a wider scan (0.40–0.95) unless you set one.

**Exact citations (optional).** By default citations are found by matching titles in the
References text. With a Scopus API key (`SCOPUS_API_KEY`, from dev.elsevier.com) S3 fetches
each paper's reference list from the Scopus API instead, so only real links count. This
needs Scopus entitlement: your institution's network or an institutional token
(`SCOPUS_INST_TOKEN`).

```python
res = s3bib.run_s3("scopus.csv", s3bib.S3Config(model_name="specter"))
s3bib.enrich_citations(res)                 # exact citations via the Scopus API
s3bib.S3Config(threshold=0.78)              # reproduce a published threshold
s3bib.S3Config(community_method="louvain")  # reproduce the S3 web app instead
s3bib.S3Config(dedupe=False)                # keep duplicates
s3bib.S3Config(mi_denominator="all")        # MI over all in-corpus links
s3bib.S3Config(citation_match="strict")     # short titles need year + first author too
s3bib.S3Config(map_weights="text+coupling") # one of four maps (below)
```

**Strict citation matching.** Matching titles inside References text can link a short,
generic title such as "Introduction" to every reference that contains the word. On the
Innovation & Diversity corpus about a quarter of the matched citations came from titles of
25 characters or fewer, and only 11% of those carried the cited paper's year or first
author. `citation_match="strict"` (CLI `--strict-citations`) keeps a short-title match only
when the year (±1) and first author also appear in the reference. The default stays
`"lenient"`, as in the S3 web app and the published paper.

**Choose the map: four options.** The published S3 map uses text similarity alone. You can
add one citation relation at 25%:

| `map_weights` (CLI `--map`) | The map is built from |
|---|---|
| `"text"` (default) | text similarity (S3), the published method |
| `"text+coupling"` | 75% text + 25% bibliographic coupling (shared references) |
| `"text+cocitation"` | 75% text + 25% co-citation (cited together) |
| `"text+direct"` | 75% text + 25% direct citation |

```python
res = s3bib.run_s3("scopus.csv", s3bib.S3Config(map_weights="text+cocitation"))
```

```bash
s3bib report scopus.csv --map text+cocitation
```

Every result follows the choice: the threshold scan, communities, network map, knowledge
tree, citation flows, bridges, interdisciplinarity and the report header (which names the
map). Coupling and co-citation are Salton cosines (0–1). Because the citation part is 0 for
most pairs, the threshold scan matches link density: each threshold links as many pairs as
one S3 threshold (0.75–0.85) on the same corpus, and SII picks the best, as for S3. Custom
mixes also work, e.g. `map_weights={"s3": 0.5, "coupling": 0.5}`.

On the command line, `s3bib report scopus.csv` asks which model to use and whether to use the
Scopus API; in Claude, the assistant asks the same two questions before the first run.

**Reproducing the paper.** On the paper's ‘Innovation and Diversity’ corpus, with
`S3Config(threshold=0.78, dedupe=False)` the run retains exactly the paper's 1,826 papers,
and the ten largest Leiden communities and their QI, local and global citations closely
match Table 1.

**Visuals.** The report includes the paper's figures (S3 network map as in Panel 1, SII
curve as in Fig. 3, elbow plot as in Fig. 4, citation-flow heatmap as in Table 2), an
interactive paper-level network map drawn with
[force-graph](https://github.com/vasturiano/force-graph), the library behind the S3 web app's map
(hover for the paper, click for its abstract, drag, zoom, play the network year by year,
dark/light, click a community to highlight it; a built-in map is used offline), a **knowledge tree** (communities per 5-year window, linked by citations and
S3 similarity from earlier to later periods; switch between citation, similarity or both
links, filter weak links, click a community to follow its lineage), a citation-flow
diagram, a stacked growth chart, and author × community and country × community maps.
Community colours are the same in every chart.

```python
kt = res.knowledge_tree(window_years=5)   # {"windows", "nodes", "links"} as DataFrames
```

## The networks behind S3: one-mode, two-mode, knowledge graph

S3 answers its questions from three levels of network. You can export each of them:

| Level | Nodes | Ties | Questions |
|---|---|---|---|
| **One-mode** | papers | SIMILAR_TO (S3 ≥ threshold) | Q1–Q3, Q5–Q7, Q9, Q12–Q13, Q21–Q23 |
| **One-mode, multiplex** | papers | SIMILAR_TO + CITES | Q4, Q8, Q10, Q11, Q24 |
| **Two-mode** | papers + authors, keywords, journals, institutions, countries, funders or references | AUTHORED_BY, HAS_KEYWORD, … | Q14–Q16, Q19, Q20; co-citation and bibliographic coupling (Q4) are projections of the paper–reference network |
| **Knowledge graph** | all of the above + communities | all tie types + IN_COMMUNITY, LOCATED_IN | Q17, Q18 |

Communities are not in the data: S3 derives them from the one-mode similarity network. The
knowledge graph adds them as nodes so that questions across communities become graph queries.

```python
res.networks()                        # node and tie types and counts per level
G = res.network("kg")                 # networkx MultiDiGraph; also "similarity", "citation",
                                      # "one-mode", "paper-author", "paper-keyword", ...
res.export_networks("networks/")      # node/edge CSVs (Neo4j, igraph) + GraphML (Gephi)
                                      # add two_mode_kinds=[..., "reference"] for the large
                                      # paper-reference network
```

## The guided pipeline: 34 research questions

Every analysis answers one research question. Go through them in order, or jump to the
ones you need. Questions whose data is missing from your export are skipped and say why.

| Stage | Questions |
|---|---|
| **1. Trust the map** | Q1 Best similarity threshold (SII) · Q2 Strength of the community structure (silhouette, modularity) · Q3 Papers that don't belong · Q4 Agreement with bibliographic coupling and co-citation |
| **2. Map the field** | Q5 Main communities · Q6 What each is about · Q7 Core papers (centrality) · Q8 Foundational papers, local vs global (Appendix Table C1) · Q9 How close communities are |
| **3. Knowledge flows** | Q10 Citation flows, QI and MI (Tables 1–2) · Q11 Knowledge gaps (similar but not citing) · Q12 Papers that bridge communities |
| **4. Evolution** | Q13 Emerging, growing and declining communities · Q14 Rising and fading concepts |
| **5. Actors** | Q15 Leading authors (RFI) · Q16 Broker authors · Q17 Collaboration gaps · Q18 Countries and institutions · Q19 Journals · Q20 Funders |
| **6. Interdisciplinarity** | Q21 How interdisciplinary each community is (Rao-Stirling with S3 distances) · Q22 Are interdisciplinary papers cited more? · Q23 Is the field becoming more interdisciplinary? |
| **7. Link prediction** *(off by default)* | Q24 Which earlier papers will new papers cite? · Q25 Which knowledge gaps will close? · Q26 Which concepts will enter which communities? · Q27 Which authors will move into a new community? · Q28 Who will co-author next? · Q29 Which institutions and funders will enter which communities? |
| **8. Graph learning (GNN)** *(off by default)* | Q30 Community of new or outlier papers (GraphSAGE) · Q31 Future impact: content or network position? (GraphSAGE regression) · Q32 Expertise search and partner matching (R-GCN on the knowledge graph) · Q33 Which communities will grow or decline next? (GCN + GRU) · Q34 Missing links in the knowledge graph (TransE, RotatE) |

```python
res.questions()              # the ranked list, and which ones your data supports
res.answer("Q8")             # findings + tables for one question
res.report("report.html")    # every answer in one HTML report with charts
```

From the command line:

```bash
s3bib report scopus.csv -o report.html
```

**Prediction (stages 7–8) is off by default** (`S3Config(prediction=False)`). Until you
switch it on, Q24–Q34 do not appear in the question list, the report, the CLI or Claude, and
asking for one returns a note that prediction is off. Switch it on for a whole run, or for
one call:

```python
res = s3bib.run_s3("scopus.csv", s3bib.S3Config(prediction=True))   # everything on for this run
res.report("report.html")                    # then includes Q24–Q34
res.answer("Q30")
res.find_experts("board diversity and green innovation", kind="author")   # R-GCN (Q32)

res.report("report.html", prediction=True)   # or per call, on a run without it
res.answer("Q30", prediction=True)
```

```bash
s3bib report scopus.csv --prediction
s3bib questions --prediction
```

Every prediction is tested honestly before it is used: the corpus is split in time (the last
5 years are the test window), models see only information from before the split, and each
result is compared with simple baselines (popularity, community size, Adamic-Adar, nearest
paper) with 95% intervals, so the report says when a model does not beat chance. The GNNs
(GraphSAGE, R-GCN, a temporal GCN + GRU, TransE and RotatE) are small and written in plain
PyTorch, which sentence-transformers already installs; nothing else is needed.

**Charts or tables.** Result sections are closed until you click them open. Tables open as bar
charts; switch any of them to the table (and pick the column to plot), or switch all of them
at once at the top of the report. Q6 has a word cloud per community (or the whole field):
10 words by default, up to 50.

In Claude, use the **`s3_guided`** prompt, or ask: *"Walk me through the S3 questions for
my Scopus file."*

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
| `export_networks` | One-mode, two-mode and knowledge-graph networks as CSV + GraphML |

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

## Tests

The tests run on real data: 109 [OpenAlex](https://openalex.org) records (CC0) on absorptive
capacity and work group diversity, with their real all-mpnet-base-v2 embeddings, in
`tests/data/`. `python tests/data/build_openalex_sample.py` rebuilds them.

## Citation

If you use this package, please cite the S3 paper: *Semantic Similarity Score: A New
Bibliometric Method for Mapping Intellectual Structures in Business Research.*

## License

MIT
