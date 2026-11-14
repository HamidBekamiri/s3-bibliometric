# S3 Bibliometric

> **S3 project: [github.com/AI-Growth-Lab/S3](https://github.com/AI-Growth-Lab/S3)**: the method,
> the paper, the S3 web app and proposals for new analyses.
> **Paper: [doi.org/10.1016/j.jbusres.2026.116582](https://doi.org/10.1016/j.jbusres.2026.116582)**. This repository holds the code of the
> `s3-bibliometric` Python library and MCP server.

**Semantic Similarity Score (S3): a bibliometric method for mapping intellectual
structures**, packaged as a free Python library and an MCP server that LLM assistants
(Claude, and any other MCP client) can call.

Give it a Scopus export. It finds the research communities of the field, their central
papers, outliers, and the citation flows between communities. Your AI assistant can then
label the communities and write up the intellectual structure.

**No coding needed**: see [Use it without coding: Claude, ChatGPT or Colab](#use-it-without-coding-claude-chatgpt-or-colab)
for step-by-step setup with Claude Desktop, Claude Code, Claude.ai, ChatGPT or a Google Colab notebook,
or the visual guides: [S3 step by step](https://ai-growth-lab.github.io/S3/) and
[Scopus export for S3](https://ai-growth-lab.github.io/S3/scopus-export.html).

## How S3 works

The library follows the published method: Bekamiri, Østergaard & Timmermans (2027),
*Semantic similarity score: A new method for mapping intellectual structures in business
research*, Journal of Business Research 219, 116582. https://doi.org/10.1016/j.jbusres.2026.116582

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

**Exact, complete references (optional).** A Scopus CSV export can miss references: some
records have none, and a long reference list can be cut at 32,767 characters, the cell limit
of spreadsheet programs (do not open and re-save the export in Excel). By default S3 finds
citations by matching titles in the References text and builds bibliographic coupling from
the reference strings. With a Scopus API key (`SCOPUS_API_KEY`, from dev.elsevier.com),
`enrich_citations` fetches every paper's full reference list as Scopus IDs, using the EID
column of your original export (no new export needed):

- in-corpus citations become exact (direct citation, co-citation, QI and MI);
- bibliographic coupling counts shared cited Scopus IDs instead of matching reference text,
  for every paper the API returns references for (the others keep their References text);
- if the map uses citation relations, or was chosen with `map_weights="auto"`, the map and
  communities are rebuilt on the exact data (saved labels are dropped, so label afterwards).

This needs Scopus entitlement: your institution's network or VPN, or an institutional token
(`SCOPUS_INST_TOKEN`). It makes about one request per paper (one per 200 references); Elsevier
sets a weekly quota per key, shown on the developer portal. Results are cached in
`~/s3bib_runs/_scopus`, so an interrupted run continues where it stopped.

```bash
export SCOPUS_API_KEY="your-key"      # in your terminal, never in a chat
s3bib report scopus.csv --scopus --map auto -o report.html
```

```python
res = s3bib.run_s3("scopus.csv", s3bib.S3Config(model_name="specter"))
s3bib.enrich_citations(res)                 # exact citations via the Scopus API
s3bib.S3Config(threshold=0.78)              # reproduce a published threshold
s3bib.S3Config(community_method="louvain")  # reproduce the S3 web app instead
s3bib.S3Config(dedupe=False)                # keep duplicates
s3bib.S3Config(mi_denominator="all")        # MI over all in-corpus links
s3bib.S3Config(citation_match="strict")     # short titles need year + first author too
s3bib.S3Config(map_weights="text+coupling") # one of four maps (below)
s3bib.S3Config(map_weights="auto")          # compare the maps first, use the recommended one
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

**Which map? Compare them before the analysis.** `compare_maps` builds each candidate map
(threshold scan and Leiden only, so it takes minutes, not a full run per map) and scores it
on measures that do not favour one kind of map:

| Measure | What it checks |
|---|---|
| **Held-out citations** | 20% of the in-corpus citation links are hidden while the maps are built. Share of hidden links that fall inside one community, minus the share for random communities of the same sizes. A citation map is not judged on its own input. |
| **TCI gain over random** | Text coherence of the communities (paper Table 5) beyond random communities of the same sizes |
| **Stability** | Adjusted Rand index with the partitions from two other seeds and two 90% samples of the papers |
| **Coverage** | Share of papers placed in a community |
| Agreement with text | Adjusted Rand index with the text map: how different the result is (reported, not used) |

A rule fixed in advance recommends a map: keep the text map (the published method) unless a
hybrid recovers at least 1 point more held-out citations beyond random, keeps the TCI gain
(at most 0.005 lower), and loses at most 0.05 stability and 5 points coverage. Among the
hybrids that pass, the highest held-out gain wins. The table also gives a 95% bootstrap interval
(over the hidden links) of each map's difference from the text map, so you can see whether a win
is larger than sampling noise. Relations that give a tie to fewer than 5% of the papers are left
out, with the reason. `extended=True` also tries 10% and 50% citation
weight and the significant-ties variants.

```python
from s3bib.mapchoice import compare_maps
ch = compare_maps("scopus.csv", s3bib.S3Config(cache_dir="~/s3bib_cache"))
ch["table"], ch["recommended"], ch["reasons"]
res = s3bib.run_s3("scopus.csv", s3bib.S3Config(map_weights="auto"))   # compare, then run on the winner
s3bib.enrich_citations(res)        # exact Scopus reference lists, then
res.compare_maps(refresh=True)     # compare the maps again on them
```

```bash
s3bib compare-maps scopus.csv            # the table and the recommendation
s3bib report scopus.csv --map auto       # compare, then the full report on the recommended map
```

The report shows the comparison as **Q0** at the top of the methods appendix, and the methods
paragraph says which map was used and why. In Claude, the `compare_maps` tool runs it before
`run_s3_analysis`.

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
| **One-mode, multiplex** | papers | SIMILAR_TO + CITES (+ coupling, co-citation, co-authorship in Q35–Q37) | Q4, Q8, Q10, Q11, Q24, Q35–Q37 |
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

## Three levels of analysis

| Level | What it adds | Questions |
|---|---|---|
| **1. S3** — the published method | The text-similarity map and what the S3 paper reports from it. Questions marked *S3 paper* are in the paper; *extension* questions use the same map. | **S3 paper:** Q1 threshold (SII) · Q2 structure quality · Q3 outliers · Q38 evaluation against coupling, BERTopic and random clusters (paper Tables 5–7) · Q5 communities · Q6 topics · Q7 core papers · Q8 local vs global foundations · Q10 citation flows, QI, MI (Tables 1–2). **Extensions:** Q0 which map to use (text alone or text plus citations) · Q9 closeness · Q11 knowledge gaps · Q12 bridges · Q13 growth · Q14 concepts · Q21–Q23 interdisciplinarity |
| **2. Two-mode networks and knowledge graph** (default) | Authors, institutions, countries, journals, funders and references; the map options that mix in citation relations | Q4 agreement with coupling and co-citation · Q15–Q20 actors · Q35 layer overlap and projection check · Q36 homophily · Q37 citation direction |
| **3. Prediction** (off unless asked) | Link prediction and graph neural networks | Q24–Q34 |

Each level includes the ones below it. Choose it for a run or per call:

```python
res = s3bib.run_s3("scopus.csv", s3bib.S3Config(level=1))   # only the published method
res.questions(level=2)                                     # the ranked list, and which ones your data supports
res.answer("Q8")                                           # findings + tables for one question
res.report("report.html", level=1)                         # the report at one level
```

```bash
s3bib report scopus.csv -o report.html --level 1
s3bib questions --level 1
```

Q38 compares the methods as in the paper: the Topic Coherence Index (TCI = 1 − Jensen–Shannon
divergence between each paper's TF-IDF words and its community's), the silhouette on the SBERT
embeddings for every method, and QI. BERTopic is included when installed
(`pip install "s3-bibliometric[bertopic]"`). Absolute values depend on text cleaning, so compare
the methods within one run rather than with the numbers printed in the paper.

## The report: organised for a literature review

`res.report("report.html")` writes one self-contained HTML page in the order a review is written:

1. **The field at a glance** — the numbers, the themes, one paper per theme to read first, the
   foundations the field builds on, where it is going, the gaps to discuss (and leading authors at level 2).
2. **Scope** — records in and out (PRISMA-style counts) and papers that do not belong.
3. **Themes** — one card per theme: terms and keywords, papers to read first, foundations,
   growth, which themes it builds on and is used by, the closest themes in content, and at
   level 2 its authors, countries, journals and funders. Each card is one subsection of your review.
4. **How themes connect**, **Evolution**, **Who does the work** (level 2).
5. **Research agenda** — knowledge gaps, emerging themes, collaboration gaps (level 2) and
   forecasts (level 3), for your discussion and future-research sections.
6. **Methods appendix** — a methods paragraph to adapt (with the references), a glossary, and the
   analyses that show the map can be trusted.

Every result carries a *For your review* note and a tag: **L1 · S3 paper**, **L1 · extension**,
**L2** or **L3**. When the map mixes in citation relations, the report says that even the
level-1 results are no longer the published method. Downloads for writing:

```python
res.export_review("review/")   # reading_list.ris (Zotero, EndNote, Mendeley), reading_list.bib,
                               # themes.csv, prisma.csv, methods.txt
res.report("report.html", layout="questions")   # the previous layout: every question in order
```

**Prediction (level 3, stages 7–8) is off by default** (`S3Config(prediction=False)`, `level=2`). Until you
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

**Multiplex, homophily and direction (Q35–Q37).** The same papers are tied in five ways:
text (the map's S3 links), bibliographic coupling, co-citation, direct citation and
co-authorship.

| Question | Measures | What it tells you |
|---|---|---|
| Q35 | Jaccard overlap between layers; lift = how many times more often a tie of one layer is also a tie of another than a random pair | Which relations the text map misses, so which map option (`map_weights`) adds the most |
| Q36 | Attribute assortativity r (−1 to +1) per attribute and layer, with the within-group share expected by chance | Whether ties stay inside the same community, journal, country or funder, and in which layer |
| Q37 | PageRank on in-corpus citations; per community a source score (+1 only cited, −1 only citing); reciprocity of flows between communities against random direction | Which papers and communities are knowledge sources, and whether knowledge moves one way |

A paper can only cite older papers, so reciprocity is measured between communities, not papers.

**Projection check (Q35).** Coupling, co-citation and co-authorship are one-mode projections of
two-mode networks (papers–references, cited–citing papers, papers–authors). Projections
inflate ties and clustering: one popular reference links every paper that cites it, and a
ten-author paper makes a clique of ten (Jankowski et al. 2026, *Commun. Phys.* 9:35). Q35 therefore
reports, for each projected layer:

- **Significant ties:** two papers share more items than chance predicts from how many items
  each has and how popular the items are (stochastic degree sequence model with Chung–Lu
  probabilities, Poisson test, Benjamini–Hochberg FDR 0.05 over all paper pairs). With the ties
  shuffled and all degrees kept, no tie is significant.
- **Four-cycle clustering** of the two-mode network itself (Robins & Alexander), next to the
  clustering of the projection, so you can see how much clustering the projection creates.

To build the map and Q35–Q37 from significant ties only:

```python
res = s3bib.run_s3("scopus.csv", s3bib.S3Config(map_weights="text+coupling", backbone=True))
```

```bash
s3bib report scopus.csv --map text+coupling --backbone
```

```python
from s3bib.layers import layers, overlap, homophily, pagerank, flow_direction
J, lift, summary = overlap(res)      # Q35 tables
homophily(res)                       # Q36: attribute, layer, r
pagerank(res)                        # Q37: PageRank per paper
```

**Charts or tables.** Result sections are closed until you click them open. Tables open as bar
charts; switch any of them to the table (and pick the column to plot), or switch all of them
at once at the top of the report. Q6 shows one word cloud per main
community (the top 10, five per row): 30 words each by default, 10 to 50 with one menu.

In Claude, use the **`s3_guided`** prompt, or ask: *"Walk me through the S3 questions for
my Scopus file."*

## Use it without coding: Claude, ChatGPT or Colab

S3 runs on your own computer, so your data stays with you, or in a Google Colab notebook in
your browser. How you use it depends on your app:

| Your app | How | Setup |
|---|---|---|
| **Claude Desktop** (Mac, Windows) | Chat with S3 directly: Claude runs the analysis and explains it | Once: install uv, paste 6 lines into Claude's settings |
| **Claude Code** | Same, from the terminal | Once: install uv, one command |
| **Claude.ai (browser, phone) or ChatGPT** | Make the report with one command, then upload it to the chat | Once: install uv |
| **Google Colab** (browser) or Jupyter | Paste three cells into a notebook; it runs S3 and downloads the report | Nothing to install in Colab |

Claude.ai in the browser and ChatGPT cannot start programs on your computer, so they cannot
run S3 themselves; they can read and explain the report and files S3 makes.

### Step 0: export your papers from Scopus

In Scopus, select all results → **Export** → **CSV**. Tick every information group
(citation, bibliographical, abstract and keywords, funding, and **references**). Title and
abstract are required; the rest (year, citations, authors, affiliations, keywords, references,
funding) unlock more of the analysis. Web of Science and OpenAlex exports work if they have
Title and Abstract columns.

### Step 1: install uv (once)

uv starts S3 for you and installs everything it needs.

- **Mac**: open *Terminal* and paste:
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```
- **Windows**: open *PowerShell* and paste:
  ```powershell
  powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
  ```

Close and reopen the Terminal (or PowerShell) window afterwards.

### Step 2a: Claude Desktop

1. In Claude Desktop open **Settings → Developer → Edit Config**. This opens
   `claude_desktop_config.json`.
2. Paste this and save (if the file already has `"mcpServers"`, add only the
   `"s3-bibliometric"` part inside it):
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
3. Quit Claude Desktop completely and open it again. S3's tools now appear in Claude's
   tools menu in the chat box.
4. Ask, for example:

   > Run S3 on ~/Downloads/scopus.csv. Label the communities, walk me through the
   > level-1 questions, and build the literature-review report.

   Claude asks which language model to use (mpnet, the model of the S3 paper, is the default)
   and whether you have a Scopus API key for exact citations (optional; never paste a key into
   the chat). To use one, add it to the config above, next to `"args"`:
   `"env": {"SCOPUS_API_KEY": "your-key"}`, and restart Claude Desktop. The first run downloads the model (about 420 MB) and takes a few minutes; later
   runs are fast.

**If the tools do not appear**: Claude Desktop may not find `uvx`. In Terminal run
`which uvx` (Mac) or `where uvx` (Windows) and put that full path in `"command"`, e.g.
`"command": "/Users/you/.local/bin/uvx"`.

### Step 2b: Claude Code

```bash
claude mcp add s3-bibliometric -- uvx s3-bibliometric
```

Then ask the same as above.

### Step 2c: Claude.ai in the browser, or ChatGPT

1. Make the report with one command in Terminal (Mac) or PowerShell (Windows). Replace the
   file path with yours:
   ```bash
   uvx --from s3-bibliometric s3bib report ~/Downloads/scopus.csv -o ~/Desktop/s3_report.html
   ```
   It asks which language model to use (press Enter for the default) and whether you have a
   Scopus API key (answer n if not). Add `--level 1` for only the published S3 method.
2. Open `s3_report.html` in your browser. At the top, click the downloads: **Themes table
   (CSV)**, **Methods paragraph**, and the **reading list (RIS)** for Zotero, EndNote or Mendeley.
3. Upload the report and the downloaded files to Claude.ai or ChatGPT, and ask, for example:

   > This is an S3 analysis of my literature. Name each theme in 3-5 words, summarise the
   > field's structure, and draft the structure of my literature review with the gaps to
   > discuss.

### Step 2d: a Google Colab notebook (or Jupyter)

Nothing to install: Colab runs Python in your browser, on Google's servers. Skip step 1.

1. Open [colab.research.google.com](https://colab.research.google.com) → **New notebook**.
   Optional: **Runtime → Change runtime type → T4 GPU** makes the embedding faster.
2. In the first cell, install S3 and press ▶ (or Shift+Enter):
   ```python
   !pip install -q s3-bibliometric
   ```
3. In a new cell, upload your Scopus CSV (pick the file when asked):
   ```python
   from google.colab import files
   csv = list(files.upload())[0]
   ```
4. In a new cell, run S3 and download the report. `map_weights="auto"` first compares the
   maps (text alone or text plus a citation relation) and uses the recommended one:
   ```python
   import s3bib
   res = s3bib.run_s3(csv, s3bib.S3Config(map_weights="auto"))
   print(res.answer("Q0"))            # the map comparison and the map used
   res.report("s3_report.html")       # add level=1 for only the published S3 method
   files.download("s3_report.html")
   ```
   The first run downloads the language model (about 420 MB). Then open the report and use it
   as in step 2c, or keep asking in the notebook, e.g. `print(res.answer("Q8"))` or
   `res.questions()` for the list of questions.

Notes:

- Colab runs on Google's servers, so your file is uploaded there. A free Colab session ends
  after a while; download the report before you close it.
- **Scopus API key in Colab**: click the key icon (**Secrets**), add `SCOPUS_API_KEY`, allow
  the notebook to use it, and run this before the report:
  ```python
  import os
  from google.colab import userdata
  os.environ["SCOPUS_API_KEY"] = userdata.get("SCOPUS_API_KEY")
  s3bib.enrich_citations(res)
  ```
  Reference lists need your institution's Scopus access, and Google's servers are not on your
  campus network, so this usually also needs an institutional token (`SCOPUS_INST_TOKEN`,
  from your library), stored the same way.
- **Jupyter on your own computer**: install with `pip install s3-bibliometric` in your
  environment, skip the upload cell, and use the file path, e.g.
  `csv = "/Users/you/Downloads/scopus.csv"`; open `s3_report.html` instead of downloading it.

### What you can ask

- *"Which questions can S3 answer for my data?"* — the ranked research questions, in three
  levels: **1** the published S3 method, **2** authors, countries, journals, funders and
  citation networks, **3** predictions (only when you ask).
- *"What are the main themes, and which papers should I read first?"*
- *"Where are the knowledge gaps?"* · *"Which themes are emerging?"*
- *"Write the methods paragraph for my paper."*
- *"Build the literature-review report at level 1."*

### The tools Claude uses

| Tool | What it does |
|---|---|
| `compare_maps` | Before the analysis: compare the text map with text+coupling, text+cocitation and text+direct on your corpus and recommend one |
| `run_s3_analysis` | Run S3 on a CSV/XLSX (year range, map option, strict citation matching); returns a `run_id` and a summary |
| `list_embedding_models` | The language models you can choose (default: mpnet, as in the S3 paper) |
| `enrich_citations` | Exact, complete reference lists from the Scopus API: exact citations, and coupling from cited Scopus IDs (needs `SCOPUS_API_KEY` in the server's environment) |
| `list_runs` | Saved runs |
| `list_questions` | The ranked research questions for a level (1, 2 or 3) and whether your data supports each |
| `answer_question` | Findings and tables for one question, e.g. `Q8` |
| `build_report` | The literature-review report (or every question in order), at a level |
| `get_threshold_scan` | Silhouette / SII / communities for each threshold |
| `list_communities` | Community table incl. QI/MI and saved labels |
| `get_community` | Central papers with abstracts, keywords, TF-IDF terms |
| `save_community_labels` | Store the labels the assistant proposes |
| `get_outliers` | Unconnected papers + nearest community |
| `get_citation_structure` | Community × community citation matrix + QI/MI |
| `get_knowledge_tree` | How communities grow and split over time |
| `search_papers` | Semantic search inside the corpus |
| `export_results` | Write CSV outputs |
| `export_networks` | One-mode, two-mode and knowledge-graph networks as CSV + GraphML |

Prompts: **`s3_guided`** walks through the questions stage by stage; **`s3_report`** writes the
intellectual-structure report. Runs are saved in `~/s3bib_runs` (change it with the
`S3BIB_HOME` environment variable).

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

S3 project: [github.com/AI-Growth-Lab/S3](https://github.com/AI-Growth-Lab/S3). To propose a new
analysis, open an issue there; report problems with this library
[here](https://github.com/HamidBekamiri/s3-bibliometric/issues).

If you use this package, please cite the S3 paper:

Bekamiri, H., Østergaard, C. R., & Timmermans, B. (2027). Semantic similarity score: A new method
for mapping intellectual structures in business research. *Journal of Business Research*, 219,
116582. https://doi.org/10.1016/j.jbusres.2026.116582

## License

MIT
