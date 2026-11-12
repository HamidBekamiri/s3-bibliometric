"""The networks behind S3, as tables you can export (Gephi, Neo4j, igraph, networkx).

Three levels, from simple to rich:

  one-mode   papers only. Tie types: SIMILAR_TO (S3 >= the chosen threshold, weight = S3)
             and CITES (citing -> cited paper inside the corpus). Both together = multiplex.
  two-mode   papers and one other kind of node: author, keyword, journal, institution,
             country, funder or reference (the cited works; co-citation and bibliographic
             coupling are projections of this paper-reference network).
  knowledge  papers, communities, authors, keywords, journals, institutions, countries and
  graph      funders together, with typed ties (SIMILAR_TO, CITES, IN_COMMUNITY, AUTHORED_BY,
             HAS_KEYWORD, PUBLISHED_IN, AFFILIATED_WITH, LOCATED_IN, FUNDED_BY).

Communities are not in the data: S3 derives them from the one-mode similarity network.
In the knowledge graph they are nodes so that questions such as "which authors span
communities" become simple graph queries.

Every function returns (nodes, edges) DataFrames. Node ids are typed strings such as
"paper:2-s2.0-85...", "author:Lokshin B.", "community:3". Edges have source, target, type
and weight (1 unless stated).
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import networkx as nx
import numpy as np
import pandas as pd

TWO_MODE_KINDS = ("author", "keyword", "journal", "institution", "country", "funder", "reference")
KG_KINDS = ("author", "keyword", "journal", "institution", "country", "funder")
_EDGE_TYPE = {"author": "AUTHORED_BY", "keyword": "HAS_KEYWORD", "journal": "PUBLISHED_IN",
              "institution": "AFFILIATED_WITH", "country": "IN_COUNTRY", "funder": "FUNDED_BY",
              "reference": "REFERENCES"}

Tables = Tuple[pd.DataFrame, pd.DataFrame]


def _blank(v) -> bool:
    return v is None or (isinstance(v, float) and np.isnan(v)) or str(v).strip() in ("", "nan")


def paper_ids(res) -> List[str]:
    """'paper:<EID>' (or 'paper:<row>' when the EID is missing)."""
    return [f"paper:{e}" if not _blank(e) else f"paper:{i}" for i, e in zip(res.papers.index, res.papers["eid"])]


def _paper_nodes(res) -> pd.DataFrame:
    P = res.papers
    comm = P["community"]
    return pd.DataFrame({
        "id": paper_ids(res), "type": "paper", "label": P["title"].astype(str),
        "year": P["year"], "cited_by": P["cited_by"],
        "community": comm.astype("Int64"),
        "community_label": [res.labels.get(int(c), "") if pd.notna(c) else "" for c in comm],
        "doi": P["doi"].where(P["doi"].notna(), ""), "eid": P["eid"].where(P["eid"].notna(), ""),
    })


def _edges(rows, columns=("source", "target", "type", "weight")) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=list(columns))


# --------------------------------------------------------------------------
# One-mode
# --------------------------------------------------------------------------
def one_mode(res, ties: Sequence[str] = ("similar", "cites")) -> Tables:
    """Paper-paper network. ties: "similar" (S3 links at the chosen threshold, weight = S3,
    from the earlier to the later paper) and/or "cites" (citing -> cited)."""
    ids = paper_ids(res)
    years = res.papers["year"].to_numpy(dtype=float)
    rows = []
    if "similar" in ties:
        for u, v, w in res.graph().edges(data="weight"):
            a, b = (u, v) if not (years[v] < years[u]) else (v, u)  # earlier -> later
            rows.append((ids[a], ids[b], "SIMILAR_TO", round(float(w), 4)))
    if "cites" in ties:
        rows += [(ids[c], ids[d], "CITES", 1) for c, d in res.citation_pairs]
    return _paper_nodes(res), _edges(rows)


# --------------------------------------------------------------------------
# Two-mode
# --------------------------------------------------------------------------
def _memberships(res, kind: str) -> List[Tuple[int, str, Dict]]:
    """(paper row, entity name, extra node attributes) for one kind of entity."""
    from .questions import _authors, _funders, _paper_affils, _split, _split_refs
    P = res.papers
    out: List[Tuple[int, str, Dict]] = []
    if kind == "author":
        out = [(i, a, {}) for i, v in P["authors"].items() for a in dict.fromkeys(_authors(v))]
    elif kind == "keyword":
        out = [(i, k, {}) for i, v in P["keywords"].items()
               for k in dict.fromkeys(x.lower() for x in _split(v) if len(x) > 1)]
    elif kind == "journal":
        out = [(i, str(v).strip(), {}) for i, v in P["source"].items() if not _blank(v)]
    elif kind in ("institution", "country"):
        A = _paper_affils(res)
        if kind == "institution":
            A = A[A["institution"] != ""]
            out = [(r.paper, r.institution, {"country": r.country}) for r in A.itertuples()]
        else:
            A = A[A["country"] != ""].drop_duplicates(["paper", "country"])
            out = [(r.paper, r.country, {}) for r in A.itertuples()]
    elif kind == "funder":
        out = [(i, f, {}) for i, v in P["funding"].items() for f in dict.fromkeys(_funders(v))]
    elif kind == "reference":
        out = [(i, r, {}) for i, v in P["references"].items() for r in dict.fromkeys(_split_refs(v))]
    else:
        raise ValueError(f"kind must be one of {TWO_MODE_KINDS}, not {kind!r}")
    return out


def _entity_nodes(kind: str, members: List[Tuple[int, str, Dict]]) -> pd.DataFrame:
    if not members:
        return pd.DataFrame(columns=["id", "type", "label", "papers"])
    df = pd.DataFrame([{"id": f"{kind}:{name}", "label": name, **extra} for _, name, extra in members])
    papers = df.groupby("id").size().rename("papers")
    nodes = df.drop_duplicates("id").set_index("id").join(papers).reset_index()
    nodes.insert(1, "type", kind)
    return nodes


def two_mode(res, kind: str) -> Tables:
    """Paper-<kind> network (kind: author, keyword, journal, institution, country, funder,
    reference). Ties run from the paper to the entity."""
    members = _memberships(res, kind)
    ids = paper_ids(res)
    rows = [(ids[i], f"{kind}:{name}", _EDGE_TYPE[kind], 1) for i, name, _ in members]
    nodes = pd.concat([_paper_nodes(res), _entity_nodes(kind, members)], ignore_index=True)
    return nodes, _edges(rows)


# --------------------------------------------------------------------------
# Knowledge graph
# --------------------------------------------------------------------------
def knowledge_graph(res, kinds: Sequence[str] = KG_KINDS,
                    ties: Sequence[str] = ("similar", "cites"), communities: bool = True) -> Tables:
    """Papers plus several kinds of entities and tie types in one graph."""
    nodes, edges = one_mode(res, ties)
    node_parts, edge_parts = [nodes], [edges]
    ids = paper_ids(res)
    if communities:
        C = res.communities()
        comm_nodes = pd.DataFrame({
            "id": [f"community:{int(c)}" for c in C["community"]], "type": "community",
            "label": [f"C{int(c)} {res.labels.get(int(c), '')}".strip() for c in C["community"]],
            "papers": C["n_papers"].astype(int).to_numpy()})
        node_parts.append(comm_nodes)
        edge_parts.append(_edges([(ids[i], f"community:{int(c)}", "IN_COMMUNITY", 1)
                                  for i, c in res.papers["community"].items() if pd.notna(c)]))
    for kind in kinds:
        members = _memberships(res, kind)
        ent = _entity_nodes(kind, members)
        if kind == "institution" and not ent.empty:
            countries = ent.dropna(subset=["country"])
            countries = countries[countries["country"] != ""]
            edge_parts.append(_edges([(r.id, f"country:{r.country}", "LOCATED_IN", 1)
                                      for r in countries.itertuples()]))
            if "country" not in kinds:  # make sure the countries exist as nodes
                cn = countries.drop_duplicates("country")
                node_parts.append(pd.DataFrame({"id": "country:" + cn["country"], "type": "country",
                                                "label": cn["country"]}))
        if kind == "country" and "institution" in kinds:
            # papers reach countries through their institutions; keep direct ties only
            # for papers whose affiliations name no institution
            with_inst = {i for i, _, _ in _memberships(res, "institution")}
            members = [m for m in members if m[0] not in with_inst]
        node_parts.append(ent)
        edge_parts.append(_edges([(ids[i], f"{kind}:{name}", _EDGE_TYPE[kind], 1) for i, name, _ in members]))
    nodes = pd.concat([n for n in node_parts if not n.empty], ignore_index=True).drop_duplicates("id")
    edges = pd.concat([e for e in edge_parts if not e.empty], ignore_index=True).drop_duplicates()
    return nodes.reset_index(drop=True), edges.reset_index(drop=True)


# --------------------------------------------------------------------------
# Conversion and export
# --------------------------------------------------------------------------
def to_networkx(nodes: pd.DataFrame, edges: pd.DataFrame) -> nx.MultiDiGraph:
    """A directed multigraph (several tie types may join the same pair of nodes)."""
    G = nx.MultiDiGraph()
    for r in nodes.to_dict("records"):
        G.add_node(r["id"], **{k: v for k, v in r.items() if k != "id" and not _blank(v)})
    for r in edges.to_dict("records"):
        G.add_edge(r["source"], r["target"], key=r["type"], type=r["type"], weight=r["weight"])
    return G


def network(res, kind: str = "kg") -> nx.MultiDiGraph:
    """kind: "similarity", "citation", "one-mode" (both tie types), "paper-<entity>"
    (two-mode, e.g. "paper-author"), or "kg" (knowledge graph)."""
    k = kind.lower().replace("_", "-")
    if k in ("kg", "knowledge-graph"):
        t = knowledge_graph(res)
    elif k in ("similarity", "s3"):
        t = one_mode(res, ("similar",))
    elif k in ("citation", "citations", "cites"):
        t = one_mode(res, ("cites",))
    elif k in ("one-mode", "multiplex"):
        t = one_mode(res)
    elif k.startswith("paper-") and k[6:] in TWO_MODE_KINDS:
        t = two_mode(res, k[6:])
    else:
        raise ValueError(f"Unknown network {kind!r}: use similarity, citation, one-mode, "
                         f"paper-<{'|'.join(TWO_MODE_KINDS)}> or kg")
    return to_networkx(*t)


def _graphml_safe(nodes: pd.DataFrame, edges: pd.DataFrame) -> nx.MultiDiGraph:
    G = to_networkx(nodes, edges)
    for _, a in G.nodes(data=True):
        for k, v in list(a.items()):
            a[k] = int(v) if isinstance(v, (np.integer,)) else float(v) if isinstance(v, np.floating) else \
                v if isinstance(v, (int, float, str, bool)) else str(v)
    for *_, a in G.edges(data=True, keys=True):
        a["weight"] = float(a["weight"])
    return G


def export_networks(res, directory: Union[str, Path],
                    two_mode_kinds: Optional[Iterable[str]] = None,
                    formats: Sequence[str] = ("csv", "graphml")) -> Dict[str, str]:
    """Write the one-mode, two-mode and knowledge-graph networks.

    csv      <name>_nodes.csv + <name>_edges.csv (Neo4j / igraph / any tool)
    graphml  <name>.graphml (Gephi, Cytoscape, yEd, networkx)

    Two-mode kinds without data in the export are skipped. The paper-reference network is
    large (every cited work is a node: ~260,000 for 5,000 papers), so it is written only when
    asked for: two_mode_kinds=[..., "reference"]."""
    d = Path(directory).expanduser()
    d.mkdir(parents=True, exist_ok=True)
    from .questions import _has
    col = {"author": "authors", "keyword": "keywords", "journal": "source", "institution": "affiliations",
           "country": "affiliations", "funder": "funding", "reference": "references"}
    wanted = two_mode_kinds or [k for k in TWO_MODE_KINDS if k != "reference"]
    kinds = [k for k in wanted if _has(res, col[k])]
    nets = {"one_mode_similarity": one_mode(res, ("similar",))}
    if res.citation_pairs:
        nets["one_mode_citation"] = one_mode(res, ("cites",))
        nets["one_mode_multiplex"] = one_mode(res)
    for k in kinds:
        nets[f"two_mode_paper_{k}"] = two_mode(res, k)
    nets["knowledge_graph"] = knowledge_graph(res, [k for k in KG_KINDS if k in kinds])
    out: Dict[str, str] = {}
    for name, (nodes, edges) in nets.items():
        if "csv" in formats:
            nodes.to_csv(d / f"{name}_nodes.csv", index=False)
            edges.to_csv(d / f"{name}_edges.csv", index=False)
            out[f"{name}_nodes_csv"] = str(d / f"{name}_nodes.csv")
            out[f"{name}_edges_csv"] = str(d / f"{name}_edges.csv")
        if "graphml" in formats:
            nx.write_graphml(_graphml_safe(nodes, edges), d / f"{name}.graphml")
            out[f"{name}_graphml"] = str(d / f"{name}.graphml")
    return out


def summary(res) -> pd.DataFrame:
    """Node and tie counts per network, to see what each level adds."""
    rows = []
    def add(level, name, t):
        n, e = t
        rows.append({"level": level, "network": name, "node types": ", ".join(sorted(n["type"].unique())),
                     "nodes": len(n), "tie types": ", ".join(sorted(e["type"].unique())) if len(e) else "",
                     "ties": len(e)})
    add("one-mode", "similarity", one_mode(res, ("similar",)))
    if res.citation_pairs:
        add("one-mode", "citation", one_mode(res, ("cites",)))
        add("one-mode (multiplex)", "similarity + citation", one_mode(res))
    from .questions import _has
    col = {"author": "authors", "keyword": "keywords", "journal": "source", "institution": "affiliations",
           "country": "affiliations", "funder": "funding", "reference": "references"}
    kinds = [k for k in TWO_MODE_KINDS if _has(res, col[k])]
    for k in kinds:
        add("two-mode", f"paper-{k}", two_mode(res, k))
    add("knowledge graph", "all", knowledge_graph(res, [k for k in KG_KINDS if k in kinds]))
    return pd.DataFrame(rows)
