"""Data loading and graph construction utilities for ConnectMe.

Given a student dataset (CSV or Excel) with columns such as:
- id (unique student identifier) OPTIONAL; if absent an index will be used
- name (student name) OPTIONAL
- cgpa (numeric)
- hostel (categorical)
- mess (categorical)
- sports (multi-valued; separated by comma/semicolon)
- clubs (multi-valued; separated by comma/semicolon)
You can include additional multi-valued or single-valued columns; configure below.

Dynamic header inference: any CSV/Excel headers provided will be used. If no header row is detected a fallback list
is constructed from DEFAULT_CONFIG ordering. Similarity fields (multi/single/numeric) can be inferred automatically.
"""
from __future__ import annotations
import pandas as pd
import networkx as nx
from networkx.algorithms.community import greedy_modularity_communities
from typing import List, Dict, Any, Tuple
import re

# Default configuration describing how to interpret columns
DEFAULT_CONFIG = {
    "id_col": "id",          # if not present will use dataframe index
    "name_col": "name",      # optional; if absent will fall back to id
    "multi_fields": ["sports", "clubs"],  # columns containing multi-valued membership
    "single_fields": ["hostel", "mess"],   # columns whose identical value counts as 1 similarity
    "numeric_fields": {"cgpa": {"max": 10.0, "similarity": "inverse_diff"}},  # numeric similarity rule
    "min_weight": 1.0,        # minimum edge weight to include
    "intersection_weight": 1.0, # weight per shared item in multi-valued fields
    "single_match_weight": 1.0, # weight for identical single-valued field
    "numeric_similarity_weight": 1.0, # multiplier for numeric similarity contribution
}


def load_data(path: str) -> pd.DataFrame:
    """Load CSV or Excel file into DataFrame, stripping whitespace.

    Auto-detect missing header: if expected column names are absent, treat first row as data and assign
    standard column names (id,name,cgpa,hostel,mess,sports,clubs,... based on DEFAULT_CONFIG)."""
    if path.lower().endswith(".csv"):
        df = pd.read_csv(path)
    else:
        df = pd.read_excel(path)
    # Standardize column names (lowercase / strip)
    df.columns = [c.strip().lower() for c in df.columns]
    expected_names = [
        DEFAULT_CONFIG["id_col"],
        DEFAULT_CONFIG["name_col"],
        *DEFAULT_CONFIG["single_fields"],
        *list(DEFAULT_CONFIG["numeric_fields"].keys()),
        *DEFAULT_CONFIG["multi_fields"],
    ]
    if not any(c in df.columns for c in expected_names):
        # Re-read without header assumption
        if path.lower().endswith(".csv"):
            df = pd.read_csv(path, header=None)
        else:
            df = pd.read_excel(path, header=None)
        # Assign names up to available columns
        df = df.iloc[:, :len(expected_names)]
        df.columns = expected_names[:df.shape[1]]
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].fillna("").astype(str).str.strip()
    return df


def _split_multi_value(value: str) -> List[str]:
    if not value or pd.isna(value):
        return []
    # Support comma or semicolon separation
    parts = [p.strip().lower() for p in str(value).replace(";", ",").split(",") if p.strip()]
    return list(dict.fromkeys(parts))  # deduplicate preserving order


def preprocess(df: pd.DataFrame, config: Dict[str, Any] = None) -> pd.DataFrame:
    """Expand multi-valued fields into list form for later intersection computation.

    Additionally aggregates multiple single club columns (e.g., Club1, Club2, ...) into one synthetic
    multi-valued field 'clubs' if not already present, so they contribute to similarity by intersection.
    """
    if config is None:
        config = DEFAULT_CONFIG
    df = df.copy()

    # Aggregate club columns like Club1, Club2 into 'clubs'
    club_cols = [c for c in df.columns if re.match(r"club\d+", c, re.I)]
    if club_cols and 'clubs' not in df.columns:
        df['clubs'] = df[club_cols].apply(lambda row: [v.strip().lower() for v in row if isinstance(v, str) and v.strip()], axis=1)
        # Ensure uniqueness preserving order
        df['clubs'] = df['clubs'].apply(lambda lst: list(dict.fromkeys(lst)))
        if 'clubs' not in config.get('multi_fields', []):
            config['multi_fields'] = list(config.get('multi_fields', [])) + ['clubs']
        # Remove original club columns from single_fields if present
        if 'single_fields' in config:
            config['single_fields'] = [c for c in config['single_fields'] if c not in club_cols]

    for field in config.get("multi_fields", []):
        if field in df.columns:
            df[field] = df[field].apply(_split_multi_value if df[field].dtype == object else (lambda x: x)) if field != 'clubs' else df[field]
    return df


def infer_config(df: pd.DataFrame, base: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Infer similarity configuration from dataframe columns.

    Heuristics:
    - id/name columns: look for names matching /(id|student[_ ]?id)/i and /(name|student[_ ]?name)/i
    - numeric fields: columns where >80% non-empty values convert to float; keep cgpa first if present
    - multi-valued fields: object columns with any cell containing a comma/semicolon
    - single-valued fields: remaining object columns (excluding id/name) with 2 <= unique <= 0.5 * rows
    """
    if base is None:
        base = DEFAULT_CONFIG.copy()
    cols = list(df.columns)
    id_col = next((c for c in cols if re.search(r"^(id|student[_ ]?id)$", c, re.I)), None) or base.get("id_col", "id")
    name_col = next((c for c in cols if re.search(r"^(name|student[_ ]?name)$", c, re.I)), None) or base.get("name_col", "name")

    # Detect numeric
    numeric_fields = []
    for c in cols:
        series = df[c]
        non_empty = series.replace("", pd.NA).dropna()
        if not len(non_empty):
            continue
        success = 0
        for v in non_empty.head(50):
            try:
                float(str(v).strip())
                success += 1
            except Exception:
                pass
        if success / max(len(non_empty.head(50)), 1) >= 0.8:
            numeric_fields.append(c)
    # Prioritise cgpa naming
    numeric_fields = sorted(numeric_fields, key=lambda x: (0 if re.search(r"cgpa", x, re.I) else 1, x))
    numeric_fields_meta = {f: {"max": 10.0, "similarity": "inverse_diff"} for f in numeric_fields}

    # Multi-valued detection
    multi_fields = []
    for c in cols:
        if c in (id_col, name_col):
            continue
        if df[c].dtype == object:
            sample = " ".join(map(str, df[c].dropna().head(20)))
            if "," in sample or ";" in sample:
                multi_fields.append(c)
    # Single-valued categorical (exclude multi/numeric/id/name)
    single_fields = []
    for c in cols:
        if c in (id_col, name_col) or c in multi_fields or c in numeric_fields:
            continue
        if df[c].dtype == object:
            uniq = df[c].dropna().unique()
            if 1 < len(uniq) <= max(len(df) * 0.5, 2):
                single_fields.append(c)
    config = {
        "id_col": id_col,
        "name_col": name_col,
        "multi_fields": multi_fields,
        "single_fields": single_fields,
        "numeric_fields": numeric_fields_meta,
        # weights use base defaults (can be overridden externally)
        "min_weight": base.get("min_weight", 1.0),
        "intersection_weight": base.get("intersection_weight", 1.0),
        "single_match_weight": base.get("single_match_weight", 1.0),
        "numeric_similarity_weight": base.get("numeric_similarity_weight", 1.0),
    }
    return config


def compute_pair_weight(row_i: pd.Series, row_j: pd.Series, config: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
    """Compute similarity weight between two students and return details dict."""
    weight = 0.0
    detail = {"shared_multi": {}, "shared_single": [], "numeric_similarity": {}}

    # Multi-valued overlaps
    for field in config.get("multi_fields", []):
        if field in row_i and field in row_j:
            set_i = set(row_i[field]) if isinstance(row_i[field], list) else set()
            set_j = set(row_j[field]) if isinstance(row_j[field], list) else set()
            inter = sorted(set_i.intersection(set_j))
            if inter:
                w = len(inter) * config.get("intersection_weight", 1.0)
                weight += w
                detail["shared_multi"][field] = inter

    # Single-valued matches
    for field in config.get("single_fields", []):
        if field in row_i and field in row_j and row_i[field] and (row_i[field] == row_j[field]):
            weight += config.get("single_match_weight", 1.0)
            detail["shared_single"].append(field)

    # Numeric similarities
    for field, meta in config.get("numeric_fields", {}).items():
        if field in row_i and field in row_j:
            try:
                v1 = float(row_i[field])
                v2 = float(row_j[field])
            except (TypeError, ValueError):
                continue
            maxv = float(meta.get("max", 1.0))
            diff = abs(v1 - v2) / maxv
            # Inverse difference similarity (1 means identical, 0 means max apart)
            sim = 1.0 - min(diff, 1.0)
            contrib = sim * config.get("numeric_similarity_weight", 1.0)
            if contrib > 0:
                weight += contrib
                detail["numeric_similarity"][field] = sim
    return weight, detail


def build_student_graph(df: pd.DataFrame, config: Dict[str, Any] = None) -> nx.Graph:
    """Construct an undirected weighted similarity graph of students.

    If config is None a dynamic configuration is inferred from df columns."""
    if config is None:
        config = infer_config(df)
    df = preprocess(df, config)

    id_col = config.get("id_col")
    name_col = config.get("name_col")
    if id_col not in df.columns:
        df[id_col] = df.index.astype(str)
    if name_col not in df.columns:
        df[name_col] = df[id_col]

    G = nx.Graph()

    # Add nodes with attributes
    for _, row in df.iterrows():
        node_id = row[id_col]
        attrs = row.to_dict()
        # Convert list-valued fields to comma string for ease of display
        for f in config.get("multi_fields", []):
            if f in attrs and isinstance(attrs[f], list):
                attrs[f + "_list"] = attrs[f]  # keep list
                attrs[f] = ", ".join(attrs[f])
        G.add_node(node_id, label=row[name_col], **attrs)

    # Add edges based on similarity
    rows = list(df.itertuples(index=False))
    min_w = config.get("min_weight", 0.0)
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            row_i = rows[i]._asdict()
            row_j = rows[j]._asdict()
            w, detail = compute_pair_weight(row_i, row_j, config)
            if w >= min_w and w > 0:
                edge_attrs = {
                    "weight": w,
                    "shared_multi": detail["shared_multi"],
                    "shared_single": detail["shared_single"],
                    "numeric_similarity": detail["numeric_similarity"],
                }
                G.add_edge(row_i[id_col], row_j[id_col], **edge_attrs)
    # Compute graph metrics
    compute_graph_metrics(G)
    return G


def compute_graph_metrics(G: nx.Graph) -> Dict[str, Any]:
    """Compute and attach centrality and community metrics to nodes/edges."""
    if G.number_of_nodes() == 0:
        return {}
    # Node centralities
    betw = nx.betweenness_centrality(G, normalized=True, weight="weight")
    deg_cent = nx.degree_centrality(G)
    close_cent = nx.closeness_centrality(G, distance=lambda u, v, d: 1 / d.get("weight", 1)) if G.number_of_edges() else {}
    # Edge betweenness
    edge_betw = nx.edge_betweenness_centrality(G, normalized=True, weight="weight")

    # Communities (greedy modularity)
    try:
        communities = list(greedy_modularity_communities(G, weight="weight"))
    except Exception:
        communities = []
    community_map = {}
    for idx, comm in enumerate(communities):
        for node in comm:
            community_map[node] = idx

    for n in G.nodes():
        G.nodes[n]["betweenness"] = betw.get(n, 0.0)
        G.nodes[n]["degree_centrality"] = deg_cent.get(n, 0.0)
        G.nodes[n]["closeness_centrality"] = close_cent.get(n, 0.0)
        G.nodes[n]["community"] = community_map.get(n, -1)
    for (u, v), val in edge_betw.items():
        G[u][v]["edge_betweenness"] = val

    summary = {
        "num_nodes": G.number_of_nodes(),
        "num_edges": G.number_of_edges(),
        "communities": len(communities),
        "avg_degree": sum(dict(G.degree()).values()) / max(G.number_of_nodes(), 1),
    }
    return summary


def graph_to_edge_dataframe(G: nx.Graph) -> pd.DataFrame:
    data = []
    for u, v, d in G.edges(data=True):
        data.append({
            "source": u,
            "target": v,
            "weight": d.get("weight"),
            "edge_betweenness": d.get("edge_betweenness"),
            "shared_multi": d.get("shared_multi"),
            "shared_single": d.get("shared_single"),
        })
    return pd.DataFrame(data)


def graph_to_node_dataframe(G: nx.Graph) -> pd.DataFrame:
    data = []
    for n, d in G.nodes(data=True):
        data.append({
            "id": n,
            "label": d.get("label", n),
            "betweenness": d.get("betweenness"),
            "degree_centrality": d.get("degree_centrality"),
            "closeness_centrality": d.get("closeness_centrality"),
            "community": d.get("community"),
        })
    return pd.DataFrame(data)


def build_student_table(G: nx.Graph, df: pd.DataFrame, config: Dict[str, Any] = None) -> pd.DataFrame:
    """Build a rich student table with all attributes, metrics, and connection counts.
    
    Returns a DataFrame with columns: id, name, all attributes from df, degree, betweenness, community.
    """
    if config is None:
        config = DEFAULT_CONFIG
    
    id_col = config.get("id_col", "id")
    name_col = config.get("name_col", "name")
    
    data = []
    for n, node_data in G.nodes(data=True):
        row = {
            "id": n,
            "name": node_data.get("label", n),
            "num_connections": G.degree(n),
            "betweenness": round(node_data.get("betweenness", 0.0), 4),
            "degree_centrality": round(node_data.get("degree_centrality", 0.0), 4),
            "closeness_centrality": round(node_data.get("closeness_centrality", 0.0), 4),
            "community": node_data.get("community", -1),
        }
        # Add original attributes from graph nodes (exclude label/list versions)
        for k, v in node_data.items():
            if k not in ("label",) and not k.endswith("_list"):
                row[k] = v
        data.append(row)
    
    result_df = pd.DataFrame(data)
    # Reorder columns for clarity
    cols = ["id", "name", "num_connections", "betweenness", "community"]
    extra_cols = [c for c in result_df.columns if c not in cols]
    return result_df[cols + extra_cols]

__all__ = [
    "load_data",
    "build_student_graph",
    "graph_to_edge_dataframe",
    "graph_to_node_dataframe",
    "build_student_table",
    "DEFAULT_CONFIG",
    "infer_config",
]
