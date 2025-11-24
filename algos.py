import networkx as nx
from typing import Dict, Any, List, Tuple, Set

def compute_communities(G: nx.Graph) -> Dict[str, int]:
    """Assign community ids via greedy modularity."""
    from networkx.algorithms.community import greedy_modularity_communities
    comms = list(greedy_modularity_communities(G, weight="weight"))
    mapping = {}
    for i, cset in enumerate(comms):
        for n in cset:
            mapping[n] = i
    return mapping

def betweenness_centrality(G: nx.Graph) -> Dict[str, float]:
    return nx.betweenness_centrality(G, weight="weight", normalized=True)

def list_maximal_cliques(G: nx.Graph, min_size: int = 3, limit: int = 50) -> List[List[str]]:
    """Return up to limit maximal cliques >= min_size."""
    cliques = []
    for c in nx.find_cliques(G):
        if len(c) >= min_size:
            cliques.append(c)
            if len(cliques) >= limit:
                break
    return cliques

def max_weight_mentor_matching(
    G: nx.Graph,
    mentor_min_cgpa: float = 8.5,
    mentee_max_cgpa: float = 8.0
) -> List[Tuple[str, str, float]]:
    """Produce mentor-mentee pairs using max weight matching across two partitions:
    mentors: cgpa >= mentor_min_cgpa
    mentees: cgpa <= mentee_max_cgpa
    Uses edge weight as compatibility; only edges crossing partitions considered.
    """
    mentors = {n for n, d in G.nodes(data=True) if _cgpa(d) >= mentor_min_cgpa}
    mentees = {n for n, d in G.nodes(data=True) if _cgpa(d) <= mentee_max_cgpa and n not in mentors}
    # Build bipartite weighted subgraph
    B = nx.Graph()
    for u in mentors:
        for v in mentees:
            if G.has_edge(u, v):
                w = float(G[u][v].get("weight", 1.0))
                if w > 0:
                    B.add_edge(u, v, weight=w)
    if B.number_of_edges() == 0:
        return []
    matching = nx.algorithms.matching.max_weight_matching(B, maxcardinality=False, weight="weight")
    pairs = []
    for u, v in matching:
        w = float(B[u][v]["weight"])
        if u in mentors:
            pairs.append((u, v, w))
        else:
            pairs.append((v, u, w))
    # Sort by descending weight
    return sorted(pairs, key=lambda x: x[2], reverse=True)

def professor_candidate_filter(
    G: nx.Graph,
    min_cgpa: float = 8.0,
    required_subjects: List[str] | None = None
) -> List[str]:
    """Filter students satisfying min_cgpa and possessing all required subjects (multi-valued 'subjects')."""
    req = {s.strip().lower() for s in (required_subjects or []) if s.strip()}
    out = []
    for n, d in G.nodes(data=True):
        if _cgpa(d) < min_cgpa:
            continue
        if req:
            subjects_raw = d.get("subjects", [])
            if isinstance(subjects_raw, str):
                subjects = {s.strip().lower() for s in subjects_raw.split(",") if s.strip()}
            elif isinstance(subjects_raw, list):
                subjects = {str(s).strip().lower() for s in subjects_raw if str(s).strip()}
            else:
                subjects = set()
            if not req.issubset(subjects):
                continue
        out.append(n)
    return out

def _cgpa(data: Dict[str, Any]) -> float:
    try:
        return float(data.get("cgpa", 0))
    except Exception:
        return 0.0

def aggregate_metrics(G: nx.Graph) -> Dict[str, Any]:
    """Compute bundle of metrics for optional display."""
    metrics: Dict[str, Any] = {}
    metrics["degree"] = dict(G.degree())
    metrics["weighted_degree"] = {n: sum(d.get("weight", 1.0) for _, _, d in G.edges(n, data=True)) for n in G.nodes()}
    metrics["betweenness"] = betweenness_centrality(G)
    metrics["clustering"] = nx.clustering(G, weight="weight")
    return metrics