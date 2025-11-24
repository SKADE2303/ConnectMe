import json
import pathlib
import pandas as pd
import networkx as nx
from typing import Dict, Any
from data import (
    load_data,
    build_student_graph,
    graph_to_node_dataframe,
    graph_to_edge_dataframe,
    build_student_table,
    extract_ego_graph,
    DEFAULT_CONFIG,
    compute_graph_metrics,
    infer_config,
)
from algos import (
    compute_communities,
    betweenness_centrality,
    list_maximal_cliques,
    max_weight_mentor_matching,
    professor_candidate_filter,
    aggregate_metrics
)

try:
    import plotly.express as px
    import plotly.graph_objects as go
except ImportError:  # lightweight fallback
    px = None
    go = None

try:
    import streamlit as st
except ImportError:
    st = None

try:
    from streamlit_plotly_events import plotly_events
except ImportError:
    plotly_events = None

APP_INFO = None  # previously displayed introduction removed

# ------------------- Core helpers -------------------

HOSTEL_COLOR_MAP = {
    'h1': '#FFD700',  # gold
    'h2': '#8A2BE2',  # purple
    'h3': '#FF8C00',  # dark orange
    'h4': '#00CED1',  # turquoise
}
MESS_COLOR_MAP = {
    'm1': '#1f77b4',
    'm2': '#2ca02c',
    'm3': '#d62728',
    'm4': '#9467bd',
}
CGPA_COLOR_MAP = {
    'high': '#008000',   # >9 green
    'low': '#FF0000',    # <6 red
    'mid': '#FFD700',    # between 6 and 9 yellow/gold
}

def build_and_summarise(path: str, config: Dict[str, Any] = None):
    df = load_data(path)
    if config is None:
        # infer dynamically
        config = infer_config(df)
    G = build_student_graph(df, config)
    nodes_df = graph_to_node_dataframe(G)
    edges_df = graph_to_edge_dataframe(G)
    summary = {
        "nodes": len(nodes_df),
        "edges": len(edges_df),
        "communities": nodes_df["community"].nunique() if len(nodes_df) else 0,
    }
    return G, nodes_df, edges_df, summary


def plot_interactive_network(
    G: nx.Graph,
    color_mode: str = 'Community',
    selected_node: str | None = None,
    highlight_neighbors: bool = True,
    show_edge_labels: bool = True,
    edge_mode: str = 'All Similarities',
    group_highlight_field: str | None = None,
    group_highlight_values: list[str] | None = None,
    fig_height: int = 900,
    layout_spread: float = 1.8
):
    """Return a Plotly Figure.

    group_highlight_field/group_highlight_values:
      Highlight (glow) nodes and edges where both endpoints share at least one chosen value
      in the selected multi-valued field (e.g. clubs='literature').
    """
    if go is None:
        raise RuntimeError("Plotly not installed. Please pip install plotly.")
    if G.number_of_nodes() == 0:
        return go.Figure()

    pos = nx.spring_layout(G, seed=None, k=layout_spread)

    def cgpa_band(val) -> str:
        try:
            v = float(val)
            if v > 9: return 'high'
            if v < 6: return 'low'
            return 'mid'
        except Exception:
            return 'mid'

    edge_items = []
    if edge_mode == 'All Similarities':
        def edge_style(weight: float) -> tuple[str, float]:
            if weight >= 6: return '#ff0000', min(10, 3 + weight * 0.9)
            if weight >= 4: return '#ff7f0e', 2.5 + weight * 0.6
            if weight >= 2: return '#ffd700', 2 + weight * 0.4
            return '#1f77b4', 1 + weight * 0.3
        for u, v, d in G.edges(data=True):
            w = float(d.get('weight', 0.0))
            c, width = edge_style(w)
            edge_items.append((u, v, w, c, width, f"{w:.2f}"))
    else:
        if edge_mode == 'Same Club':
            palette = (px.colors.qualitative.Plotly if px else
                       ['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd',
                        '#8c564b','#e377c2','#7f7f7f','#bcbd22','#17becf'])
            club_color = {}
            club_index = 0
        if edge_mode == 'Same Hostel':
            groups = {}
            for n, d in G.nodes(data=True):
                host = str(d.get('hostel','')).strip().lower()
                if host: groups.setdefault(host, []).append(n)
            for host, nodes in groups.items():
                color = HOSTEL_COLOR_MAP.get(host, '#888888')
                for i in range(len(nodes)):
                    for j in range(i+1, len(nodes)):
                        edge_items.append((nodes[i], nodes[j], 1.0, color, 3.0, host.upper()))
        elif edge_mode == 'Same CGPA Band':
            bands = {}
            for n, d in G.nodes(data=True):
                band = cgpa_band(d.get('cgpa'))
                bands.setdefault(band, []).append(n)
            for band, nodes in bands.items():
                color = CGPA_COLOR_MAP.get(band, '#888888')
                for i in range(len(nodes)):
                    for j in range(i+1, len(nodes)):
                        edge_items.append((nodes[i], nodes[j], 1.0, color, 3.0, band))
        elif edge_mode == 'Same Club':
            club_groups = {}
            for n, d in G.nodes(data=True):
                clubs = d.get('clubs') or []
                if isinstance(clubs, str):
                    clubs = [c.strip().lower() for c in clubs.split(',') if c.strip()]
                for club in clubs:
                    ckey = club.strip().lower()
                    if ckey:
                        club_groups.setdefault(ckey, []).append(n)
            for club, nodes in club_groups.items():
                if club not in club_color:
                    club_color[club] = palette[club_index % len(palette)]
                    club_index += 1
                color = club_color[club]
                for i in range(len(nodes)):
                    for j in range(i+1, len(nodes)):
                        edge_items.append((nodes[i], nodes[j], 1.0, color, 2.5, club))

    # Group highlight sets
    group_highlight_values = [v.lower().strip() for v in (group_highlight_values or []) if v.strip()]
    highlight_group_nodes = set()
    if group_highlight_field and group_highlight_values:
        for n, d in G.nodes(data=True):
            raw = d.get(group_highlight_field, [])
            if isinstance(raw, str):
                items = [i.strip().lower() for i in raw.split(',') if i.strip()]
            elif isinstance(raw, list):
                items = [str(i).strip().lower() for i in raw if str(i).strip()]
            else:
                items = []
            if set(items) & set(group_highlight_values):
                highlight_group_nodes.add(n)

    highlighted_edges = set()
    neighbor_set = set()
    if selected_node and selected_node in G:
        if edge_mode == 'All Similarities':
            neighbor_set = set(G.neighbors(selected_node))
            for nbr in neighbor_set:
                highlighted_edges.add(tuple(sorted((selected_node, nbr))))
        else:
            for u, v, *_ in edge_items:
                if u == selected_node or v == selected_node:
                    neighbor_set.add(v if u == selected_node else u)
                    highlighted_edges.add(tuple(sorted((u, v))))

    # Add group highlight edges
    if group_highlight_field and group_highlight_values:
        for u, v, *rest in edge_items:
            if u in highlight_group_nodes and v in highlight_group_nodes:
                highlighted_edges.add(tuple(sorted((u, v))))

    # Build traces
    edge_traces = []
    label_x = []; label_y = []; label_text = []
    grouped = {}
    for u, v, w, c, width, lbl in edge_items:
        key = tuple(sorted((u, v)))
        is_selected_highlight = key in highlighted_edges
        if is_selected_highlight:
            c_draw = '#00ffff'
            width_draw = width + 3
        else:
            c_draw = c
            width_draw = width
        grouped.setdefault((c_draw, width_draw), []).append((u, v, lbl, is_selected_highlight))
    for (c_draw, width_draw), lst in grouped.items():
        xs = []; ys = []
        for (u, v, lbl, is_h) in lst:
            x0, y0 = pos[u]; x1, y1 = pos[v]
            xs += [x0, x1, None]; ys += [y0, y1, None]
            if show_edge_labels and is_h:
                label_x.append((x0 + x1)/2); label_y.append((y0 + y1)/2); label_text.append(lbl)
        edge_traces.append(go.Scatter(
            x=xs, y=ys, mode='lines',
            line=dict(color=c_draw, width=width_draw),
            hoverinfo='none', showlegend=False
        ))

    # Node colors
    node_colors = []
    for n, data in G.nodes(data=True):
        if color_mode == 'Hostel':
            node_colors.append(HOSTEL_COLOR_MAP.get(str(data.get('hostel','')).lower(), '#cccccc'))
        elif color_mode == 'Mess':
            node_colors.append(MESS_COLOR_MAP.get(str(data.get('mess','')).lower(), '#cccccc'))
        elif color_mode == 'CGPA':
            node_colors.append(CGPA_COLOR_MAP.get(cgpa_band(data.get('cgpa')), '#cccccc'))
        elif color_mode == 'Community':
            node_colors.append(data.get('community', -1))
        else:
            node_colors.append('#888888')

    node_x = []; node_y = []; hover_text = []; marker_vals = []
    node_ids = []; node_sizes = []; node_line_w = []; node_line_c = []
    for idx, (n, data) in enumerate(G.nodes(data=True)):
        x, y = pos[n]
        node_x.append(x); node_y.append(y); node_ids.append(n)
        sports_val = data.get('sports', '')
        sports_str = ", ".join(sports_val) if isinstance(sports_val, list) else str(sports_val)
        clubs_val = data.get('clubs', '')
        clubs_str = ", ".join(clubs_val) if isinstance(clubs_val, list) else str(clubs_val)
        hover_text.append(
            f"{data.get('label', n)}"
            f"<br>cgpa={data.get('cgpa','NA')}"
            f"<br>hostel={data.get('hostel','')}"
            f"<br>mess={data.get('mess','')}"
            f"<br>sport={sports_str}"
            f"<br>clubs={clubs_str}"
            f"<br>community={data.get('community')}"
        )
        marker_vals.append(node_colors[idx])  # FIX: previously empty -> black nodes

        # Node highlight precedence: selected node > group highlight > neighbor
        if selected_node and n == selected_node:
            node_sizes.append(36); node_line_w.append(6); node_line_c.append('#00ffff')
        elif n in highlight_group_nodes:
            node_sizes.append(30); node_line_w.append(4); node_line_c.append('#ffa500')
        elif selected_node and n in neighbor_set:
            node_sizes.append(26); node_line_w.append(3); node_line_c.append('#ff00ff')
        else:
            node_sizes.append(20); node_line_w.append(1.5); node_line_c.append('#222222')

    node_trace = go.Scatter(
        x=node_x, y=node_y, mode="markers",
        hoverinfo="text", text=hover_text, customdata=node_ids,
        marker=dict(
            showscale=(color_mode == 'Community'),
            colorscale="Viridis" if color_mode == 'Community' else None,
            color=marker_vals, size=node_sizes,
            colorbar=dict(title="Community") if color_mode == 'Community' else None,
            line=dict(width=node_line_w, color=node_line_c),
        ),
    )
    label_trace = go.Scatter(
        x=label_x, y=label_y, mode='text', text=label_text,
        textposition='middle center', hoverinfo='none', showlegend=False
    )
    title_suffix = "" if edge_mode == 'All Similarities' else f" ({edge_mode})"
    if group_highlight_field and group_highlight_values:
        title_suffix += f" | Highlight: {group_highlight_field}={', '.join(group_highlight_values)}"
    fig = go.Figure(
        data=edge_traces + [node_trace, label_trace],
        layout=go.Layout(
            title=f"Student Similarity Network{title_suffix}",
            showlegend=False, hovermode="closest",
            margin=dict(b=10, l=10, r=10, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=fig_height
        )
    )
    return fig


def plot_ego_network(ego_graph: nx.Graph, center_node_id: str, color_mode: str = 'Community'):
    """Plot ego graph (center node + neighbors) with edge labels showing similarity."""
    if go is None or ego_graph.number_of_nodes() == 0:
        return None
    
    # Use circular layout centered at origin, neighbors around the perimeter
    pos = nx.spring_layout(ego_graph, k=2, iterations=50, seed=42)
    
    # Build edge traces with labels
    edge_traces = []
    edge_labels_x = []
    edge_labels_y = []
    edge_labels_text = []
    
    for u, v, d in ego_graph.edges(data=True):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        weight = d.get('weight', 0)
        shared_multi = d.get('shared_multi', {})
        
        # Create hover text for edge
        hover_text = f"Weight: {weight:.2f}"
        if shared_multi:
            for field, items in shared_multi.items():
                hover_text += f"\n{field}: {', '.join(items)}"
        
        edge_traces.append(go.Scatter(
            x=[x0, x1, None],
            y=[y0, y1, None],
            mode='lines',
            hoverinfo='text',
            text=hover_text,
            line=dict(width=2, color='#888'),
            showlegend=False
        ))
        
        # Add edge label at midpoint
        mid_x, mid_y = (x0 + x1) / 2, (y0 + y1) / 2
        edge_labels_x.append(mid_x)
        edge_labels_y.append(mid_y)
        edge_labels_text.append(f"{weight:.1f}")
    
    # Build node trace
    node_x = []
    node_y = []
    node_text = []
    node_size = []
    node_color = []
    
    for n, d in ego_graph.nodes(data=True):
        x, y = pos[n]
        node_x.append(x)
        node_y.append(y)
        
        # Center node is larger and red
        if n == center_node_id:
            node_size.append(40)
            node_color.append('#ff0000')
        else:
            node_size.append(20)
            node_color.append('#1f77b4' if color_mode == 'Community' else '#888')
        
        label = d.get('label', n)
        cgpa = d.get('cgpa', 'N/A')
        node_text.append(f"{label}<br>CGPA: {cgpa}")
    
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        text=[n for n in ego_graph.nodes()],
        textposition='top center',
        hoverinfo='text',
        hovertext=node_text,
        marker=dict(size=node_size, color=node_color, line=dict(width=2, color='white')),
        showlegend=False
    )
    
    # Edge labels trace
    edge_label_trace = go.Scatter(
        x=edge_labels_x, y=edge_labels_y,
        text=edge_labels_text,
        mode='text',
        hoverinfo='none',
        showlegend=False
    )
    
    fig = go.Figure(data=edge_traces + [node_trace, edge_label_trace], layout=go.Layout(
        title=f"Ego Network: {ego_graph.nodes[center_node_id].get('label', center_node_id)}",
        showlegend=False,
        hovermode='closest',
        margin=dict(b=20, l=20, r=20, t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor='#111111',
        paper_bgcolor='#111111',
        font=dict(color='white')
    ))
    return fig


# ------------------- Streamlit UI -------------------

def run_streamlit_app():
    st.set_page_config(page_title="ConnectMe Graph", layout="wide")
    st.title("ConnectMe: Student Similarity Graph")
    uploaded = st.file_uploader("Upload CSV / Excel", type=["csv","xlsx","xls"])
    if not uploaded:
        st.info("Upload a data file to begin.")
        return
    suffix = pathlib.Path(uploaded.name).suffix.lower()
    temp_path = pathlib.Path("_uploaded_tmp" + suffix)
    temp_path.write_bytes(uploaded.getbuffer())

    try:
        base_df = load_data(str(temp_path))
    except Exception as e:
        st.error(f"Failed to parse file: {e}")
        return
    inferred_cfg = infer_config(base_df, DEFAULT_CONFIG)

    st.sidebar.header("Weights")
    w_min = st.sidebar.number_input("Min edge weight", 0.0, 100.0, float(DEFAULT_CONFIG["min_weight"]), 0.1)
    w_inter = st.sidebar.number_input("Weight per shared multi item", 0.0, 10.0, float(DEFAULT_CONFIG["intersection_weight"]), 0.1)
    w_single = st.sidebar.number_input("Weight per identical single field", 0.0, 10.0, float(DEFAULT_CONFIG["single_match_weight"]), 0.1)
    w_numeric = st.sidebar.number_input("Numeric similarity multiplier", 0.0, 10.0, float(DEFAULT_CONFIG["numeric_similarity_weight"]), 0.1)

    st.sidebar.header("Fields")
    multi_sel = st.sidebar.multiselect("Multi-valued", inferred_cfg.get("multi_fields", []), default=inferred_cfg.get("multi_fields", []))
    single_sel = st.sidebar.multiselect("Single-valued", inferred_cfg.get("single_fields", []), default=inferred_cfg.get("single_fields", []))
    numeric_all = list(inferred_cfg.get("numeric_fields", {}).keys())
    numeric_sel = st.sidebar.multiselect("Numeric", numeric_all, default=numeric_all)

    st.sidebar.header("Layout")
    fig_height = st.sidebar.number_input("Graph height (px)", min_value=500, max_value=1600, value=900, step=50)
    layout_spread = st.sidebar.slider("Layout spread", min_value=0.5, max_value=3.0, value=1.8, step=0.1)

    color_mode = st.sidebar.selectbox("Color nodes by", ["Community", "Hostel", "CGPA", "Mess"])
    edge_mode = st.sidebar.selectbox("Show edges for", ["All Similarities", "Same Club", "Same Hostel", "Same CGPA Band"])

    config = inferred_cfg.copy()
    config.update({
        "min_weight": w_min,
        "intersection_weight": w_inter,
        "single_match_weight": w_single,
        "numeric_similarity_weight": w_numeric,
        "multi_fields": multi_sel,
        "single_fields": single_sel,
        "numeric_fields": {k: inferred_cfg["numeric_fields"][k] for k in numeric_sel},
    })
    G, _, _, _ = build_and_summarise(str(temp_path), config)
    if G.number_of_nodes() == 0:
        st.error("No nodes generated. Check input file format.")
        return

    # Compute & assign communities if missing
    if any("community" not in d for _, d in G.nodes(data=True)):
        comm_map = compute_communities(G)
        for n, cid in comm_map.items():
            G.nodes[n]["community"] = cid

    # Search & selection
    selected_node = None
    search_query = st.sidebar.text_input("Search name/id")
    if search_query:
        matches = []
        for n, data in G.nodes(data=True):
            label = str(data.get('label', n))
            if search_query.lower() in label.lower() or search_query.lower() in str(n).lower():
                matches.append((n, label))
        if matches:
            disp = [f"{lab} ({nid})" for nid, lab in matches]
            sel = st.sidebar.selectbox("Matches", disp)
            if sel:
                idx = disp.index(sel)
                selected_node = matches[idx][0]
        else:
            st.sidebar.info("No matches")

    # Group highlight controls (club / sport / subjects etc.)
    st.sidebar.header("Group Highlight")
    highlight_source_options = ["(none)"] + multi_sel
    gh_field = st.sidebar.selectbox("Highlight multi-field", highlight_source_options)
    gh_values = []
    if gh_field != "(none)":
        values = set()
        for _, d in G.nodes(data=True):
            raw = d.get(gh_field, [])
            if isinstance(raw, str):
                items = [i.strip().lower() for i in raw.split(',') if i.strip()]
            elif isinstance(raw, list):
                items = [str(i).strip().lower() for i in raw if str(i).strip()]
            else:
                items = []
            values.update(items)
        gh_values = st.sidebar.multiselect("Values", sorted(values))

    # Mentor/Mentee suggestions
    st.sidebar.header("Mentor/Mentee")
    mentor_min = st.sidebar.number_input("Mentor min CGPA", 0.0, 10.0, 8.5, 0.1)
    mentee_max = st.sidebar.number_input("Mentee max CGPA", 0.0, 10.0, 8.0, 0.1)
    show_pairs = st.sidebar.checkbox("Show mentor-mentee pairs")

    pairs = []
    if show_pairs:
        pairs = max_weight_mentor_matching(G, mentor_min, mentee_max)
        st.sidebar.write(f"Pairs: {len(pairs)}")

    # Professor selection
    st.sidebar.header("Professor Filter")
    prof_min_cgpa = st.sidebar.number_input("Min CGPA", 0.0, 10.0, 8.0, 0.1)
    subjects_field = "subjects" if "subjects" in G.nodes[next(iter(G.nodes()))] else None
    prof_required = []
    if subjects_field:
        subj_values = set()
        for _, d in G.nodes(data=True):
            raw = d.get("subjects", [])
            if isinstance(raw, str):
                items = [i.strip().lower() for i in raw.split(",") if i.strip()]
            elif isinstance(raw, list):
                items = [str(i).strip().lower() for i in raw if str(i).strip()]
            else:
                items = []
            subj_values.update(items)
        prof_required = st.sidebar.multiselect("Required subjects", sorted(subj_values))
    show_prof = st.sidebar.checkbox("Show professor candidate list")

    prof_candidates = []
    if show_prof:
        prof_candidates = professor_candidate_filter(G, prof_min_cgpa, prof_required)

    # Algorithm summaries (simple toggle)
    st.sidebar.header("Algorithms")
    show_algo_metrics = st.sidebar.checkbox("Show centrality/clique info")

    fig = plot_interactive_network(
        G,
        color_mode=color_mode,
        selected_node=selected_node,
        edge_mode=edge_mode,
        group_highlight_field=(None if gh_field == "(none)" else gh_field),
        group_highlight_values=gh_values,
        fig_height=fig_height,
        layout_spread=layout_spread
    )
    st.plotly_chart(fig, use_container_width=True)

    # Details panel
    if plotly_events:
        selected = plotly_events(fig, click_event=True, hover_event=False)
        if selected:
            node_id = selected[0].get('customdata')
            if node_id in G.nodes:
                d = G.nodes[node_id]
                sports = d.get('sports', [])
                clubs = d.get('clubs', [])
                if isinstance(sports, list): sports = ", ".join(sports)
                if isinstance(clubs, list): clubs = ", ".join(clubs)
                st.subheader(f"Student: {d.get('label', node_id)}")
                st.markdown(f"**CGPA:** {d.get('cgpa','NA')}  \n**Hostel:** {d.get('hostel','')}  \n**Mess:** {d.get('mess','')}  \n**Sport:** {sports}  \n**Clubs:** {clubs}  \n**Community:** {d.get('community')}")
    # Mentor/Mentee table
    if pairs:
        st.subheader("Mentor-Mentee Suggestions")
        rows = []
        for mentor, mentee, w in pairs:
            md = G.nodes[mentor]; sd = G.nodes[mentee]
            rows.append({
                "Mentor": md.get("label", mentor),
                "Mentor CGPA": md.get("cgpa"),
                "Mentee": sd.get("label", mentee),
                "Mentee CGPA": sd.get("cgpa"),
                "Compatibility Weight": round(w, 2)
            })
        st.dataframe(rows, use_container_width=True)

    # Professor candidate list
    if prof_candidates:
        st.subheader("Professor Candidate List")
        rows = []
        for n in prof_candidates:
            d = G.nodes[n]
            rows.append({
                "ID": n,
                "Name": d.get("label", n),
                "CGPA": d.get("cgpa"),
                "Subjects": d.get("subjects")
            })
        st.dataframe(rows, use_container_width=True)

    # Algorithm metrics
    if show_algo_metrics:
        metrics = aggregate_metrics(G)
        st.subheader("Centrality & Clique Data")
        top_bt = sorted(metrics["betweenness"].items(), key=lambda x: x[1], reverse=True)[:10]
        cliques = list_maximal_cliques(G, min_size=3, limit=10)
        st.markdown("**Top Betweenness (bridges):**")
        st.write([{ "Node": G.nodes[n].get("label", n), "Betweenness": round(v,4)} for n,v in top_bt])
        st.markdown("**Sample Maximal Cliques (size ≥3):**")
        st.write([{"Size": len(c), "Members": [G.nodes[n].get("label", n) for n in c]} for c in cliques])

    # Legend
    if color_mode == 'Hostel':
        st.caption("Hostel legend:")
        for k, v in HOSTEL_COLOR_MAP.items():
            st.markdown(f"<span style='display:inline-block;width:14px;height:14px;background:{v};border-radius:2px;margin-right:6px;'></span>{k.upper()}", unsafe_allow_html=True)
    elif color_mode == 'Mess':
        st.caption("Mess legend:")
        for k, v in MESS_COLOR_MAP.items():
            st.markdown(f"<span style='display:inline-block;width:14px;height:14px;background:{v};border-radius:2px;margin-right:6px;'></span>{k.upper()}", unsafe_allow_html=True)
    elif color_mode == 'CGPA':
        st.caption("CGPA bands:")
        for key,label in [('high','>9'), ('mid','6-9'), ('low','<6')]:
            v = CGPA_COLOR_MAP[key]
            st.markdown(f"<span style='display:inline-block;width:14px;height:14px;background:{v};border-radius:2px;margin-right:6px;'></span>{label}", unsafe_allow_html=True)
    else:
        st.caption("Community colorscale applied.")

    # Click details
    if plotly_events:
        selected = plotly_events(fig, click_event=True, hover_event=False)
        if selected:
            node_id = selected[0].get('customdata')
            if node_id in G.nodes:
                node_data = G.nodes[node_id]
                # Prepare clubs/sports display
                sports_val = node_data.get('sports', '')
                if isinstance(sports_val, list):
                    sports_str = ", ".join(sports_val)
                else:
                    sports_str = str(sports_val)
                clubs_val = node_data.get('clubs', '')
                if isinstance(clubs_val, list):
                    clubs_str = ", ".join(clubs_val)
                else:
                    clubs_str = str(clubs_val)
                st.subheader(f"Details: {node_data.get('label', node_id)}")
                st.markdown(f"""
**ID:** {node_id}  
**CGPA:** {node_data.get('cgpa','NA')}  
**Hostel:** {node_data.get('hostel','')}  
**Mess:** {node_data.get('mess','')}  
**Sport:** {sports_str}  
**Clubs:** {clubs_str}  
**Community:** {node_data.get('community')}
""")
                # Raw JSON (optional)
                with st.expander("Raw node attributes"):
                    st.json(node_data)
    else:
        st.caption("Install streamlit-plotly-events for clickable node details.")

    if selected_node and selected_node in G.nodes:
        st.sidebar.markdown("---")
        st.sidebar.subheader("Selected student")
        st.sidebar.write(G.nodes[selected_node].get('label', selected_node))
        attrs = {k: v for k, v in G.nodes[selected_node].items() if k not in ('label',)}
        st.sidebar.json(attrs)


# ------------------- CLI -------------------

def run_cli():
    import argparse
    parser = argparse.ArgumentParser(description="Build and visualise student similarity graph")
    parser.add_argument("--input", required=True, help="Input CSV/Excel path")
    parser.add_argument("--output-prefix", default="graph_output", help="Prefix for output files")
    args = parser.parse_args()

    G, nodes_df, edges_df, summary = build_and_summarise(args.input, DEFAULT_CONFIG)
    print("Summary:", summary)
    nodes_df.to_csv(f"{args.output_prefix}_nodes.csv", index=False)
    edges_df.to_csv(f"{args.output_prefix}_edges.csv", index=False)
    nx.write_gml(G, f"{args.output_prefix}.gml")
    if go:
        fig = plot_interactive_network(G)
        fig.write_html(f"{args.output_prefix}.html")
        print("Interactive HTML written (no click filtering in CLI).")
    else:
        print("Plotly not installed; skipping HTML export.")


if __name__ == "__main__":
    # Always run Streamlit app if streamlit is available
    if st is not None:
        run_streamlit_app()
    else:
        run_cli()
