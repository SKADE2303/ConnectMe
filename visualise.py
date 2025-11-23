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
    DEFAULT_CONFIG,
    compute_graph_metrics,
    infer_config,
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


def plot_interactive_network(G: nx.Graph, color_mode: str = 'Community'):
    if go is None:
        raise RuntimeError("Plotly not installed. Please pip install plotly.")
    # Build edge traces with optional color per edge
    edge_traces = []
    # Prepare node color list based on selected mode
    node_colors = []
    if color_mode == 'Hostel':
        for n, data in G.nodes(data=True):
            host = str(data.get('hostel', '')).lower()
            node_colors.append(HOSTEL_COLOR_MAP.get(host, '#CCCCCC'))
    elif color_mode == 'CGPA':
        for n, data in G.nodes(data=True):
            try:
                cg = float(data.get('cgpa', 0))
            except (TypeError, ValueError):
                cg = 0
            if cg > 9:
                node_colors.append(CGPA_COLOR_MAP['high'])
            elif cg < 6:
                node_colors.append(CGPA_COLOR_MAP['low'])
            else:
                node_colors.append(CGPA_COLOR_MAP['mid'])
    elif color_mode == 'Mess':
        for n, data in G.nodes(data=True):
            mess = str(data.get('mess', '')).lower()
            node_colors.append(MESS_COLOR_MAP.get(mess, '#CCCCCC'))
    elif color_mode == 'Community':
        # will use community index with colorscale
        pass
    else:
        for _ in G.nodes():
            node_colors.append('#CCCCCC')

    # Decide edge colors (match node attribute if endpoints share category)
    def edge_color(u, v, du, dv):
        if color_mode == 'Hostel':
            hu = str(du.get('hostel', '')).lower()
            hv = str(dv.get('hostel', '')).lower()
            return HOSTEL_COLOR_MAP.get(hu, '#888') if hu == hv and hu else '#BBBBBB'
        if color_mode == 'CGPA':
            try:
                cu = float(du.get('cgpa', 0))
                cv = float(dv.get('cgpa', 0))
            except (TypeError, ValueError):
                return '#BBBBBB'
            bucket_u = 'high' if cu > 9 else ('low' if cu < 6 else 'mid')
            bucket_v = 'high' if cv > 9 else ('low' if cv < 6 else 'mid')
            return CGPA_COLOR_MAP[bucket_u] if bucket_u == bucket_v else '#BBBBBB'
        if color_mode == 'Mess':
            mu = str(du.get('mess', '')).lower()
            mv = str(dv.get('mess', '')).lower()
            return MESS_COLOR_MAP.get(mu, '#888') if mu == mv and mu else '#BBBBBB'
        if color_mode == 'Community':
            cu = du.get('community', -1)
            cv = dv.get('community', -1)
            return '#444444' if cu == cv and cu >= 0 else '#AAAAAA'
        return '#888888'

    # Build edge segments grouped by color for performance
    edge_segments_by_color = {}
    pos = nx.spring_layout(G, weight='weight', seed=42)
    for u, v, d in G.edges(data=True):
        c = edge_color(u, v, G.nodes[u], G.nodes[v])
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_segments_by_color.setdefault(c, {'x': [], 'y': []})
        edge_segments_by_color[c]['x'] += [x0, x1, None]
        edge_segments_by_color[c]['y'] += [y0, y1, None]
    for c, seg in edge_segments_by_color.items():
        edge_traces.append(go.Scatter(x=seg['x'], y=seg['y'], line=dict(width=1, color=c), hoverinfo='none', mode='lines'))

    # Node trace
    node_x = []
    node_y = []
    text = []
    marker_color_vals = []
    node_ids = []
    for idx, (n, data) in enumerate(G.nodes(data=True)):
        x, y = pos[n]
        node_x.append(x)
        node_y.append(y)
        node_ids.append(n)
        text.append(
            f"{data.get('label', n)}<br>cgpa={data.get('cgpa','NA')}<br>hostel={data.get('hostel','')}<br>mess={data.get('mess','')}<br>community={data.get('community')}"
        )
        if color_mode == 'Community':
            comm = data.get("community", -1)
            marker_color_vals.append(comm if comm >= 0 else -1)
        else:
            marker_color_vals.append(node_colors[idx])

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers",
        hoverinfo="text",
        text=text,
        customdata=node_ids,  # enable click -> node id
        marker=dict(
            showscale=True,
            colorscale="Viridis" if color_mode == 'Community' else None,
            color=marker_color_vals,
            size=14,
            colorbar=dict(title="Community" if color_mode == 'Community' else "Attribute"),
            line_width=1,
        ),
    )

    fig = go.Figure(data=edge_traces + [node_trace], layout=go.Layout(
        title="Student Similarity Network",
        showlegend=False,
        hovermode="closest",
        margin=dict(b=20, l=20, r=20, t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    ))
    return fig


# ------------------- Streamlit UI -------------------

def run_streamlit_app():
    st.set_page_config(page_title="ConnectMe Graph", layout="wide")
    st.title("ConnectMe: Student Similarity Graph")
    uploaded = st.file_uploader(
        "Upload CSV / Excel (or PDF with table)",
        type=["csv", "xlsx", "xls", "pdf"],
        help="Data file. For PDF you must install pdfplumber (first table only)."
    )
    if not uploaded:
        st.info("Use the uploader above to select a data file.")
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

    # Sidebar controls (weights + active fields)
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

    color_mode = st.sidebar.selectbox("Color by", ["Community", "Hostel", "CGPA", "Mess"])

    # Build graph
    G, _, _, _ = build_and_summarise(str(temp_path), config)

    if G.number_of_nodes() == 0:
        st.error("No nodes generated. Check input file format.")
        return
    if G.number_of_edges() == 0:
        st.warning("Graph has zero edges at current weights. Lower 'Min edge weight' or adjust weights/fields.")

    if go:
        fig = plot_interactive_network(G, color_mode=color_mode)
        st.plotly_chart(fig, use_container_width=True)
        # Legend
        if color_mode == 'Hostel':
            for k, v in HOSTEL_COLOR_MAP.items():
                st.markdown(f"<span style='display:inline-block;width:14px;height:14px;background:{v};border-radius:2px;margin-right:6px;'></span>{k.upper()}", unsafe_allow_html=True)
        elif color_mode == 'Mess':
            for k, v in MESS_COLOR_MAP.items():
                st.markdown(f"<span style='display:inline-block;width:14px;height:14px;background:{v};border-radius:2px;margin-right:6px;'></span>{k.upper()}", unsafe_allow_html=True)
        elif color_mode == 'CGPA':
            order = [('high','>9'), ('mid','6-9'), ('low','<6')]
            for key,label in order:
                v = CGPA_COLOR_MAP[key]
                st.markdown(f"<span style='display:inline-block;width:14px;height:14px;background:{v};border-radius:2px;margin-right:6px;'></span>{label}", unsafe_allow_html=True)
        else:
            st.caption("Community colorscale applied.")
        # Node details on click
        if plotly_events:
            selected = plotly_events(fig, click_event=True, hover_event=False)
            if selected:
                node_id = selected[0].get('customdata')
                if node_id in G.nodes:
                    st.subheader(f"Details: {G.nodes[node_id].get('label', node_id)}")
                    st.json(G.nodes[node_id])
        else:
            st.caption("Install streamlit-plotly-events for clickable node details: pip install streamlit-plotly-events")
    else:
        st.error("Plotly not installed. Run: pip install plotly")


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
