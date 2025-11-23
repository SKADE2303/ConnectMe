# ConnectMe

ConnectMe builds an interactive, graph‑based view of student similarity to help discover peers with shared interests, backgrounds, and performance profiles.

## Why this is important
Traditional lists / spreadsheets make it hard to see clusters of students who share multiple attributes (hostel, mess, sports, clubs, academic similarity). A similarity graph:
- Reveals communities (detected via modularity) that may correspond to natural working / interest groups.
- Highlights bridging students via betweenness and edge betweenness centrality (potential peer mentors / connectors).
- Enables targeted outreach (e.g. identify all high‑CGPA students who also do robotics or football).
- Helps plan events or study groups by visualising overlap patterns.

## Core features
- Dynamic CSV / Excel (and optional PDF table) ingestion with automatic header & field type inference.
- Automatic detection of:
  - Multi‑valued fields (comma / semicolon separated) → intersection similarity.
  - Single categorical fields → match similarity.
  - Numeric fields → inverse difference similarity (e.g. CGPA).
- Aggregation of Club1, Club2, ... style columns into a synthetic multi field `clubs`.
- Weighted similarity edges (configurable weights for multi/shared/numeric matches + minimum edge threshold).
- Graph metrics: node betweenness, degree, closeness, edge betweenness, greedy modularity communities.
- Interactive Streamlit + Plotly network (color by community, hostel, mess, or CGPA buckets; click for details).
- Filtering by selectable active fields (limit which attributes contribute to similarity).

## Installation
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# Optional for PDF support and clickable nodes
pip install pdfplumber streamlit-plotly-events
```

## Data format
Headers are inferred; recommended columns (case insensitive examples):
- id, name
- cgpa (numeric) or any other numeric metric
- hostel, mess (categorical)
- sports, clubs (multi-valued, separated by commas / semicolons)
- Club1, Club2, Club3 ... (will be merged into `clubs` automatically)

Example row:
```
S001,Aditi Sharma,8.4,H1,M2,Basketball,Robotics, Music
```

## Running the interactive app (recommended)
```bash
streamlit run visualise.py
```
Then:
1. Upload your CSV / Excel (or PDF table with pdfplumber installed).
2. Adjust weights & select which fields contribute to similarity.
3. Choose a color mode (Community / Hostel / CGPA / Mess).
4. Click nodes (if `streamlit-plotly-events` installed) to view full attributes & centralities.

## Command line batch mode
```bash
python visualise.py --input students.csv --output-prefix out
```
Outputs:
- `out_nodes.csv` / `out_edges.csv`
- `out.gml` (GraphML-exportable) or HTML network (if Plotly installed)

## Configuration / Tuning
Weights (in sidebar) affect edge formation and density:
- Min edge weight: prunes weak similarities.
- Shared multi item weight: each shared item (sport/club/etc.).
- Single field weight: identical hostel/mess/etc.
- Numeric similarity multiplier: scales inverse difference contribution for numeric fields.

Reduce `min edge weight` if graph is too sparse; increase if too dense.


## Contributing
1. Fork & create a feature branch.
2. Ensure new dependencies go into `requirements.txt`.
3. Provide clear commit messages & update README if behaviour changes.
