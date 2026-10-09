"""Streamlit dashboard for FloodGuard Lite.

The interface is split into four pages to mirror the project specification:
Home, Flood Prediction, Route Planner and Analytics. The pages consume the
same ML model and graph-routing modules used elsewhere in the project so that
all behaviour is consistent with the training and routing phases.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import streamlit as st
from streamlit_folium import folium_static

from dashboard.map_builder import build_map
from routing.algorithms import astar, compare_algorithms, dijkstra
from routing.cost import apply_costs
from routing.graph_builder import attach_static_features, build_grid_graph, fetch_river_lines, load_osm_graph
from routing.risk import assign_flood_risk, load_model_bundle
from routing.shelters import find_shelters, route_to_best_shelter
from utils.config import DEFAULT_FLOOD_WEIGHT, METRICS_PATH, ROAD_TYPES, VALID_RANGES
from utils.features import build_features

logger = logging.getLogger(__name__)


def load_metrics(path: Path = METRICS_PATH) -> Dict[str, object]:
    """Load saved training metrics or return an empty structure."""
    if not path.exists():
        return {"best_model": "Unavailable", "results": {}}
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def build_scenario_graph(rainfall_mm: float, river_level_m: float,
                         flood_weight: float = DEFAULT_FLOOD_WEIGHT) -> nx.DiGraph:
    """Create a scenario graph with ML risk assigned and road costs applied.

    The function tries the real OSM-backed graph first, but falls back to the
    synthetic grid graph so the dashboard still works in offline class/demo
    environments.
    """
    try:
        graph = load_osm_graph()
        rivers = fetch_river_lines()
    except (RuntimeError, ValueError):
        graph = build_grid_graph()
        rivers = None

    attach_static_features(graph, rivers)
    bundle = load_model_bundle()
    assign_flood_risk(graph, rainfall_mm, river_level_m, bundle)
    apply_costs(graph, flood_weight)
    return graph


def _node_labels(graph: nx.DiGraph) -> List[Tuple[str, int]]:
    """Build readable node labels for selectboxes."""
    labels: List[Tuple[str, int]] = []
    for node in sorted(graph.nodes()):
        x = float(graph.nodes[node]["x"])
        y = float(graph.nodes[node]["y"])
        labels.append((f"Node {node}  ({y:.4f}, {x:.4f})", node))
    return labels


def render_home() -> None:
    """Explain the project and how the AI + routing system fits together."""
    st.title("FloodGuard Lite")
    st.subheader("AI-based Flood Evacuation Route Planner")

    st.markdown(
        """
        FloodGuard Lite is a small, local prototype for flood-aware route planning.
        It combines a supervised ML model with graph search to estimate which roads
        are likely to flood and then select a safer evacuation path.
        """
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("ML model", "Decision Tree / Random Forest")
    col2.metric("Planner", "A* + Dijkstra")
    col3.metric("Output", "Interactive evac map")

    st.markdown("### Why this design?")
    st.markdown(
        """
        - The ML layer predicts flood probability from rainfall, river level,
          elevation, road type, and historical flood patterns.
        - The graph layer turns roads into nodes and edges so route cost can reflect
          both distance and flood risk.
        - The route engine chooses the least-cost path rather than simply the shortest path.
        """
    )

    st.markdown("### System architecture")
    st.code(
        """Data -> Feature Engineering -> Model -> Flood Risk -> Graph Cost -> Route Search -> Map""",
        language="text",
    )

    st.markdown("### AI concepts in this project")
    st.markdown(
        """
        - Supervised learning: the model sees labelled road examples and learns which combinations
          of features lead to flooding.
        - Classification: each road is predicted as safe or flooded using a probability score.
        - Search algorithms: Dijkstra guarantees shortest paths under costs, while A* uses a
          heuristic to reach the goal faster without sacrificing optimality.
        """
    )


def render_prediction() -> None:
    """Predict flood probability for a road segment using the trained model."""
    st.title("Flood Prediction")
    st.caption("Enter a road scenario and estimate flood probability for that segment.")

    with st.form("prediction_form"):
        rainfall_mm = st.slider("Rainfall (mm)", min_value=0.0, max_value=400.0, value=180.0, step=5.0)
        river_level_m = st.slider("River level (m)", min_value=0.0, max_value=8.0, value=3.5, step=0.1)
        elevation_m = st.slider("Elevation (m)", min_value=0.0, max_value=25.0, value=8.0, step=0.5)
        distance_from_river_m = st.slider(
            "Distance from river (m)", min_value=0.0, max_value=5000.0, value=1200.0, step=50.0
        )
        historical_flood_freq = st.slider(
            "Historical flood frequency",
            min_value=0.0,
            max_value=10.0,
            value=3.0,
            step=0.1,
        )
        road_type = st.selectbox("Road type", options=ROAD_TYPES)
        submitted = st.form_submit_button("Predict Flood")

    if not submitted:
        st.info("Choose the conditions and click Predict Flood to compute a score.")
        return

    row = pd.DataFrame(
        [{
            "rainfall_mm": rainfall_mm,
            "river_level_m": river_level_m,
            "elevation_m": elevation_m,
            "distance_from_river_m": distance_from_river_m,
            "historical_flood_freq": historical_flood_freq,
            "road_type": road_type,
        }]
    )

    try:
        bundle = load_model_bundle()
        features = build_features(row)
        probability = float(bundle["model"].predict_proba(features[bundle["feature_columns"]])[0, 1])
    except (FileNotFoundError, ValueError) as exc:
        st.error(f"Prediction could not run: {exc}")
        return

    st.markdown(f"### Flood probability: {probability:.2%}")
    st.progress(min(1.0, max(0.0, probability)))

    if probability < 0.3:
        status = "Safe"
    elif probability < 0.7:
        status = "Moderate risk"
    else:
        status = "High flood risk"
    st.success(status)

    st.write("The probability is interpreted as the chance that this road segment will flood under the given conditions.")


def render_route_planner() -> None:
    """Let the user select a route and view the safe evacuation map."""
    st.title("Route Planner")

    rainfall_mm = st.slider("Rainfall (mm)", min_value=0.0, max_value=400.0, value=200.0, step=5.0)
    river_level_m = st.slider("River level (m)", min_value=0.0, max_value=8.0, value=4.0, step=0.1)
    flood_weight = st.slider("Flood risk weight", min_value=0.0, max_value=25.0, value=DEFAULT_FLOOD_WEIGHT, step=0.5)

    graph = build_scenario_graph(rainfall_mm, river_level_m, flood_weight)
    shelters = find_shelters(graph)
    node_labels = _node_labels(graph)

    start_choice = st.selectbox(
        "Start node",
        options=node_labels,
        format_func=lambda item: item[0],
        index=0,
    )
    end_choice = st.selectbox(
        "Destination node",
        options=node_labels,
        format_func=lambda item: item[0],
        index=len(node_labels) - 1,
    )
    start_node = start_choice[1]
    end_node = end_choice[1]

    if start_node == end_node:
        st.warning("Start and destination are the same node. Choose a different destination.")
        return

    with st.spinner("Computing the safest evacuation path..."):
        route_graph = graph.copy()
        apply_costs(route_graph, flood_weight)
        route = astar(route_graph, start_node, end_node)

        baseline_graph = graph.copy()
        apply_costs(baseline_graph, 0.0)
        baseline = dijkstra(baseline_graph, start_node, end_node)

    st.subheader("Route summary")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Recommended route", f"{route.distance_m / 1000:.2f} km")
    col2.metric("Mean risk", f"{route.mean_risk:.0%}")
    col3.metric("A* nodes visited", f"{route.nodes_visited}")
    col4.metric("Flood weight", f"{flood_weight:.1f}")

    st.markdown("### Comparison with plain shortest-distance route")
    st.write(
        f"Shortest path: {baseline.distance_m / 1000:.2f} km | "
        f"Mean risk: {baseline.mean_risk:.0%} | Nodes visited: {baseline.nodes_visited}"
    )

    m = build_map(graph, route, baseline, start_node, end_node, shelters)
    folium_static(m, width=1000, height=600)

    st.caption("Blue = recommended route, grey dashed = shortest-distance route. Green/yellow/red roads show the risk level.")


def render_analytics() -> None:
    """Display model quality metrics and algorithm comparison."""
    st.title("Analytics")
    metrics = load_metrics()
    results = metrics.get("results", {})

    if not results:
        st.warning("No training metrics are available yet. Run the training pipeline first.")
        return

    model_rows = []
    for name, values in results.items():
        model_rows.append(
            {
                "Model": name,
                "Accuracy": values.get("accuracy", 0.0),
                "Precision": values.get("precision", 0.0),
                "Recall": values.get("recall", 0.0),
                "F1": values.get("f1", 0.0),
                "ROC AUC": values.get("roc_auc", 0.0),
            }
        )
    df = pd.DataFrame(model_rows)
    st.dataframe(df, use_container_width=True)

    st.subheader("Confusion matrices")
    cols = st.columns(len(results))
    for idx, (name, values) in enumerate(results.items()):
        with cols[idx]:
            cm = values.get("confusion_matrix", [[0, 0], [0, 0]])
            fig, ax = plt.subplots(figsize=(3.2, 3.2))
            ax.imshow(cm, cmap="Blues")
            ax.set_xticks([0, 1])
            ax.set_yticks([0, 1])
            ax.set_xticklabels(["Safe", "Flooded"])
            ax.set_yticklabels(["Safe", "Flooded"])
            ax.set_title(name)
            for i in range(2):
                for j in range(2):
                    ax.text(j, i, cm[i][j], ha="center", va="center", color="black")
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

    st.subheader("Algorithm comparison")
    graph = build_scenario_graph(rainfall_mm=200.0, river_level_m=4.0, flood_weight=DEFAULT_FLOOD_WEIGHT)
    start = min(graph.nodes(), key=lambda n: graph.nodes[n]["y"])
    destination = max(graph.nodes(), key=lambda n: graph.nodes[n]["y"])
    comparisons = compare_algorithms(graph, start, destination, repeats=5)
    comparison_rows = [
        {
            "Algorithm": result.algorithm,
            "Cost": round(result.cost, 2),
            "Distance (km)": round(result.distance_m / 1000, 2),
            "Mean risk": round(result.mean_risk, 3),
            "Nodes visited": result.nodes_visited,
            "Runtime (s)": round(result.runtime_s, 6),
        }
        for result in comparisons
    ]
    st.dataframe(pd.DataFrame(comparison_rows), use_container_width=True)

    st.markdown(
        """
        A* usually performs better than Dijkstra in this routing task because it uses a
        heuristic based on straight-line distance to the destination. This lets it focus on
        promising routes instead of expanding every possible node, especially in a large road graph.
        """
    )


def main() -> None:
    """Run the FloodGuard Lite dashboard."""
    st.set_page_config(page_title="FloodGuard Lite", page_icon="🌊", layout="wide")

    pages = {
        "Home": render_home,
        "Flood Prediction": render_prediction,
        "Route Planner": render_route_planner,
        "Analytics": render_analytics,
    }

    page = st.sidebar.radio("Navigation", list(pages.keys()))
    pages[page]()


if __name__ == "__main__":
    main()
