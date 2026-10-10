"""Streamlit dashboard for FloodGuard Lite.

The interface is split into four pages to mirror the project specification:
Home, Flood Prediction, Route Planner and Analytics. The pages use the same
ML model and graph-routing modules as the rest of the project, so behaviour
is identical to the training and routing phases.

Performance note: Streamlit re-runs this whole script on every interaction.
The road network, the trained model and the shelter list never change during
a session, so they are loaded once with ``st.cache_resource`` and every
scenario works on a private copy of the cached graph.
"""
from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

import networkx as nx
import pandas as pd
import streamlit as st
from matplotlib.figure import Figure
from streamlit_folium import st_folium

from dashboard.map_builder import build_map, risk_class
from routing.algorithms import (NoRouteError, SearchResult, astar,
                                compare_algorithms)
from routing.cost import apply_costs
from routing.graph_builder import (attach_static_features, build_grid_graph,
                                   fetch_river_lines, load_osm_graph)
from routing.risk import assign_flood_risk, load_model_bundle
from routing.shelters import find_shelters, node_elevation, route_to_best_shelter
from utils.config import (DEFAULT_FLOOD_WEIGHT, METRICS_PATH, RANDOM_SEED,
                          ROAD_TYPES, VALID_RANGES)
from utils.features import build_features

logger = logging.getLogger(__name__)

RISK_LABELS: Dict[str, str] = {
    "safe": "Safe", "moderate": "Moderate risk", "flooded": "High flood risk"}
MAX_FLOOD_WEIGHT: float = 25.0
BENCHMARK_PAIRS: int = 15


# --------------------------------------------------------------------------
# Cached resources and data helpers
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading road network...")
def get_base_graph() -> nx.DiGraph:
    """Load the road graph with static features (once per server session).

    Falls back to the synthetic grid so the dashboard works offline.
    Callers must ``copy()`` the result before modifying it.
    """
    try:
        graph = load_osm_graph()
        rivers = fetch_river_lines()
    except (RuntimeError, ValueError) as exc:
        logger.warning("Using synthetic grid graph (%s)", exc)
        graph, rivers = build_grid_graph(), None
    attach_static_features(graph, rivers)
    return graph


@st.cache_resource(show_spinner=False)
def get_model_bundle() -> Dict[str, Any]:
    """Load the trained model bundle once per server session."""
    return load_model_bundle()


@st.cache_resource(show_spinner=False)
def get_shelters() -> List[int]:
    """Choose shelters once; they depend only on static road features."""
    return find_shelters(get_base_graph())


def load_metrics(path: Path = METRICS_PATH) -> Dict[str, Any]:
    """Load saved training metrics or return an empty structure."""
    if not path.exists():
        return {"best_model": "Unavailable", "results": {}}
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def build_scenario_graph(rainfall_mm: float, river_level_m: float,
                         flood_weight: float = DEFAULT_FLOOD_WEIGHT) -> nx.DiGraph:
    """Return a private graph copy with ML flood risk and road costs applied."""
    graph = get_base_graph().copy()
    assign_flood_risk(graph, rainfall_mm, river_level_m, get_model_bundle())
    apply_costs(graph, flood_weight)
    return graph


def _node_label(graph: nx.DiGraph, node: int) -> str:
    """Readable selectbox label for one intersection."""
    return f"Node {node}  ({graph.nodes[node]['y']:.4f}, {graph.nodes[node]['x']:.4f})"


def _require_model() -> bool:
    """Show a friendly error and return False if the model is missing."""
    try:
        get_model_bundle()
    except FileNotFoundError as exc:
        st.error(f"{exc}")
        return False
    return True


def compute_route_result(rainfall_mm: float, river_level_m: float, flood_weight: float,
                         start: int, destination: Optional[int]) -> Dict[str, Any]:
    """Run the flood-aware search plus the plain shortest-distance baseline.

    ``destination=None`` means "nearest (cheapest) shelter".

    Raises:
        NoRouteError: if no passable route exists.
    """
    graph = build_scenario_graph(rainfall_mm, river_level_m, flood_weight)
    shelters = get_shelters()
    if destination is None:
        route = route_to_best_shelter(graph, start, shelters)
    else:
        route = astar(graph, start, destination)
    target = route.path[-1]
    apply_costs(graph, 0.0)            # private copy: switch to plain distance
    baseline = astar(graph, start, target)
    return {"graph": graph, "route": route, "baseline": baseline, "start": start,
            "destination": target, "shelters": shelters, "weight": flood_weight}


@st.cache_data(show_spinner="Benchmarking Dijkstra and A* ...")
def algorithm_benchmark(rainfall_mm: float, river_level_m: float, flood_weight: float,
                        pairs: int = BENCHMARK_PAIRS) -> pd.DataFrame:
    """Average both algorithms over several random start/destination pairs."""
    graph = build_scenario_graph(rainfall_mm, river_level_m, flood_weight)
    rng = random.Random(RANDOM_SEED)
    nodes = list(graph.nodes)
    rows: List[Dict[str, Any]] = []
    for _ in range(pairs):
        src, dst = rng.sample(nodes, 2)
        try:
            for result in compare_algorithms(graph, src, dst, repeats=3):
                rows.append({"Algorithm": result.algorithm, "Cost": result.cost,
                             "Distance (km)": result.distance_m / 1000,
                             "Mean risk": result.mean_risk,
                             "Nodes visited": result.nodes_visited,
                             "Runtime (ms)": result.runtime_s * 1000})
        except NoRouteError:
            continue
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).groupby("Algorithm", sort=False).mean().round(3).reset_index()


# --------------------------------------------------------------------------
# Pages
# --------------------------------------------------------------------------
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
    if not _require_model():
        return

    def slider(label: str, key: str, default: float, step: float) -> float:
        low, high = VALID_RANGES[key]
        return st.slider(label, min_value=low, max_value=high, value=default, step=step)

    with st.form("prediction_form"):
        rainfall_mm = slider("Rainfall (mm)", "rainfall_mm", 180.0, 5.0)
        river_level_m = slider("River level (m)", "river_level_m", 3.5, 0.1)
        elevation_m = slider("Elevation (m)", "elevation_m", 8.0, 0.5)
        distance_m = slider("Distance from river (m)", "distance_from_river_m", 1200.0, 50.0)
        history = slider("Historical flood frequency", "historical_flood_freq", 3.0, 0.1)
        road_type = st.selectbox("Road type", options=ROAD_TYPES)
        submitted = st.form_submit_button("Predict Flood")

    if not submitted:
        st.info("Choose the conditions and click Predict Flood to compute a score.")
        return

    row = pd.DataFrame([{
        "rainfall_mm": rainfall_mm, "river_level_m": river_level_m,
        "elevation_m": elevation_m, "distance_from_river_m": distance_m,
        "historical_flood_freq": history, "road_type": road_type}])
    try:
        bundle = get_model_bundle()
        features = build_features(row)[bundle["feature_columns"]]
        probability = float(bundle["model"].predict_proba(features)[0, 1])
    except ValueError as exc:
        st.error(f"Prediction could not run: {exc}")
        return

    st.markdown(f"### Flood probability: {probability:.2%}")
    st.progress(min(1.0, max(0.0, probability)))
    label = RISK_LABELS[risk_class(probability)]
    (st.success if label == "Safe" else st.warning if label == "Moderate risk" else st.error)(label)
    st.write("The probability is interpreted as the chance that this road segment "
             "will flood under the given conditions.")


def _show_route_result(result: Dict[str, Any]) -> None:
    """Display metrics, a comparison table and the map for a computed route."""
    route: SearchResult = result["route"]
    baseline: SearchResult = result["baseline"]
    if len(route.path) < 2:
        st.info("The start is already a shelter, so no travel is needed.")
    extra = (route.distance_m / baseline.distance_m - 1.0) if baseline.distance_m else 0.0

    st.subheader("Route summary")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Recommended route", f"{route.distance_m / 1000:.2f} km", f"{extra:+.0%} vs shortest")
    col2.metric("Mean risk", f"{route.mean_risk:.0%}",
                f"{route.mean_risk - baseline.mean_risk:+.0%} vs shortest", delta_color="inverse")
    col3.metric("Max risk on route", f"{route.max_risk:.0%}")
    col4.metric("Flood weight", f"{result['weight']:.1f}")

    st.markdown("### Comparison with plain shortest-distance route")
    st.dataframe(pd.DataFrame([
        {"Route": "Recommended (flood-aware A*)", "Distance (km)": round(route.distance_m / 1000, 2),
         "Mean risk": round(route.mean_risk, 3), "Max risk": round(route.max_risk, 3),
         "Nodes visited": route.nodes_visited},
        {"Route": "Shortest distance (A*)", "Distance (km)": round(baseline.distance_m / 1000, 2),
         "Mean risk": round(baseline.mean_risk, 3), "Max risk": round(baseline.max_risk, 3),
         "Nodes visited": baseline.nodes_visited},
    ]), hide_index=True)

    fmap = build_map(result["graph"], route if len(route.path) > 1 else None,
                     baseline if len(baseline.path) > 1 else None,
                     result["start"], result["destination"], result["shelters"])
    st_folium(fmap, height=600, use_container_width=True, returned_objects=[])
    st.caption("Blue = recommended route, grey dashed = shortest-distance route. "
               "Green/yellow/red roads show the risk level; purple markers are shelters.")


def render_route_planner() -> None:
    """Let the user choose a scenario and view the safest evacuation route."""
    st.title("Route Planner")
    if not _require_model():
        return

    graph = get_base_graph()
    nodes = sorted(graph.nodes)
    elevation = node_elevation(graph)
    default_start = nodes.index(min(elevation, key=elevation.get))   # lowest-lying junction

    with st.form("route_form"):
        rain_lo, rain_hi = VALID_RANGES["rainfall_mm"]
        river_lo, river_hi = VALID_RANGES["river_level_m"]
        rainfall_mm = st.slider("Rainfall (mm)", rain_lo, rain_hi, 200.0, 5.0)
        river_level_m = st.slider("River level (m)", river_lo, river_hi, 4.0, 0.1)
        flood_weight = st.slider("Flood risk weight", 0.0, MAX_FLOOD_WEIGHT,
                                 float(DEFAULT_FLOOD_WEIGHT), 0.5)
        start_node = st.selectbox("Start node", options=nodes, index=default_start,
                                  format_func=lambda n: _node_label(graph, n))
        destination = st.selectbox(
            "Destination", options=[None] + nodes, index=0,
            format_func=lambda n: "Nearest shelter (automatic)" if n is None
            else _node_label(graph, n))
        submitted = st.form_submit_button("Compute Route")

    if submitted:
        if destination == start_node:
            st.warning("Start and destination are the same node. Choose a different destination.")
            st.session_state.pop("route_result", None)
        else:
            try:
                with st.spinner("Computing the safest evacuation path..."):
                    st.session_state["route_result"] = compute_route_result(
                        rainfall_mm, river_level_m, flood_weight, start_node, destination)
            except NoRouteError as exc:
                st.session_state.pop("route_result", None)
                st.error(f"No passable route: {exc}")

    if "route_result" in st.session_state:
        _show_route_result(st.session_state["route_result"])
    else:
        st.info("Choose a scenario and click Compute Route.")


def _confusion_figure(name: str, matrix: List[List[int]]) -> Figure:
    """Draw one confusion matrix (uses Figure directly: no global pyplot state)."""
    fig = Figure(figsize=(3.2, 3.2))
    ax = fig.subplots()
    ax.imshow(matrix, cmap="Blues")
    ax.set_xticks([0, 1], labels=["Safe", "Flooded"])
    ax.set_yticks([0, 1], labels=["Safe", "Flooded"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(name)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, matrix[i][j], ha="center", va="center", color="black")
    fig.tight_layout()
    return fig


def render_analytics() -> None:
    """Display model quality metrics and algorithm comparison."""
    st.title("Analytics")
    metrics = load_metrics()
    results = metrics.get("results", {})

    if not results:
        st.warning("No training metrics are available yet. Run the training pipeline first.")
        return

    st.caption(f"Selected model: {metrics.get('best_model', 'n/a')}")
    st.dataframe(pd.DataFrame([
        {"Model": name, "Accuracy": v.get("accuracy", 0.0), "Precision": v.get("precision", 0.0),
         "Recall": v.get("recall", 0.0), "F1": v.get("f1", 0.0), "ROC AUC": v.get("roc_auc", 0.0)}
        for name, v in results.items()]), hide_index=True)

    st.subheader("Confusion matrices")
    for column, (name, values) in zip(st.columns(len(results)), results.items()):
        with column:
            st.pyplot(_confusion_figure(name, values.get("confusion_matrix", [[0, 0], [0, 0]])))

    st.subheader("Algorithm comparison")
    if not _require_model():
        return
    table = algorithm_benchmark(200.0, 4.0, float(DEFAULT_FLOOD_WEIGHT))
    if table.empty:
        st.warning("No routable start/destination pairs were found for the benchmark.")
    else:
        st.caption(f"Average over up to {BENCHMARK_PAIRS} random start/destination pairs "
                   "(scenario: 200 mm rain, 4 m river level).")
        st.dataframe(table, hide_index=True)

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