"""Evacuation shelter selection and nearest-safe-shelter routing.

Shelters are chosen as well-separated high-ground intersections. This is a
prototype stand-in for real shelters (schools, community halls, hospitals),
which could be pulled from OpenStreetMap amenity tags in future work.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Callable, Dict, List

import networkx as nx

from routing.algorithms import NoRouteError, SearchResult, astar
from routing.graph_builder import haversine_m


def node_elevation(graph: nx.DiGraph) -> Dict[int, float]:
    """Estimate node elevation as the mean elevation of its incident edges."""
    totals: Dict[int, List[float]] = defaultdict(list)
    for u, v, data in graph.edges(data=True):
        if "elevation_m" not in data:
            raise ValueError("Edges lack elevation_m; call attach_static_features first")
        totals[u].append(data["elevation_m"])
        totals[v].append(data["elevation_m"])
    return {n: sum(vals) / len(vals) for n, vals in totals.items()}


def find_shelters(graph: nx.DiGraph, count: int = 5,
                  min_separation_m: float = 800.0) -> List[int]:
    """Greedily pick high-elevation nodes that are far apart from each other.

    Args:
        graph: graph with static features attached.
        count: number of shelters wanted.
        min_separation_m: minimum spacing between any two shelters.

    Returns:
        Up to ``count`` node ids (fewer if the area is too small).
    """
    if count < 1:
        raise ValueError("count must be >= 1")
    elevation = node_elevation(graph)
    chosen: List[int] = []
    for node in sorted(elevation, key=elevation.get, reverse=True):
        far_enough = all(
            haversine_m(graph.nodes[node]["y"], graph.nodes[node]["x"],
                        graph.nodes[c]["y"], graph.nodes[c]["x"]) >= min_separation_m
            for c in chosen)
        if far_enough:
            chosen.append(node)
        if len(chosen) == count:
            break
    return chosen


def route_to_best_shelter(graph: nx.DiGraph, source: int, shelters: List[int],
                          search: Callable[[nx.DiGraph, int, int], SearchResult] = astar
                          ) -> SearchResult:
    """Return the lowest-cost route from ``source`` to any reachable shelter.

    Raises:
        NoRouteError: if no shelter can be reached.
    """
    best = None
    for shelter in shelters:
        try:
            result = search(graph, source, shelter)
        except NoRouteError:
            continue
        if best is None or (result.cost, result.distance_m) < (best.cost, best.distance_m):
            best = result
    if best is None:
        raise NoRouteError(f"No shelter reachable from node {source}")
    return best
