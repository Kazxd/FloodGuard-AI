"""Dijkstra and A* shortest-path search over the flood-aware road graph.

Both share one best-first search loop. They differ only in the priority:
Dijkstra uses g(n) (cost so far); A* uses g(n) + h(n), where h is the
straight-line distance to the goal. h never overestimates the true remaining
cost (every edge cost >= its length >= the straight-line gap), so A* is
optimal and, with a consistent h, never expands more nodes than Dijkstra
(up to tie-breaking).
"""
from __future__ import annotations

import heapq
import itertools
import math
import statistics
import time
from dataclasses import dataclass
from typing import Callable, Dict, List

import networkx as nx

from routing.graph_builder import haversine_m

Heuristic = Callable[[int], float]


class NoRouteError(Exception):
    """Raised when no passable route exists between source and target."""


@dataclass
class SearchResult:
    """Outcome and metrics of one route search."""
    algorithm: str
    path: List[int]
    cost: float
    distance_m: float
    mean_risk: float        # length-weighted average flood probability
    max_risk: float
    nodes_visited: int      # nodes expanded (removed from the priority queue)
    runtime_s: float


def _search(graph: nx.DiGraph, source: int, target: int,
            heuristic: Heuristic, name: str) -> SearchResult:
    """Generic best-first search; returns metrics for the optimal path."""
    for node in (source, target):
        if node not in graph:
            raise KeyError(f"Node {node} not in graph")

    start = time.perf_counter()
    counter = itertools.count()  # tie-breaker so the heap never compares nodes
    g_score: Dict[int, float] = {source: 0.0}
    parent: Dict[int, int] = {}
    closed = set()
    heap = [(heuristic(source), next(counter), source)]
    found = False

    while heap:
        _, _, u = heapq.heappop(heap)
        if u in closed:
            continue
        closed.add(u)
        if u == target:
            found = True
            break
        for v, data in graph[u].items():
            if v in closed:
                continue
            cost = data["cost"]
            if math.isinf(cost):
                continue  # blocked (flooded) road
            new_g = g_score[u] + cost
            if new_g < g_score.get(v, math.inf):
                g_score[v] = new_g
                parent[v] = u
                heapq.heappush(heap, (new_g + heuristic(v), next(counter), v))
    runtime = time.perf_counter() - start

    if not found:
        raise NoRouteError(f"No passable route from {source} to {target}")

    path = [target]
    while path[-1] != source:
        path.append(parent[path[-1]])
    path.reverse()

    lengths = [graph[a][b]["length"] for a, b in zip(path, path[1:])]
    risks = [graph[a][b]["flood_prob"] for a, b in zip(path, path[1:])]
    total = sum(lengths)
    mean_risk = sum(l * r for l, r in zip(lengths, risks)) / total if total else 0.0
    return SearchResult(name, path, g_score[target], total, mean_risk,
                        max(risks, default=0.0), len(closed), runtime)


def dijkstra(graph: nx.DiGraph, source: int, target: int) -> SearchResult:
    """Dijkstra's algorithm on the ``cost`` edge attribute (h = 0)."""
    return _search(graph, source, target, lambda _n: 0.0, "Dijkstra")


def astar(graph: nx.DiGraph, source: int, target: int) -> SearchResult:
    """A* using great-circle distance to the target as heuristic."""
    ty, tx = graph.nodes[target]["y"], graph.nodes[target]["x"]

    def heuristic(node: int) -> float:
        return haversine_m(graph.nodes[node]["y"], graph.nodes[node]["x"], ty, tx)

    return _search(graph, source, target, heuristic, "A*")


def compare_algorithms(graph: nx.DiGraph, source: int, target: int,
                       repeats: int = 10) -> List[SearchResult]:
    """Run both algorithms and report median runtime over ``repeats`` runs."""
    if repeats < 1:
        raise ValueError("repeats must be >= 1")
    results = []
    for fn in (dijkstra, astar):
        runs = [fn(graph, source, target) for _ in range(repeats)]
        best = runs[0]
        best.runtime_s = statistics.median(r.runtime_s for r in runs)
        results.append(best)
    return results
