"""Edge cost functions that fuse distance and flood risk.

Two formulations are provided so they can be compared in the paper:

* ``additive``      cost = length + weight * risk        (literal project spec)
* ``proportional``  cost = length * (1 + weight * risk)  (risk scales with exposure)

Both are non-negative and >= length, which keeps the straight-line A*
heuristic admissible.
"""
from __future__ import annotations

import math
from typing import Optional

import networkx as nx

COST_MODES = ("additive", "proportional")


def edge_cost(length_m: float, risk: float, weight: float,
              mode: str = "proportional",
              block_threshold: Optional[float] = None) -> float:
    """Compute the traversal cost of one road segment.

    Args:
        length_m: segment length in metres.
        risk: flood probability in [0, 1].
        weight: risk penalty strength (>= 0). weight=0 gives plain distance.
        mode: ``"additive"`` or ``"proportional"``.
        block_threshold: if set, edges with ``risk >= threshold`` are
            impassable (cost = inf).

    Raises:
        ValueError: on invalid arguments.
    """
    if mode not in COST_MODES:
        raise ValueError(f"mode must be one of {COST_MODES}")
    if weight < 0 or length_m < 0 or not 0.0 <= risk <= 1.0:
        raise ValueError("weight/length must be >= 0 and risk within [0, 1]")
    if block_threshold is not None and risk >= block_threshold:
        return math.inf
    if mode == "additive":
        return length_m + weight * risk
    return length_m * (1.0 + weight * risk)


def apply_costs(graph: nx.DiGraph, weight: float, mode: str = "proportional",
                block_threshold: Optional[float] = None) -> None:
    """Store ``cost`` on every edge (requires ``length`` and ``flood_prob``)."""
    for _, _, data in graph.edges(data=True):
        if "flood_prob" not in data:
            raise ValueError("Edges lack flood_prob; call assign_flood_risk first")
        data["cost"] = edge_cost(data["length"], data["flood_prob"], weight,
                                 mode, block_threshold)
