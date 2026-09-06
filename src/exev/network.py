from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
from math import isfinite

from .models import Edge


class CityNetwork:
    """Directed graph with mutable route availability and edge occupancy."""

    def __init__(self) -> None:
        self._adjacency: dict[str, list[str]] = defaultdict(list)
        self._edges: dict[tuple[str, str], Edge] = {}
        self._occupancy: dict[tuple[str, str], int] = defaultdict(int)
        self._incoming: dict[str, list[str]] = defaultdict(list)
        self._positions: dict[str, tuple[float, float]] = {}
        self._edge_hazards: dict[tuple[str, str], float] = defaultdict(float)
        self._node_hazards: dict[str, float] = defaultdict(float)
        self._topology_version = 0
        self._hazard_version = 0

    @property
    def topology_version(self) -> int:
        """Changes to geometry, costs, or closures invalidate route caches."""
        return self._topology_version

    @property
    def hazard_version(self) -> int:
        """Changes to hazard intensity invalidate hazard-aware route caches."""
        return self._hazard_version

    @property
    def edges(self) -> tuple[Edge, ...]:
        return tuple(self._edges.values())

    def add_node(self, node: str) -> None:
        if node not in self._adjacency:
            self._adjacency[node] = []
            self._topology_version += 1

    def set_position(self, node: str, x: float, y: float) -> None:
        if not isfinite(x) or not isfinite(y):
            raise ValueError("coordinates must be finite")
        self.add_node(node)
        if self._positions.get(node) != (x, y):
            self._positions[node] = (x, y)
            self._topology_version += 1

    def position(self, node: str) -> tuple[float, float] | None:
        return self._positions.get(node)

    def occupancy_snapshot(self) -> dict[tuple[str, str], int]:
        return dict(self._occupancy)

    @property
    def nodes(self) -> set[str]:
        result = set(self._adjacency)
        for source, target in self._edges:
            result.add(source)
            result.add(target)
        return result

    def add_edge(
        self,
        source: str,
        target: str,
        distance: float,
        capacity: int,
        speed_limit: float = 1.0,
        *,
        bidirectional: bool = True,
    ) -> None:
        edges = [Edge(source, target, distance, capacity, speed_limit)]
        if bidirectional and source != target:
            edges.append(Edge(target, source, distance, capacity, speed_limit))
        # Validate both directions before mutating either direction.
        if any(self._occupancy.get((edge.source, edge.target), 0) for edge in edges):
            raise ValueError("cannot replace an occupied edge")
        for edge in edges:
            self._add_directed(edge)

    def _add_directed(self, edge: Edge) -> None:
        key = (edge.source, edge.target)
        if self._occupancy.get(key, 0):
            raise ValueError("cannot replace an occupied edge")
        if key not in self._edges:
            self._adjacency[edge.source].append(edge.target)
            self._incoming[edge.target].append(edge.source)
        self._edges[key] = edge
        self._topology_version += 1

    def predecessors(self, node: str) -> list[str]:
        return [source for source in self._incoming.get(node, ())
                if not self._edges[(source, node)].blocked]

    def neighbors(self, node: str, *, include_blocked: bool = False) -> list[str]:
        if include_blocked:
            return list(self._adjacency.get(node, ()))
        return [
            target
            for target in self._adjacency.get(node, ())
            if not self._edges[(node, target)].blocked
        ]

    def edge(self, source: str, target: str) -> Edge:
        return self._edges[(source, target)]

    def has_edge(self, source: str, target: str) -> bool:
        return (source, target) in self._edges

    def occupancy(self, source: str, target: str) -> int:
        return self._occupancy[(source, target)]

    def congestion(self, source: str, target: str) -> float:
        edge = self.edge(source, target)
        return self.occupancy(source, target) / edge.capacity

    def can_enter(self, source: str, target: str) -> bool:
        edge = self.edge(source, target)
        return not edge.blocked and self.occupancy(source, target) < edge.capacity

    def enter(self, source: str, target: str) -> None:
        if not self.can_enter(source, target):
            raise RuntimeError(f"edge {source}->{target} is unavailable")
        self._occupancy[(source, target)] += 1

    def leave(self, source: str, target: str) -> None:
        key = (source, target)
        if self._occupancy[key] <= 0:
            raise RuntimeError(f"edge {source}->{target} has no occupants")
        self._occupancy[key] -= 1

    def set_blocked(self, source: str, target: str, blocked: bool = True) -> None:
        key = (source, target)
        if self._edges[key].blocked != blocked:
            self._edges[key] = replace(self._edges[key], blocked=blocked)
            self._topology_version += 1

    def set_edge_hazard(self, source: str, target: str, intensity: float) -> None:
        if not isfinite(intensity) or intensity < 0:
            raise ValueError("hazard intensity must be finite and non-negative")
        key = (source, target)
        if key not in self._edges:
            raise KeyError(key)
        if self._edge_hazards[key] != intensity:
            self._edge_hazards[key] = intensity
            self._hazard_version += 1

    def edge_hazard(self, source: str, target: str) -> float:
        return self._edge_hazards[(source, target)]

    def set_node_hazard(self, node: str, intensity: float) -> None:
        if not isfinite(intensity) or intensity < 0:
            raise ValueError("hazard intensity must be finite and non-negative")
        if node not in self.nodes:
            raise KeyError(node)
        if self._node_hazards[node] != intensity:
            self._node_hazards[node] = intensity
            self._hazard_version += 1

    def node_hazard(self, node: str) -> float:
        return self._node_hazards[node]

    def travel_rate(
        self, source: str, target: str, agent_speed: float, *, occupancy: int | None = None
    ) -> float:
        """Distance per tick under a deterministic volume-delay relationship."""
        edge = self.edge(source, target)
        free_flow = min(agent_speed, edge.speed_limit)
        load = (self.occupancy(source, target) if occupancy is None else occupancy) / edge.capacity
        slowdown = 1.0 + 2.0 * load * load
        return free_flow / slowdown
