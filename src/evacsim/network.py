from __future__ import annotations

from collections import defaultdict
from dataclasses import replace

from .models import Edge


class CityNetwork:
    """Directed graph with mutable route availability and edge occupancy."""

    def __init__(self) -> None:
        self._adjacency: dict[str, list[str]] = defaultdict(list)
        self._edges: dict[tuple[str, str], Edge] = {}
        self._occupancy: dict[tuple[str, str], int] = defaultdict(int)

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
        self._add_directed(Edge(source, target, distance, capacity, speed_limit))
        if bidirectional:
            self._add_directed(Edge(target, source, distance, capacity, speed_limit))

    def _add_directed(self, edge: Edge) -> None:
        key = (edge.source, edge.target)
        if key not in self._edges:
            self._adjacency[edge.source].append(edge.target)
        self._edges[key] = edge

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
        self._edges[key] = replace(self._edges[key], blocked=blocked)

    def travel_rate(self, source: str, target: str, agent_speed: float) -> float:
        """Distance per tick under a deterministic volume-delay relationship."""
        edge = self.edge(source, target)
        free_flow = min(agent_speed, edge.speed_limit)
        load = self.congestion(source, target)
        slowdown = 1.0 + 2.0 * load * load
        return free_flow / slowdown

