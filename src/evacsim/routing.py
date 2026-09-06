"""Classical routing with explicit objectives and reproducible cache policies."""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from heapq import heappop, heappush
from math import hypot, isfinite
from typing import Protocol

from .models import Agent, Shelter
from .network import CityNetwork

ROUTER_NAMES = ("dijkstra", "astar", "congestion-aware", "min-cost-flow")


@dataclass(slots=True)
class RoutingStats:
    route_calls: int = 0
    searches: int = 0
    expanded_nodes: int = 0
    cache_hits: int = 0


class RoutingStrategy(Protocol):
    def route(self, agent: Agent, network: CityNetwork,
              shelters: Mapping[str, Shelter], tick: int) -> list[str] | None: ...


class BaseRouter:
    name = "custom"
    replan_at_nodes = False

    def __init__(self) -> None:
        self.stats = RoutingStats()

    def prepare(self, agents: Sequence[Agent], network: CityNetwork,
                shelters: Mapping[str, Shelter]) -> None:
        self.stats = RoutingStats()

    def begin_tick(self, network: CityNetwork, shelters: Mapping[str, Shelter], tick: int) -> None:
        pass

    def can_admit(self, agent: Agent, node: str) -> bool:
        return True


def shortest_path(network: CityNetwork, start: str, targets: set[str],
                  stats: RoutingStats | None = None,
                  heuristic: Callable[[str], float] | None = None) -> list[str] | None:
    """Forward Dijkstra, or A* with the provided admissible heuristic."""
    if stats is not None:
        stats.searches += 1
    if not targets:
        return None
    h = heuristic or (lambda node: 0.0)
    queue = [(h(start), 0.0, start)]
    distance = {start: 0.0}
    previous: dict[str, str] = {}
    while queue:
        _, cost, node = heappop(queue)
        if cost != distance[node]:
            continue
        if stats is not None:
            stats.expanded_nodes += 1
        if node in targets:
            path = [node]
            while path[-1] != start:
                path.append(previous[path[-1]])
            return path[::-1]
        for target in network.neighbors(node):
            candidate = cost + network.edge(node, target).distance
            if candidate < distance.get(target, float("inf")):
                distance[target] = candidate
                previous[target] = node
                heappush(queue, (candidate + h(target), candidate, target))
    return None


def _reverse_tree(network: CityNetwork, goals: tuple[str, ...],
                  weight: Callable[[str, str], float], stats: RoutingStats
                  ) -> tuple[dict[str, str], set[str]]:
    """One multi-source Dijkstra tree supplies routes from every origin."""
    stats.searches += 1
    queue: list[tuple[float, str]] = []
    distance = {goal: 0.0 for goal in goals}
    next_hop: dict[str, str] = {}
    for goal in goals:
        heappush(queue, (0.0, goal))
    while queue:
        cost, node = heappop(queue)
        if cost != distance[node]:
            continue
        stats.expanded_nodes += 1
        for source in network.predecessors(node):
            candidate = cost + weight(source, node)
            if candidate < distance.get(source, float("inf")):
                distance[source] = candidate
                next_hop[source] = node
                heappush(queue, (candidate, source))
    return next_hop, set(distance)


def _tree_path(start: str, tree: tuple[dict[str, str], set[str]]) -> list[str] | None:
    next_hop, reachable = tree
    if start not in reachable:
        return None
    path = [start]
    while path[-1] in next_hop:
        path.append(next_hop[path[-1]])
    return path


def _goals(shelters: Mapping[str, Shelter]) -> tuple[str, ...]:
    return tuple(sorted(node for node, shelter in shelters.items() if shelter.available > 0))


class DijkstraRouter(BaseRouter):
    """Cached reverse Dijkstra, minimizing distance to any open shelter."""
    name = "dijkstra"

    def __init__(self) -> None:
        super().__init__()
        self._key: tuple[CityNetwork, int, tuple[str, ...]] | None = None
        self._tree: tuple[dict[str, str], set[str]] = ({}, set())

    def prepare(self, agents: Sequence[Agent], network: CityNetwork,
                shelters: Mapping[str, Shelter]) -> None:
        super().prepare(agents, network, shelters)
        self._key = None
        self._tree = ({}, set())

    def route(self, agent: Agent, network: CityNetwork,
              shelters: Mapping[str, Shelter], tick: int) -> list[str] | None:
        goals = _goals(shelters)
        key = (network, network.topology_version, goals)
        if key != self._key:
            self._tree = _reverse_tree(network, goals,
                                       lambda u, v: network.edge(u, v).distance, self.stats)
            self._key = key
        else:
            self.stats.cache_hits += 1
        return _tree_path(agent.current_node, self._tree)


class ShortestDistanceRouter(DijkstraRouter):
    """Compatibility alias for the original Stage 1 router."""


class AStarRouter(BaseRouter):
    """A* with scaled Euclidean lower bounds and per-origin route caching."""
    name = "astar"

    def __init__(self) -> None:
        super().__init__()
        self._key: tuple[CityNetwork, int, tuple[str, ...]] | None = None
        self._paths: dict[str, list[str] | None] = {}
        self._scale = 0.0

    def prepare(self, agents: Sequence[Agent], network: CityNetwork,
                shelters: Mapping[str, Shelter]) -> None:
        super().prepare(agents, network, shelters)
        self._key = None
        self._paths.clear()

    @staticmethod
    def _safe_scale(network: CityNetwork) -> float:
        # Raw geometry is not a distance lower bound on arbitrary weighted graphs.
        # Every edge must cost at least scale * its geometric length.
        if any(network.position(node) is None for node in network.nodes):
            return 0.0
        scale = float("inf")
        for edge in network.edges:
            a, b = network.position(edge.source), network.position(edge.target)
            assert a is not None and b is not None
            length = hypot(a[0] - b[0], a[1] - b[1])
            if length > 0:
                scale = min(scale, edge.distance / length)
        return scale if isfinite(scale) else 0.0

    def route(self, agent: Agent, network: CityNetwork,
              shelters: Mapping[str, Shelter], tick: int) -> list[str] | None:
        goals = _goals(shelters)
        key = (network, network.topology_version, goals)
        if key != self._key:
            self._paths.clear()
            self._scale = self._safe_scale(network)
            self._key = key
        start = agent.current_node
        if start in self._paths:
            self.stats.cache_hits += 1
        else:
            def heuristic(node: str) -> float:
                if not goals or not self._scale:
                    return 0.0
                position = network.position(node)
                target_positions = [network.position(goal) for goal in goals]
                if position is None or any(target is None for target in target_positions):
                    return 0.0
                return self._scale * min(
                    hypot(position[0] - target[0], position[1] - target[1])
                    for target in target_positions if target is not None
                )

            self._paths[start] = shortest_path(network, start, set(goals), self.stats, heuristic)
        path = self._paths[start]
        return list(path) if path is not None else None


class CongestionAwareRouter(BaseRouter):
    """Snapshot travel-time routing; share one tree per speed and tick.

    Projected occupancy includes the requesting agent, capped at road capacity.
    Full roads remain routeable: agents queue at their entrance. This estimate
    does not predict downstream queues or future traffic.
    """
    name = "congestion-aware"
    replan_at_nodes = True

    def __init__(self, alpha: float = 2.0) -> None:
        super().__init__()
        if not isfinite(alpha) or alpha < 0:
            raise ValueError("alpha must be finite and non-negative")
        self.alpha = alpha
        self._snapshot_key: tuple[CityNetwork, int, int] | None = None
        self._occupancy: dict[tuple[str, str], int] = {}
        self._trees: dict[tuple[float, tuple[str, ...]], tuple[dict[str, str], set[str]]] = {}

    def prepare(self, agents: Sequence[Agent], network: CityNetwork,
                shelters: Mapping[str, Shelter]) -> None:
        super().prepare(agents, network, shelters)
        self._snapshot_key = None
        self._trees.clear()

    def begin_tick(self, network: CityNetwork, shelters: Mapping[str, Shelter], tick: int) -> None:
        key = (network, network.topology_version, tick)
        if key != self._snapshot_key:
            self._snapshot_key = key
            self._occupancy = network.occupancy_snapshot()
            self._trees.clear()

    def route(self, agent: Agent, network: CityNetwork,
              shelters: Mapping[str, Shelter], tick: int) -> list[str] | None:
        self.begin_tick(network, shelters, tick)
        goals = _goals(shelters)
        key = (agent.speed, goals)
        if key not in self._trees:
            def weight(source: str, target: str) -> float:
                edge = network.edge(source, target)
                load = min(1.0, (self._occupancy.get((source, target), 0) + 1) / edge.capacity)
                return edge.distance / min(agent.speed, edge.speed_limit) * (1 + self.alpha * load**2)

            self._trees[key] = _reverse_tree(network, goals, weight, self.stats)
        else:
            self.stats.cache_hits += 1
        return _tree_path(agent.current_node, self._trees[key])


def create_router(name: str) -> RoutingStrategy:
    if name == "min-cost-flow":
        from .flow import MinCostFlowRouter
        return MinCostFlowRouter()
    factories = {"dijkstra": DijkstraRouter, "astar": AStarRouter,
                 "congestion-aware": CongestionAwareRouter}
    if name not in factories:
        raise ValueError(f"unknown router: {name}")
    return factories[name]()
