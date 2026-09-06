"""Static minimum-distance shelter assignment with integral maximum flow.

Road capacity is simultaneous occupancy in the simulator, not a budget on the
number of evacuees who may ever use a road. This transportation model constrains
shelter space; the simulation enforces road occupancy. It optimizes assigned
population first and total planned distance second, not evacuation time.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from heapq import heappop, heappush
from math import inf

from .models import Agent, AgentStatus, Shelter
from .network import CityNetwork
from .routing import BaseRouter, shortest_path


@dataclass(slots=True)
class _ResidualEdge:
    target: int
    reverse: int
    capacity: int
    cost: float


def _add_arc(
    graph: list[list[_ResidualEdge]], source: int, target: int, capacity: int, cost: float
) -> _ResidualEdge:
    forward = _ResidualEdge(target, len(graph[target]), capacity, cost)
    backward = _ResidualEdge(source, len(graph[source]), 0, -cost)
    graph[source].append(forward)
    graph[target].append(backward)
    return forward


def _min_cost_max_flow(graph: list[list[_ResidualEdge]], source: int, sink: int) -> int:
    """Successive shortest augmenting paths, including residual reassignment.

    Initial forward costs are nonnegative, so zero potentials are feasible.
    Integer capacities and augmentations guarantee integral assignments; costs
    retain the graph's floating-point distance units.
    """
    potentials = [0.0] * len(graph)
    total_flow = 0
    while True:
        distances = [inf] * len(graph)
        predecessors: list[tuple[int, int] | None] = [None] * len(graph)
        distances[source] = 0.0
        queue = [(0.0, source)]
        while queue:
            distance, node = heappop(queue)
            if distance != distances[node]:
                continue
            for index, edge in enumerate(graph[node]):
                if edge.capacity <= 0:
                    continue
                # Potentials make residual costs nonnegative; clamp roundoff.
                reduced_cost = max(0.0, edge.cost + potentials[node] - potentials[edge.target])
                candidate = distance + reduced_cost
                if candidate < distances[edge.target]:
                    distances[edge.target] = candidate
                    predecessors[edge.target] = (node, index)
                    heappush(queue, (candidate, edge.target))
        if predecessors[sink] is None:
            return total_flow
        for node, distance in enumerate(distances):
            if distance < inf:
                potentials[node] += distance

        amount: int | None = None
        node = sink
        while node != source:
            previous = predecessors[node]
            assert previous is not None
            origin, index = previous
            capacity = graph[origin][index].capacity
            amount = capacity if amount is None else min(amount, capacity)
            node = origin
        assert amount is not None and amount > 0
        node = sink
        while node != source:
            previous = predecessors[node]
            assert previous is not None
            origin, index = previous
            edge = graph[origin][index]
            edge.capacity -= amount
            graph[node][edge.reverse].capacity += amount
            node = origin
        total_flow += amount


class MinCostFlowRouter(BaseRouter):
    """Globally assign available shelter slots before the first movement tick.

    Assignments reserve shelter space without changing actual occupancy.
    Admission is allowed only at an agent's assigned shelter, even if its route
    passes through another shelter. Closures trigger new paths to existing
    destinations; revising shelter reservations requires a new plan and is
    outside this static baseline.
    """

    name = "min-cost-flow"

    def __init__(self) -> None:
        super().__init__()
        self.assignment: dict[int, str] = {}
        self.planned_flow = 0
        self.planned_distance = 0.0
        self._paths: dict[tuple[str, str], tuple[str, ...] | None] = {}
        self._topology_version: int | None = None

    def _cache_path(self, origin: str, destination: str, path: list[str] | None) -> None:
        self._paths[(origin, destination)] = tuple(path) if path is not None else None
        if path is not None:
            # Every suffix of a shortest path is itself a shortest path.
            for index, node in enumerate(path):
                self._paths[(node, destination)] = tuple(path[index:])

    def prepare(
        self, agents: Sequence[Agent], network: CityNetwork, shelters: Mapping[str, Shelter]
    ) -> None:
        super().prepare(agents, network, shelters)
        self.assignment.clear()
        self.planned_flow = 0
        self.planned_distance = 0.0
        self._paths.clear()
        self._topology_version = network.topology_version
        groups: dict[str, list[Agent]] = defaultdict(list)
        for agent in agents:
            if agent.status == AgentStatus.WAITING:
                groups[agent.current_node].append(agent)
        origins = sorted(groups)
        destinations = sorted(node for node, shelter in shelters.items() if shelter.available > 0)
        if not origins or not destinations:
            return

        source = 0
        origin_ids = {node: index + 1 for index, node in enumerate(origins)}
        target_ids = {node: len(origins) + index + 1 for index, node in enumerate(destinations)}
        sink = len(origins) + len(destinations) + 1
        residual: list[list[_ResidualEdge]] = [[] for _ in range(sink + 1)]
        assignments: dict[tuple[str, str], tuple[_ResidualEdge, int, float]] = {}
        for origin in origins:
            population = len(groups[origin])
            _add_arc(residual, source, origin_ids[origin], population, 0.0)
            for destination in destinations:
                key = (origin, destination)
                if key in self._paths:
                    cached = self._paths[key]
                    path = list(cached) if cached is not None else None
                    self.stats.cache_hits += 1
                else:
                    path = shortest_path(network, origin, {destination}, stats=self.stats)
                    self._cache_path(origin, destination, path)
                if path is None:
                    continue
                distance = sum(network.edge(a, b).distance for a, b in zip(path, path[1:]))
                edge = _add_arc(
                    residual, origin_ids[origin], target_ids[destination], population, distance
                )
                assignments[key] = (edge, population, distance)
        for destination in destinations:
            _add_arc(residual, target_ids[destination], sink, shelters[destination].available, 0.0)
        self.planned_flow = _min_cost_max_flow(residual, source, sink)

        for origin in origins:
            agents_at_origin = iter(sorted(groups[origin], key=lambda agent: agent.id))
            for destination in destinations:
                entry = assignments.get((origin, destination))
                if entry is None:
                    continue
                edge, initial_capacity, distance = entry
                count = initial_capacity - edge.capacity
                self.planned_distance += count * distance
                for _ in range(count):
                    self.assignment[next(agents_at_origin).id] = destination

    def can_admit(self, agent: Agent, node: str) -> bool:
        return self.assignment.get(agent.id) == node

    def route(
        self, agent: Agent, network: CityNetwork, shelters: Mapping[str, Shelter], tick: int
    ) -> list[str] | None:
        destination = self.assignment.get(agent.id)
        if destination is None or destination not in shelters or shelters[destination].available <= 0:
            return None
        if self._topology_version != network.topology_version:
            self._paths.clear()
            self._topology_version = network.topology_version
        key = (agent.current_node, destination)
        if key in self._paths:
            self.stats.cache_hits += 1
            path = self._paths[key]
            return list(path) if path is not None else None
        path = shortest_path(network, agent.current_node, {destination}, stats=self.stats)
        self._cache_path(agent.current_node, destination, path)
        return list(path) if path is not None else None
