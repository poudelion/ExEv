from __future__ import annotations

from collections.abc import Mapping
from heapq import heappop, heappush
from typing import Protocol

from .models import Agent, Shelter
from .network import CityNetwork


class RoutingStrategy(Protocol):
    def route(
        self,
        agent: Agent,
        network: CityNetwork,
        shelters: Mapping[str, Shelter],
        tick: int,
    ) -> list[str] | None: ...


class ShortestDistanceRouter:
    """Minimal Stage 1 router; richer baselines are introduced in Stage 2."""

    def route(
        self,
        agent: Agent,
        network: CityNetwork,
        shelters: Mapping[str, Shelter],
        tick: int,
    ) -> list[str] | None:
        destinations = {node for node, shelter in shelters.items() if shelter.available > 0}
        if agent.current_node in destinations:
            return [agent.current_node]

        queue: list[tuple[float, str]] = [(0.0, agent.current_node)]
        costs = {agent.current_node: 0.0}
        previous: dict[str, str] = {}
        destination: str | None = None
        while queue:
            cost, node = heappop(queue)
            if cost != costs[node]:
                continue
            if node in destinations:
                destination = node
                break
            for neighbor in network.neighbors(node):
                candidate = cost + network.edge(node, neighbor).distance
                if candidate < costs.get(neighbor, float("inf")):
                    costs[neighbor] = candidate
                    previous[neighbor] = node
                    heappush(queue, (candidate, neighbor))
        if destination is None:
            return None
        path = [destination]
        while path[-1] != agent.current_node:
            path.append(previous[path[-1]])
        return list(reversed(path))

