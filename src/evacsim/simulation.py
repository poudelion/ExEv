from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from statistics import mean

from .models import Agent, AgentStatus, Shelter
from .network import CityNetwork
from .routing import RoutingStrategy, ShortestDistanceRouter


@dataclass(frozen=True, slots=True)
class SimulationConfig:
    max_ticks: int = 10_000
    reroute_wait_threshold: int = 10


@dataclass(frozen=True, slots=True)
class SimulationResult:
    elapsed_ticks: int
    total_agents: int
    evacuated: int
    stranded: int
    unfinished: int
    mean_evacuation_time: float | None
    max_evacuation_time: int | None
    mean_distance_traveled: float
    mean_waiting_time: float

    @property
    def evacuation_rate(self) -> float:
        return self.evacuated / self.total_agents if self.total_agents else 1.0


class EvacuationSimulation:
    def __init__(
        self,
        network: CityNetwork,
        agents: list[Agent],
        shelters: list[Shelter],
        router: RoutingStrategy | None = None,
        config: SimulationConfig | None = None,
    ) -> None:
        if len({agent.id for agent in agents}) != len(agents):
            raise ValueError("agent IDs must be unique")
        self.network = network
        self.agents = agents
        self.shelters = {shelter.node: shelter for shelter in shelters}
        self.router = router or ShortestDistanceRouter()
        self.config = config or SimulationConfig()
        self.tick = 0

    def run(self) -> SimulationResult:
        while self.tick < self.config.max_ticks and self._has_active_agents():
            self.step()
        return self.result()

    def step(self) -> None:
        self._advance_travelers()
        self._process_nodes()
        self.tick += 1

    def _advance_travelers(self) -> None:
        for agent in self.agents:
            if agent.status != AgentStatus.TRAVELING or agent.edge is None:
                continue
            source, target = agent.edge
            distance = self.network.edge(source, target).distance
            delta = min(
                distance - agent.edge_progress,
                self.network.travel_rate(source, target, agent.speed),
            )
            agent.edge_progress += delta
            agent.distance_traveled += delta
            if agent.edge_progress + 1e-12 >= distance:
                self.network.leave(source, target)
                agent.current_node = target
                agent.route_index += 1
                agent.edge = None
                agent.edge_progress = 0.0
                agent.status = AgentStatus.WAITING

    def _process_nodes(self) -> None:
        waiting = [agent for agent in self.agents if agent.status == AgentStatus.WAITING]
        # Stable rotation avoids permanently favoring low IDs at bottlenecks.
        waiting.sort(key=lambda agent: ((agent.id - self.tick) % max(1, len(self.agents))))
        for agent in waiting:
            shelter = self.shelters.get(agent.current_node)
            if shelter and shelter.admit():
                agent.status = AgentStatus.EVACUATED
                agent.evacuated_at = self.tick
                continue

            route_invalid = (
                not agent.route
                or agent.route_index >= len(agent.route) - 1
                or agent.route[agent.route_index] != agent.current_node
                or self.network.edge(
                    agent.route[agent.route_index], agent.route[agent.route_index + 1]
                ).blocked
            )
            if route_invalid or agent.waiting_time >= self.config.reroute_wait_threshold:
                route = self.router.route(agent, self.network, self.shelters, self.tick)
                if route is None or len(route) < 2:
                    agent.status = AgentStatus.STRANDED
                    continue
                agent.route = route
                agent.route_index = 0
                agent.waiting_time = 0

            source = agent.current_node
            target = agent.route[agent.route_index + 1]
            if self.network.can_enter(source, target):
                self.network.enter(source, target)
                agent.edge = (source, target)
                agent.status = AgentStatus.TRAVELING
                agent.started_at = self.tick if agent.started_at is None else agent.started_at
                agent.waiting_time = 0
            else:
                agent.waiting_time += 1
                agent.total_waiting_time += 1

    def _has_active_agents(self) -> bool:
        terminal = {AgentStatus.EVACUATED, AgentStatus.STRANDED}
        return any(agent.status not in terminal for agent in self.agents)

    def result(self) -> SimulationResult:
        counts = Counter(agent.status for agent in self.agents)
        times = [agent.evacuated_at for agent in self.agents if agent.evacuated_at is not None]
        return SimulationResult(
            elapsed_ticks=self.tick,
            total_agents=len(self.agents),
            evacuated=counts[AgentStatus.EVACUATED],
            stranded=counts[AgentStatus.STRANDED],
            unfinished=counts[AgentStatus.WAITING] + counts[AgentStatus.TRAVELING],
            mean_evacuation_time=mean(times) if times else None,
            max_evacuation_time=max(times) if times else None,
            mean_distance_traveled=mean(a.distance_traveled for a in self.agents) if self.agents else 0.0,
            mean_waiting_time=mean(a.total_waiting_time for a in self.agents) if self.agents else 0.0,
        )
