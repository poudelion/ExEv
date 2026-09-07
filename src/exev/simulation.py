from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from statistics import mean
from time import perf_counter

from .disasters import DisasterSchedule
from .models import Agent, AgentStatus, Shelter
from .network import CityNetwork
from .routing import RoutingStats, RoutingStrategy, ShortestDistanceRouter

def _percentile(values: list[float | int], probability: float) -> float | None:
    """Linearly interpolated percentile for non-empty numeric samples."""
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


def _gini(values: list[float | int]) -> float:
    """Gini coefficient for non-negative values; zero means no dispersion."""
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    total = sum(ordered)
    if total == 0:
        return 0.0
    weighted = sum(index * value for index, value in enumerate(ordered, start=1))
    return (2 * weighted) / (len(ordered) * total) - (len(ordered) + 1) / len(ordered)


@dataclass(frozen=True, slots=True)
class SimulationConfig:
    max_ticks: int = 10_000
    reroute_wait_threshold: int = 10

    def __post_init__(self) -> None:
        if not isinstance(self.max_ticks, int) or self.max_ticks < 1:
            raise ValueError("max_ticks must be a positive integer")
        if not isinstance(self.reroute_wait_threshold, int) or self.reroute_wait_threshold < 1:
            raise ValueError("reroute_wait_threshold must be a positive integer")


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
    algorithm: str = "custom"
    routing_seconds: float = 0.0
    preparation_seconds: float = 0.0
    wall_seconds: float = 0.0
    route_calls: int = 0
    route_searches: int = 0
    cache_hits: int = 0
    expanded_nodes: int = 0
    peak_congestion: float = 0.0
    total_hazard_exposure: float = 0.0
    mean_hazard_exposure: float = 0.0
    max_hazard_exposure: float = 0.0
    events_processed: int = 0
    qubo_batches: int = 0
    qubo_total_variables: int = 0
    qubo_max_variables: int = 0
    qubo_energy: float = 0.0
    qubo_annealing_iterations: int = 0
    qubo_accepted_moves: int = 0
    qubo_fallback_batches: int = 0
    qubo_planned_flow: int = 0
    qubo_planned_cost: float = 0.0
    p50_evacuation_time: float | None = None
    p90_evacuation_time: float | None = None
    p95_evacuation_time: float | None = None
    total_waiting_time: int = 0
    p95_waiting_time: float = 0.0
    total_distance_traveled: float = 0.0
    p95_hazard_exposure: float = 0.0
    completion_time_gini: float = 0.0
    hazard_exposure_gini: float = 0.0
    slow_agent_mean_evacuation_time: float | None = None
    fast_agent_mean_evacuation_time: float | None = None
    speed_group_evacuation_gap: float | None = None
    total_edge_occupancy_ticks: int = 0
    full_edge_ticks: int = 0
    mean_edge_utilization: float = 0.0

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
        disaster_schedule: DisasterSchedule | None = None,
    ) -> None:
        if len({agent.id for agent in agents}) != len(agents):
            raise ValueError("agent IDs must be unique")
        if len({shelter.node for shelter in shelters}) != len(shelters):
            raise ValueError("shelter nodes must be unique")
        nodes = network.nodes
        if any(agent.current_node not in nodes for agent in agents):
            raise ValueError("all agent origins must be network nodes")
        if any(shelter.node not in nodes for shelter in shelters):
            raise ValueError("all shelters must be network nodes")
        if any(agent.status != AgentStatus.WAITING or agent.edge is not None for agent in agents):
            raise ValueError("start each simulation with fresh waiting agents")
        if any(network.occupancy_snapshot().values()):
            raise ValueError("start each simulation with an empty road network")
        self.network = network
        self.agents = agents
        self.shelters = {shelter.node: shelter for shelter in shelters}
        self.router = router or ShortestDistanceRouter()
        self.config = config or SimulationConfig()
        self.disaster_schedule = disaster_schedule or DisasterSchedule()
        self.disaster_schedule.validate(network, self.shelters)
        self.tick = 0
        self._ranks = {agent.id: rank for rank, agent in enumerate(sorted(agents, key=lambda a: a.id))}
        self._prepared = False
        self._wall_seconds = 0.0
        self._run_started: float | None = None
        self._preparation_seconds = 0.0
        self._routing_seconds = 0.0
        self._route_calls = 0
        self._peak_congestion = 0.0
        self._edge_utilization_sum = 0.0
        self._edge_utilization_observations = 0
        self._total_edge_occupancy_ticks = 0
        self._full_edge_ticks = 0

    def run(self, progress: Callable[[EvacuationSimulation], None] | None = None,
            progress_interval: int = 50) -> SimulationResult:
        if not isinstance(progress_interval, int) or progress_interval < 1:
            raise ValueError("progress_interval must be a positive integer")
        self._run_started = perf_counter()
        try:
            if progress is not None:
                progress(self)
            while self.tick < self.config.max_ticks and self._has_active_agents():
                self.step()
                if progress is not None and self.tick % progress_interval == 0:
                    progress(self)
            if progress is not None:
                progress(self)
        finally:
            self._wall_seconds += perf_counter() - self._run_started
            self._run_started = None
        return self.result()

    def _prepare(self) -> None:
        if not self._prepared:
            start = perf_counter()
            prepare = getattr(self.router, "prepare", None)
            if prepare is not None:
                prepare(self.agents, self.network, self.shelters)
            self._preparation_seconds += perf_counter() - start
            self._prepared = True

    def step(self) -> None:
        self.disaster_schedule.apply_tick(self.tick, self.network, self.shelters)
        self._prepare()
        self._accumulate_hazard_exposure()
        self._advance_travelers()
        begin_tick = getattr(self.router, "begin_tick", None)
        if begin_tick is not None:
            start = perf_counter()
            begin_tick(self.network, self.shelters, self.tick)
            self._routing_seconds += perf_counter() - start
        self._process_nodes()
        self._record_network_load()
        self.tick += 1

    def _accumulate_hazard_exposure(self) -> None:
        for agent in self.agents:
            if agent.status == AgentStatus.TRAVELING and agent.edge is not None:
                agent.hazard_exposure += self.network.edge_hazard(*agent.edge)
            elif agent.status == AgentStatus.WAITING:
                agent.hazard_exposure += self.network.node_hazard(agent.current_node)

    def _advance_travelers(self) -> None:
        # Every traveler sees the same occupancy for this movement phase.
        occupancy = self.network.occupancy_snapshot()
        for agent in self.agents:
            if agent.status != AgentStatus.TRAVELING or agent.edge is None:
                continue
            source, target = agent.edge
            distance = self.network.edge(source, target).distance
            delta = min(
                distance - agent.edge_progress,
                self.network.travel_rate(source, target, agent.speed,
                                         occupancy=occupancy.get((source, target), 0)),
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
        # Rotate ranks, so noncontiguous IDs cannot collide modulo population.
        waiting.sort(key=lambda agent: ((self._ranks[agent.id] - self.tick) % max(1, len(self.agents))))
        can_admit = getattr(self.router, "can_admit", lambda agent, node: True)
        for agent in waiting:
            shelter = self.shelters.get(agent.current_node)
            if shelter and can_admit(agent, agent.current_node) and shelter.admit():
                agent.status = AgentStatus.EVACUATED
                agent.evacuated_at = self.tick
                continue

            route_invalid = (
                not agent.route
                or agent.route_index >= len(agent.route) - 1
                or agent.route[agent.route_index] != agent.current_node
                or agent.route[-1] not in self.shelters
                or self.shelters[agent.route[-1]].available == 0
                or self.network.edge(
                    agent.route[agent.route_index], agent.route[agent.route_index + 1]
                ).blocked
            )
            replan_at_node = getattr(self.router, "replan_at_nodes", False) and agent.route_index > 0
            if route_invalid or replan_at_node or agent.waiting_time >= self.config.reroute_wait_threshold:
                start = perf_counter()
                route = self.router.route(agent, self.network, self.shelters, self.tick)
                self._routing_seconds += perf_counter() - start
                self._route_calls += 1
                stats = getattr(self.router, "stats", None)
                if stats is not None:
                    stats.route_calls += 1
                if route is None or len(route) < 2:
                    if self.disaster_schedule.has_pending:
                        agent.waiting_time += 1
                        agent.total_waiting_time += 1
                    else:
                        agent.status = AgentStatus.STRANDED
                    continue
                if route[0] != agent.current_node or route[-1] not in self.shelters:
                    raise ValueError("router returned a path with invalid endpoints")
                if any(v not in self.network.neighbors(u) for u, v in zip(route, route[1:])):
                    raise ValueError("router returned a path through missing or blocked roads")
                agent.route = route
                agent.route_index = 0
                agent.waiting_time = 0

            source = agent.current_node
            target = agent.route[agent.route_index + 1]
            if self.network.can_enter(source, target):
                self.network.enter(source, target)
                self._peak_congestion = max(self._peak_congestion, self.network.congestion(source, target))
                agent.edge = (source, target)
                agent.status = AgentStatus.TRAVELING
                agent.started_at = self.tick if agent.started_at is None else agent.started_at
                agent.waiting_time = 0
            else:
                agent.waiting_time += 1
                agent.total_waiting_time += 1

    def _record_network_load(self) -> None:
        occupancy = self.network.occupancy_snapshot()
        for edge in self.network.edges:
            count = occupancy.get((edge.source, edge.target), 0)
            self._total_edge_occupancy_ticks += count
            self._edge_utilization_sum += count / edge.capacity
            self._edge_utilization_observations += 1
            if count >= edge.capacity:
                self._full_edge_ticks += 1

    def _has_active_agents(self) -> bool:
        terminal = {AgentStatus.EVACUATED, AgentStatus.STRANDED}
        return any(agent.status not in terminal for agent in self.agents)

    def result(self) -> SimulationResult:
        counts = Counter(agent.status for agent in self.agents)
        times = [agent.evacuated_at for agent in self.agents if agent.evacuated_at is not None]
        stats = getattr(self.router, "stats", RoutingStats())
        exposures = [agent.hazard_exposure for agent in self.agents]
        waiting_times = [agent.total_waiting_time for agent in self.agents]
        distances = [agent.distance_traveled for agent in self.agents]
        completion_times = [
            agent.evacuated_at if agent.evacuated_at is not None else self.tick
            for agent in self.agents
        ]
        if self.agents:
            slowest_speed = min(agent.speed for agent in self.agents)
            fastest_speed = max(agent.speed for agent in self.agents)
            slow_times = [
                agent.evacuated_at
                for agent in self.agents
                if agent.speed == slowest_speed and agent.evacuated_at is not None
            ]
            fast_times = [
                agent.evacuated_at
                for agent in self.agents
                if agent.speed == fastest_speed and agent.evacuated_at is not None
            ]
        else:
            slow_times = []
            fast_times = []
        slow_mean = mean(slow_times) if slow_times else None
        fast_mean = mean(fast_times) if fast_times else None
        speed_gap = (
            slow_mean - fast_mean
            if slow_mean is not None and fast_mean is not None
            else None
        )
        wall_seconds = self._wall_seconds
        if self._run_started is not None:
            wall_seconds += perf_counter() - self._run_started
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
            algorithm=getattr(self.router, "name", type(self.router).__name__),
            routing_seconds=self._routing_seconds,
            preparation_seconds=self._preparation_seconds,
            wall_seconds=wall_seconds,
            route_calls=self._route_calls,
            route_searches=stats.searches,
            cache_hits=stats.cache_hits,
            expanded_nodes=stats.expanded_nodes,
            peak_congestion=self._peak_congestion,
            total_hazard_exposure=sum(exposures),
            mean_hazard_exposure=mean(exposures) if exposures else 0.0,
            max_hazard_exposure=max(exposures, default=0.0),
            events_processed=self.disaster_schedule.events_processed,
            qubo_batches=getattr(self.router, "qubo_batches", 0),
            qubo_total_variables=getattr(self.router, "qubo_total_variables", 0),
            qubo_max_variables=getattr(self.router, "qubo_max_variables", 0),
            qubo_energy=getattr(self.router, "qubo_energy", 0.0),
            qubo_annealing_iterations=getattr(self.router, "qubo_annealing_iterations", 0),
            qubo_accepted_moves=getattr(self.router, "qubo_accepted_moves", 0),
            qubo_fallback_batches=getattr(self.router, "qubo_fallback_batches", 0),
            qubo_planned_flow=getattr(self.router, "planned_flow", 0),
            qubo_planned_cost=getattr(self.router, "planned_cost", 0.0),
            p50_evacuation_time=_percentile(times, 0.50),
            p90_evacuation_time=_percentile(times, 0.90),
            p95_evacuation_time=_percentile(times, 0.95),
            total_waiting_time=sum(waiting_times),
            p95_waiting_time=_percentile(waiting_times, 0.95) or 0.0,
            total_distance_traveled=sum(distances),
            p95_hazard_exposure=_percentile(exposures, 0.95) or 0.0,
            completion_time_gini=_gini(completion_times),
            hazard_exposure_gini=_gini(exposures),
            slow_agent_mean_evacuation_time=slow_mean,
            fast_agent_mean_evacuation_time=fast_mean,
            speed_group_evacuation_gap=speed_gap,
            total_edge_occupancy_ticks=self._total_edge_occupancy_ticks,
            full_edge_ticks=self._full_edge_ticks,
            mean_edge_utilization=(
                self._edge_utilization_sum / self._edge_utilization_observations
                if self._edge_utilization_observations
                else 0.0
            ),
        )
