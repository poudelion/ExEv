"""QUBO models, classical solvers, and batched evacuation route assignment."""

from __future__ import annotations

import itertools
import math
import random
from collections import defaultdict
from heapq import heappop, heappush
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from math import isfinite

from .models import Agent, AgentStatus, Shelter
from .network import CityNetwork
from .routing import BaseRouter, RoutingStats, shortest_path


@dataclass(slots=True)
class QUBOModel:
    """Upper-triangular QUBO with energy constant + linear + quadratic."""

    variable_names: list[str] = field(default_factory=list)
    linear: dict[int, float] = field(default_factory=dict)
    quadratic: dict[tuple[int, int], float] = field(default_factory=dict)
    constant: float = 0.0
    _indices: dict[str, int] = field(default_factory=dict, init=False, repr=False)

    @property
    def variable_count(self) -> int:
        return len(self.variable_names)

    def add_variable(self, name: str) -> int:
        if name in self._indices:
            raise ValueError(f"duplicate QUBO variable: {name}")
        index = len(self.variable_names)
        self.variable_names.append(name)
        self._indices[name] = index
        return index

    def add_linear(self, variable: int, coefficient: float) -> None:
        self._validate_variable(variable)
        if not isfinite(coefficient):
            raise ValueError("QUBO coefficients must be finite")
        self.linear[variable] = self.linear.get(variable, 0.0) + coefficient

    def add_quadratic(self, first: int, second: int, coefficient: float) -> None:
        self._validate_variable(first)
        self._validate_variable(second)
        if not isfinite(coefficient):
            raise ValueError("QUBO coefficients must be finite")
        if first == second:
            self.add_linear(first, coefficient)
            return
        key = (first, second) if first < second else (second, first)
        self.quadratic[key] = self.quadratic.get(key, 0.0) + coefficient

    def add_squared_expression(
        self, terms: Mapping[int, float], offset: float = 0.0, weight: float = 1.0
    ) -> None:
        """Add weight * (offset + sum(coefficient * binary))**2."""
        if not isfinite(offset) or not isfinite(weight) or weight < 0:
            raise ValueError("square offset and weight must be finite; weight cannot be negative")
        items = list(terms.items())
        for variable, coefficient in items:
            self._validate_variable(variable)
            if not isfinite(coefficient):
                raise ValueError("QUBO coefficients must be finite")
            self.add_linear(
                variable, weight * (coefficient * coefficient + 2 * offset * coefficient)
            )
        for position, (first, first_coefficient) in enumerate(items):
            for second, second_coefficient in items[position + 1:]:
                self.add_quadratic(
                    first, second, 2 * weight * first_coefficient * second_coefficient
                )
        self.constant += weight * offset * offset

    def energy(self, sample: Sequence[int]) -> float:
        self.validate_sample(sample)
        result = self.constant
        result += sum(coefficient * sample[index] for index, coefficient in self.linear.items())
        result += sum(
            coefficient * sample[first] * sample[second]
            for (first, second), coefficient in self.quadratic.items()
        )
        return result

    def validate_sample(self, sample: Sequence[int]) -> None:
        if len(sample) != self.variable_count:
            raise ValueError("sample length does not match QUBO variable count")
        if any(value not in (0, 1) for value in sample):
            raise ValueError("QUBO samples must contain only zero or one")

    def _validate_variable(self, variable: int) -> None:
        if not isinstance(variable, int) or not 0 <= variable < self.variable_count:
            raise IndexError(f"unknown QUBO variable index: {variable}")


@dataclass(frozen=True, slots=True)
class QUBOSolution:
    sample: tuple[int, ...]
    energy: float
    solver: str
    iterations: int
    accepted_moves: int = 0


class ExactQUBOSolver:
    """Exhaustive reference solver for validating small QUBOs."""

    def __init__(self, max_variables: int = 24) -> None:
        if not isinstance(max_variables, int) or max_variables < 0:
            raise ValueError("max_variables must be a non-negative integer")
        self.max_variables = max_variables

    def solve(self, model: QUBOModel) -> QUBOSolution:
        if model.variable_count > self.max_variables:
            raise ValueError(
                f"exact solver supports at most {self.max_variables} variables; "
                f"received {model.variable_count}"
            )
        best_sample: tuple[int, ...] | None = None
        best_energy = float("inf")
        iterations = 0
        for sample in itertools.product((0, 1), repeat=model.variable_count):
            iterations += 1
            energy = model.energy(sample)
            if energy < best_energy:
                best_sample, best_energy = sample, energy
        assert best_sample is not None
        return QUBOSolution(best_sample, best_energy, "exact", iterations)


class SimulatedAnnealingQUBOSolver:
    """Seeded single-bit Metropolis annealing with optional feasible warm start."""

    def __init__(
        self,
        sweeps: int = 100,
        restarts: int = 3,
        seed: int = 7,
        initial_temperature: float | None = None,
        final_temperature: float | None = None,
    ) -> None:
        if not isinstance(sweeps, int) or sweeps < 1:
            raise ValueError("sweeps must be a positive integer")
        if not isinstance(restarts, int) or restarts < 1:
            raise ValueError("restarts must be a positive integer")
        if initial_temperature is not None and (
            not isfinite(initial_temperature) or initial_temperature <= 0
        ):
            raise ValueError("initial_temperature must be finite and positive")
        if final_temperature is not None and (
            not isfinite(final_temperature) or final_temperature <= 0
        ):
            raise ValueError("final_temperature must be finite and positive")
        self.sweeps = sweeps
        self.restarts = restarts
        self.seed = seed
        self.initial_temperature = initial_temperature
        self.final_temperature = final_temperature

    def solve(
        self, model: QUBOModel, initial_sample: Sequence[int] | None = None
    ) -> QUBOSolution:
        if initial_sample is not None:
            model.validate_sample(initial_sample)
        if model.variable_count == 0:
            return QUBOSolution((), model.constant, "simulated-annealing", 0)

        largest = max(
            [1.0, *(abs(value) for value in model.linear.values()),
             *(abs(value) for value in model.quadratic.values())]
        )
        initial_temperature = self.initial_temperature or largest
        final_temperature = self.final_temperature or max(1e-6, largest * 1e-3)
        if final_temperature > initial_temperature:
            raise ValueError("final_temperature cannot exceed initial_temperature")

        neighbors: list[list[tuple[int, float]]] = [
            [] for _ in range(model.variable_count)
        ]
        for (first, second), coefficient in model.quadratic.items():
            neighbors[first].append((second, coefficient))
            neighbors[second].append((first, coefficient))

        rng = random.Random(self.seed)
        best_sample = tuple(initial_sample) if initial_sample is not None else None
        best_energy = model.energy(best_sample) if best_sample is not None else float("inf")
        accepted_moves = 0
        iterations = 0
        order = list(range(model.variable_count))

        for restart in range(self.restarts):
            if restart == 0 and initial_sample is not None:
                sample = list(initial_sample)
            else:
                sample = [rng.randrange(2) for _ in range(model.variable_count)]
            current_energy = model.energy(sample)
            if current_energy < best_energy:
                best_sample, best_energy = tuple(sample), current_energy

            for sweep in range(self.sweeps):
                fraction = sweep / max(1, self.sweeps - 1)
                temperature = initial_temperature * (
                    final_temperature / initial_temperature
                ) ** fraction
                rng.shuffle(order)
                for variable in order:
                    local_field = model.linear.get(variable, 0.0) + sum(
                        coefficient * sample[other]
                        for other, coefficient in neighbors[variable]
                    )
                    delta = (1 - 2 * sample[variable]) * local_field
                    iterations += 1
                    if delta <= 0 or rng.random() < math.exp(-delta / temperature):
                        sample[variable] = 1 - sample[variable]
                        current_energy += delta
                        accepted_moves += 1
                        if current_energy < best_energy:
                            best_sample, best_energy = tuple(sample), current_energy

        assert best_sample is not None
        # Recalculate to avoid accumulated floating-point drift from delta updates.
        best_energy = model.energy(best_sample)
        return QUBOSolution(
            best_sample, best_energy, "simulated-annealing", iterations, accepted_moves
        )


@dataclass(frozen=True, slots=True)
class RouteOption:
    agent_id: int
    shelter: str | None
    path: tuple[str, ...]
    cost: float

    def __post_init__(self) -> None:
        if not isfinite(self.cost) or self.cost < 0:
            raise ValueError("route-option cost must be finite and non-negative")
        if self.shelter is not None and not self.path:
            raise ValueError("assigned route options require a path")
        if self.shelter is None and self.path:
            raise ValueError("unassigned route options cannot contain a path")

    @property
    def edges(self) -> tuple[tuple[str, str], ...]:
        return tuple(zip(self.path, self.path[1:]))


@dataclass(slots=True)
class RouteAssignmentQUBO:
    model: QUBOModel
    options: dict[int, tuple[RouteOption, ...]]
    option_variables: dict[tuple[int, int], int]
    slack_variables: dict[str, tuple[tuple[int, int], ...]]
    capacities: dict[str, int]
    congestion_weight: float
    constraint_penalty: float

    def selected_options(self, sample: Sequence[int]) -> dict[int, RouteOption]:
        self.model.validate_sample(sample)
        selected: dict[int, RouteOption] = {}
        for agent_id, options in self.options.items():
            chosen = [
                option
                for index, option in enumerate(options)
                if sample[self.option_variables[(agent_id, index)]]
            ]
            if len(chosen) == 1:
                selected[agent_id] = chosen[0]
        return selected

    def is_feasible(self, sample: Sequence[int]) -> bool:
        selected = self.selected_options(sample)
        if len(selected) != len(self.options):
            return False
        loads: dict[str, int] = defaultdict(int)
        for option in selected.values():
            if option.shelter is not None:
                loads[option.shelter] += 1
        return all(loads[shelter] <= capacity for shelter, capacity in self.capacities.items())

    def greedy_sample(self) -> tuple[int, ...]:
        """Construct a feasible warm start using marginal shared-edge cost."""
        sample = [0] * self.model.variable_count
        remaining = dict(self.capacities)
        edge_load: dict[tuple[str, str], int] = defaultdict(int)
        selected: dict[int, RouteOption] = {}
        for agent_id in sorted(self.options):
            options = self.options[agent_id]
            available = [
                option for option in options
                if option.shelter is not None and remaining.get(option.shelter, 0) > 0
            ]
            if available:
                choice = min(
                    available,
                    key=lambda option: (
                        option.cost + self.congestion_weight
                        * sum(edge_load[edge] for edge in option.edges),
                        option.shelter or "",
                    ),
                )
                remaining[choice.shelter] -= 1  # type: ignore[index]
                for edge in choice.edges:
                    edge_load[edge] += 1
            else:
                choice = next(option for option in options if option.shelter is None)
            selected[agent_id] = choice
            index = options.index(choice)
            sample[self.option_variables[(agent_id, index)]] = 1

        loads: dict[str, int] = defaultdict(int)
        for option in selected.values():
            if option.shelter is not None:
                loads[option.shelter] += 1
        for shelter, variables in self.slack_variables.items():
            target = self.capacities[shelter] - loads[shelter]
            for variable, weight in reversed(variables):
                if weight <= target:
                    sample[variable] = 1
                    target -= weight
            if target:
                raise RuntimeError("bounded binary slack failed to represent capacity")
        result = tuple(sample)
        if not self.is_feasible(result):
            raise RuntimeError("greedy QUBO warm start is infeasible")
        return result


def _bounded_binary_weights(maximum: int) -> tuple[int, ...]:
    weights: list[int] = []
    represented = 0
    next_power = 1
    while represented < maximum:
        weight = min(next_power, maximum - represented)
        weights.append(weight)
        represented += weight
        next_power *= 2
    return tuple(weights)


def build_route_assignment_qubo(
    options: Mapping[int, Sequence[RouteOption]],
    capacities: Mapping[str, int],
    *,
    congestion_weight: float = 1.0,
    constraint_penalty: float | None = None,
) -> RouteAssignmentQUBO:
    """Build a route-choice QUBO with one-hot and shelter-capacity constraints."""
    if not isfinite(congestion_weight) or congestion_weight < 0:
        raise ValueError("congestion_weight must be finite and non-negative")
    normalized_options = {agent: tuple(agent_options) for agent, agent_options in options.items()}
    if any(not agent_options for agent_options in normalized_options.values()):
        raise ValueError("every agent requires at least one route option")
    normalized_capacities = dict(capacities)
    if any(not isinstance(capacity, int) or capacity < 0
           for capacity in normalized_capacities.values()):
        raise ValueError("shelter capacities must be non-negative integers")
    for agent, agent_options in normalized_options.items():
        if any(option.agent_id != agent for option in agent_options):
            raise ValueError("route option agent IDs must match their option group")
        if sum(option.shelter is None for option in agent_options) != 1:
            raise ValueError("every agent requires exactly one unassigned option")
        if any(option.shelter is not None and option.shelter not in normalized_capacities
               for option in agent_options):
            raise ValueError("route option references an unknown shelter")

    objective_bound = sum(
        max(option.cost for option in agent_options)
        for agent_options in normalized_options.values()
    )
    if constraint_penalty is None:
        constraint_penalty = max(
            10.0,
            2.0 * (objective_bound + congestion_weight * len(normalized_options) ** 2 + 1.0),
        )
    if not isfinite(constraint_penalty) or constraint_penalty <= 0:
        raise ValueError("constraint_penalty must be finite and positive")

    model = QUBOModel()
    option_variables: dict[tuple[int, int], int] = {}
    for agent in sorted(normalized_options):
        for index, option in enumerate(normalized_options[agent]):
            variable = model.add_variable(f"route[{agent},{index}]")
            option_variables[(agent, index)] = variable
            model.add_linear(variable, option.cost)

    # Exactly one real or dummy route per agent.
    for agent, agent_options in normalized_options.items():
        terms = {
            option_variables[(agent, index)]: 1.0
            for index in range(len(agent_options))
        }
        model.add_squared_expression(terms, offset=-1.0, weight=constraint_penalty)

    # Shelter load + binary slack = capacity, which enforces load <= capacity.
    slack_variables: dict[str, tuple[tuple[int, int], ...]] = {}
    for shelter in sorted(normalized_capacities):
        terms: dict[int, float] = {}
        for agent, agent_options in normalized_options.items():
            for index, option in enumerate(agent_options):
                if option.shelter == shelter:
                    terms[option_variables[(agent, index)]] = 1.0
        slack: list[tuple[int, int]] = []
        for index, weight in enumerate(_bounded_binary_weights(normalized_capacities[shelter])):
            variable = model.add_variable(f"slack[{shelter},{index}]")
            terms[variable] = float(weight)
            slack.append((variable, weight))
        slack_variables[shelter] = tuple(slack)
        model.add_squared_expression(
            terms,
            offset=-float(normalized_capacities[shelter]),
            weight=constraint_penalty,
        )

    # Pairwise overlap is a soft proxy for simultaneous congestion. Road capacity
    # is deliberately not treated as a lifetime total-flow constraint.
    users_by_edge: dict[tuple[str, str], list[tuple[int, int]]] = defaultdict(list)
    for agent, agent_options in normalized_options.items():
        for index, option in enumerate(agent_options):
            variable = option_variables[(agent, index)]
            for edge in option.edges:
                users_by_edge[edge].append((agent, variable))
    for users in users_by_edge.values():
        for position, (first_agent, first_variable) in enumerate(users):
            for second_agent, second_variable in users[position + 1:]:
                if first_agent != second_agent:
                    model.add_quadratic(first_variable, second_variable, congestion_weight)

    return RouteAssignmentQUBO(
        model,
        normalized_options,
        option_variables,
        slack_variables,
        normalized_capacities,
        congestion_weight,
        constraint_penalty,
    )

def _candidate_paths(
    network: CityNetwork,
    start: str,
    target: str,
    limit: int,
    edge_weight: Callable[[str, str], float],
    stats: RoutingStats,
) -> tuple[tuple[str, ...], ...]:
    """Return up to k lowest-cost simple paths using Yen's algorithm."""

    def shortest(
        source: str,
        banned_nodes: set[str],
        banned_edges: set[tuple[str, str]],
    ) -> tuple[str, ...] | None:
        stats.searches += 1
        queue: list[tuple[float, tuple[str, ...]]] = [(0.0, (source,))]
        best = {source: 0.0}
        while queue:
            cost, path = heappop(queue)
            node = path[-1]
            if cost != best.get(node):
                continue
            stats.expanded_nodes += 1
            if node == target:
                return path
            for neighbor in sorted(network.neighbors(node)):
                if neighbor in banned_nodes or (node, neighbor) in banned_edges:
                    continue
                candidate = cost + edge_weight(node, neighbor)
                if candidate < best.get(neighbor, float("inf")):
                    best[neighbor] = candidate
                    heappush(queue, (candidate, (*path, neighbor)))
        return None

    first = shortest(start, set(), set())
    if first is None:
        return ()
    results = [first]
    candidate_heap: list[tuple[float, tuple[str, ...]]] = []
    candidates: set[tuple[str, ...]] = set()

    while len(results) < limit:
        previous = results[-1]
        for spur_index in range(len(previous) - 1):
            root = previous[:spur_index + 1]
            banned_edges = {
                (path[spur_index], path[spur_index + 1])
                for path in results
                if len(path) > spur_index + 1
                and path[:spur_index + 1] == root
            }
            spur = shortest(root[-1], set(root[:-1]), banned_edges)
            if spur is None:
                continue
            candidate_path = (*root[:-1], *spur)
            if candidate_path in candidates or candidate_path in results:
                continue
            cost = sum(
                edge_weight(source, destination)
                for source, destination in zip(candidate_path, candidate_path[1:])
            )
            heappush(candidate_heap, (cost, candidate_path))
            candidates.add(candidate_path)
        if not candidate_heap:
            break
        _, selected = heappop(candidate_heap)
        candidates.remove(selected)
        results.append(selected)
    return tuple(results)


class QUBOSimulatedAnnealingRouter(BaseRouter):
    """Hybrid batched route assignment solved with classical simulated annealing."""

    name = "qubo-sa"

    def __init__(
        self,
        batch_size: int = 8,
        sweeps: int = 100,
        restarts: int = 3,
        seed: int = 7,
        routes_per_shelter: int = 3,
        congestion_weight: float = 1.0,
        hazard_weight: float = 1.0,
    ) -> None:
        super().__init__()
        if not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        if not isinstance(routes_per_shelter, int) or routes_per_shelter < 1:
            raise ValueError("routes_per_shelter must be a positive integer")
        if not isfinite(congestion_weight) or congestion_weight < 0:
            raise ValueError("congestion_weight must be finite and non-negative")
        if not isfinite(hazard_weight) or hazard_weight < 0:
            raise ValueError("hazard_weight must be finite and non-negative")
        self.batch_size = batch_size
        self.sweeps = sweeps
        self.restarts = restarts
        self.seed = seed
        self.routes_per_shelter = routes_per_shelter
        self.congestion_weight = congestion_weight
        self.hazard_weight = hazard_weight
        self.assignment: dict[int, str] = {}
        self._paths: dict[tuple[int, str, str], tuple[str, ...] | None] = {}
        self._topology_version: int | None = None
        self.planned_flow = 0
        self.planned_cost = 0.0
        self.qubo_batches = 0
        self.qubo_total_variables = 0
        self.qubo_max_variables = 0
        self.qubo_energy = 0.0
        self.qubo_annealing_iterations = 0
        self.qubo_accepted_moves = 0
        self.qubo_fallback_batches = 0

    def _route_cost(self, agent: Agent, network: CityNetwork, path: Sequence[str]) -> float:
        distance = 0.0
        exposure = 0.0
        for source, target in zip(path, path[1:]):
            edge = network.edge(source, target)
            travel_time = edge.distance / min(agent.speed, edge.speed_limit)
            distance += edge.distance
            exposure += network.edge_hazard(source, target) * travel_time
            exposure += network.node_hazard(target)
        return distance + self.hazard_weight * exposure

    def _cache_path(
        self, agent_id: int, path: Sequence[str] | None, destination: str
    ) -> None:
        if path is None:
            return
        for index, node in enumerate(path):
            self._paths[(agent_id, node, destination)] = tuple(path[index:])

    def prepare(
        self, agents: Sequence[Agent], network: CityNetwork, shelters: Mapping[str, Shelter]
    ) -> None:
        super().prepare(agents, network, shelters)
        self.assignment.clear()
        self._paths.clear()
        self._topology_version = network.topology_version
        self.planned_flow = 0
        self.planned_cost = 0.0
        self.qubo_batches = 0
        self.qubo_total_variables = 0
        self.qubo_max_variables = 0
        self.qubo_energy = 0.0
        self.qubo_annealing_iterations = 0
        self.qubo_accepted_moves = 0
        self.qubo_fallback_batches = 0
        remaining = {node: shelter.available for node, shelter in shelters.items()}
        waiting = sorted(
            (agent for agent in agents if agent.status == AgentStatus.WAITING),
            key=lambda agent: agent.id,
        )
        path_cache: dict[tuple[str, str, float], tuple[tuple[str, ...], ...]] = {}

        for batch_index, start in enumerate(range(0, len(waiting), self.batch_size)):
            batch = waiting[start:start + self.batch_size]
            real_options: dict[int, list[RouteOption]] = {}
            largest_cost = 1.0
            for agent in batch:
                choices: list[RouteOption] = []
                for destination in sorted(shelters):
                    if remaining[destination] <= 0:
                        continue
                    key = (agent.current_node, destination, agent.speed)
                    if key not in path_cache:
                        path_cache[key] = _candidate_paths(
                            network,
                            agent.current_node,
                            destination,
                            self.routes_per_shelter,
                            lambda source, target: self._route_cost(
                                agent, network, (source, target)
                            ),
                            self.stats,
                        )
                    else:
                        self.stats.cache_hits += 1
                    for path in path_cache[key]:
                        cost = self._route_cost(agent, network, path)
                        largest_cost = max(largest_cost, cost)
                        choices.append(RouteOption(agent.id, destination, path, cost))
                real_options[agent.id] = choices

            unassigned_cost = 10.0 * (len(batch) + 1) * largest_cost
            options = {
                agent.id: (
                    *real_options[agent.id],
                    RouteOption(agent.id, None, (), unassigned_cost),
                )
                for agent in batch
            }
            problem = build_route_assignment_qubo(
                options,
                remaining,
                congestion_weight=self.congestion_weight,
            )
            warm_start = problem.greedy_sample()
            solver = SimulatedAnnealingQUBOSolver(
                sweeps=self.sweeps,
                restarts=self.restarts,
                seed=self.seed + batch_index,
            )
            solution = solver.solve(problem.model, warm_start)
            if problem.is_feasible(solution.sample):
                sample = solution.sample
                self.qubo_energy += solution.energy
            else:
                sample = warm_start
                self.qubo_energy += problem.model.energy(warm_start)
                self.qubo_fallback_batches += 1

            self.qubo_batches += 1
            self.qubo_total_variables += problem.model.variable_count
            self.qubo_max_variables = max(self.qubo_max_variables, problem.model.variable_count)
            self.qubo_annealing_iterations += solution.iterations
            self.qubo_accepted_moves += solution.accepted_moves
            for agent_id, option in problem.selected_options(sample).items():
                if option.shelter is None:
                    continue
                self.assignment[agent_id] = option.shelter
                remaining[option.shelter] -= 1
                self.planned_flow += 1
                self.planned_cost += option.cost
                self._cache_path(agent_id, option.path, option.shelter)

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
        key = (agent.id, agent.current_node, destination)
        if key in self._paths:
            self.stats.cache_hits += 1
            cached = self._paths[key]
            return list(cached) if cached is not None else None
        path = shortest_path(network, agent.current_node, {destination}, stats=self.stats)
        self._cache_path(agent.id, path, destination)
        return path
