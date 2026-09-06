from __future__ import annotations

import itertools
import random
import unittest
from collections import Counter

from exev.flow import MinCostFlowRouter
from exev.models import Agent, Shelter
from exev.network import CityNetwork
from exev.simulation import EvacuationSimulation


def transport_network(costs: list[list[int | None]]) -> CityNetwork:
    network = CityNetwork()
    for origin, distances in enumerate(costs):
        for destination, distance in enumerate(distances):
            if distance is not None:
                network.add_edge(
                    f"o{origin}", f"s{destination}", distance, capacity=1, bidirectional=False
                )
    return network


class MinCostFlowTests(unittest.TestCase):
    def test_agent_initially_at_shelter_is_admitted_to_its_assignment(self):
        network = CityNetwork()
        network.add_edge("o", "s", 1, 3, bidirectional=False)
        agents = [Agent(0, "s"), Agent(1, "o")]
        router = MinCostFlowRouter()
        result = EvacuationSimulation(network, agents, [Shelter("s", 2)], router).run()
        self.assertEqual(router.planned_distance, 1)
        self.assertEqual(agents[0].evacuated_at, 0)
        self.assertEqual(result.evacuated, 2)

    def test_residual_reassignment_escapes_greedy_trap(self) -> None:
        # Greedily sending o0 to s0 costs 1 + 100. Residual reassignment costs 4.
        network = transport_network([[1, 2], [2, 100]])
        agents = [Agent(0, "o0"), Agent(1, "o1")]
        shelters = {"s0": Shelter("s0", 1), "s1": Shelter("s1", 1)}
        router = MinCostFlowRouter()
        router.prepare(agents, network, shelters)
        self.assertEqual(router.assignment, {0: "s1", 1: "s0"})
        self.assertEqual(router.planned_flow, 2)
        self.assertEqual(router.planned_distance, 4)
        self.assertEqual([shelter.occupants for shelter in shelters.values()], [0, 0])

    def test_maximum_flow_takes_priority_over_distance(self) -> None:
        # o1 only reaches s0: serving both requires moving o0's reservation.
        network = transport_network([[1, 100], [2, None]])
        agents = [Agent(0, "o0"), Agent(1, "o1")]
        shelters = {"s0": Shelter("s0", 1), "s1": Shelter("s1", 1)}
        router = MinCostFlowRouter()
        router.prepare(agents, network, shelters)
        self.assertEqual(router.assignment, {0: "s1", 1: "s0"})
        self.assertEqual(router.planned_flow, 2)
        self.assertEqual(router.planned_distance, 102)

    def test_partial_capacity_and_unreachable_agents(self) -> None:
        network = transport_network([[1], [None]])
        agents = [Agent(0, "o0"), Agent(1, "o0"), Agent(2, "o1")]
        shelters = {"s0": Shelter("s0", 1)}
        router = MinCostFlowRouter()
        router.prepare(agents, network, shelters)
        self.assertEqual(router.assignment, {0: "s0"})
        self.assertEqual(router.planned_flow, 1)
        self.assertIsNone(router.route(agents[1], network, shelters, 0))
        self.assertIsNone(router.route(agents[2], network, shelters, 0))
        self.assertFalse(router.can_admit(agents[1], "s0"))

    def test_preoccupied_shelters_and_shared_origin(self) -> None:
        network = transport_network([[1, 2]])
        agents = [Agent(index, "o0") for index in range(5)]
        shelters = {"s0": Shelter("s0", 3, occupants=2), "s1": Shelter("s1", 4)}
        router = MinCostFlowRouter()
        router.prepare(agents, network, shelters)
        self.assertEqual(Counter(router.assignment.values()), {"s0": 1, "s1": 4})
        self.assertEqual(router.planned_distance, 9)
        self.assertEqual(shelters["s0"].occupants, 2)
        # Capacity 1 roads carry multiple evacuees sequentially.
        result = EvacuationSimulation(network, agents, list(shelters.values()), router=router).run()
        self.assertEqual(result.evacuated, 5)

    def test_intermediate_shelter_does_not_steal_another_agents_slot(self) -> None:
        network = CityNetwork()
        network.add_edge("o0", "s0", 1, 10, bidirectional=False)
        network.add_edge("s0", "s1", 1, 10, bidirectional=False)
        network.add_edge("o1", "s0", 8, 10, bidirectional=False)
        # o0 reserves s0 first, but its slow speed means o1 passes s0 earlier.
        agents = [Agent(0, "o0", speed=0.02), Agent(1, "o1")]
        shelters = {"s0": Shelter("s0", 1), "s1": Shelter("s1", 1)}
        router = MinCostFlowRouter()
        router.prepare(agents, network, shelters)
        self.assertEqual(router.assignment, {0: "s0", 1: "s1"})
        self.assertFalse(router.can_admit(agents[1], "s0"))
        self.assertTrue(router.can_admit(agents[1], "s1"))
        result = EvacuationSimulation(network, agents, list(shelters.values()), router=router).run()
        self.assertEqual(result.evacuated, 2)
        for agent in agents:
            self.assertEqual(agent.current_node, router.assignment[agent.id])
        self.assertLess(agents[1].evacuated_at, agents[0].evacuated_at)

    def test_route_cache_is_invalidated_by_closure(self) -> None:
        network = CityNetwork()
        network.add_edge("o", "s", 1, 10, bidirectional=False)
        network.add_edge("o", "x", 2, 10, bidirectional=False)
        network.add_edge("x", "s", 2, 10, bidirectional=False)
        agents = [Agent(0, "o")]
        shelters = {"s": Shelter("s", 1)}
        router = MinCostFlowRouter()
        router.prepare(agents, network, shelters)
        self.assertEqual(router.route(agents[0], network, shelters, 0), ["o", "s"])
        network.set_blocked("o", "s")
        self.assertEqual(router.route(agents[0], network, shelters, 1), ["o", "x", "s"])
        network.set_blocked("x", "s")
        self.assertIsNone(router.route(agents[0], network, shelters, 2))
        self.assertEqual(router.assignment, {0: "s"})

    def test_prepare_replaces_old_assignment_and_statistics(self) -> None:
        network = transport_network([[1]])
        router = MinCostFlowRouter()
        shelters = {"s0": Shelter("s0", 1)}
        agents = [Agent(0, "o0")]
        router.prepare(agents, network, shelters)
        router.route(agents[0], network, shelters, 0)
        self.assertGreater(router.stats.searches, 0)
        self.assertGreater(router.stats.cache_hits, 0)
        router.prepare([], network, shelters)
        self.assertEqual(router.assignment, {})
        self.assertEqual(router.planned_flow, 0)
        self.assertEqual(router.planned_distance, 0)
        self.assertEqual(router.stats.searches, 0)
        self.assertEqual(router.stats.cache_hits, 0)

    def test_random_tiny_problems_match_exhaustive_assignments(self) -> None:
        rng = random.Random(13)
        for trial in range(50):
            count = rng.randint(1, 5)
            destination_count = rng.randint(1, 3)
            capacities = [rng.randint(0, 3) for _ in range(destination_count)]
            costs = [
                [rng.randint(1, 12) if rng.random() > 0.25 else None for _ in capacities]
                for _ in range(count)
            ]
            optimum: tuple[int, int] | None = None
            for assignment in itertools.product(range(-1, destination_count), repeat=count):
                occupancy = Counter(assignment)
                if any(occupancy[target] > capacity for target, capacity in enumerate(capacities)):
                    continue
                if any(target >= 0 and costs[origin][target] is None for origin, target in enumerate(assignment)):
                    continue
                flow = sum(target >= 0 for target in assignment)
                distance = sum(costs[origin][target] for origin, target in enumerate(assignment) if target >= 0)
                candidate = (-flow, distance)
                if optimum is None or candidate < optimum:
                    optimum = candidate
            router = MinCostFlowRouter()
            network = transport_network(costs)
            agents = [Agent(index, f"o{index}") for index in range(count)]
            shelters = {f"s{index}": Shelter(f"s{index}", capacity) for index, capacity in enumerate(capacities)}
            router.prepare(agents, network, shelters)
            with self.subTest(trial=trial, costs=costs, capacities=capacities):
                self.assertEqual((-router.planned_flow, router.planned_distance), optimum)
                self.assertEqual(len(router.assignment), router.planned_flow)


if __name__ == "__main__":
    unittest.main()
