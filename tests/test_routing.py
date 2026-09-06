import itertools
import random
import unittest

from exev.models import Agent, Shelter
from exev.network import CityNetwork
from exev.routing import (
    AStarRouter,
    CongestionAwareRouter,
    DijkstraRouter,
    HazardAwareRouter,
    create_router,
)


def distance(network, path):
    return float("inf") if path is None else sum(network.edge(u, v).distance for u, v in zip(path, path[1:]))


class StaticRoutingTests(unittest.TestCase):
    def test_bidirectional_edit_rejection_is_atomic(self):
        network = CityNetwork()
        network.add_edge("A", "B", 1, 2)
        network.enter("B", "A")
        version = network.topology_version
        with self.assertRaises(ValueError):
            network.add_edge("A", "B", 9, 2)
        self.assertEqual(network.edge("A", "B").distance, 1)
        self.assertEqual(network.edge("B", "A").distance, 1)
        self.assertEqual(network.topology_version, version)

    def test_one_way_graph_and_topology_shelter_network_cache_changes(self):
        for router in (DijkstraRouter(), AStarRouter()):
            with self.subTest(router=router.name):
                network = CityNetwork()
                network.add_edge("A", "B", 1, 2, bidirectional=False)
                network.add_edge("B", "S", 1, 2, bidirectional=False)
                network.add_edge("A", "T", 5, 2, bidirectional=False)
                agent = Agent(1, "A")
                shelters = {"S": Shelter("S", 1), "T": Shelter("T", 1)}
                router.prepare([agent], network, shelters)
                self.assertEqual(router.route(agent, network, shelters, 0), ["A", "B", "S"])
                first_searches = router.stats.searches
                path = router.route(agent, network, shelters, 1)
                path.append("should-not-mutate-cache")
                self.assertEqual(router.route(agent, network, shelters, 2), ["A", "B", "S"])
                self.assertEqual(first_searches, router.stats.searches)
                network.set_blocked("B", "S")
                self.assertEqual(router.route(agent, network, shelters, 3), ["A", "T"])
                network.set_blocked("B", "S", False)
                self.assertEqual(router.route(agent, network, shelters, 4), ["A", "B", "S"])
                shelters["S"].admit()
                self.assertEqual(router.route(agent, network, shelters, 5), ["A", "T"])
                shelters["S"].occupants = 0
                network.add_edge("A", "T", 0.5, 2, bidirectional=False)
                self.assertEqual(router.route(agent, network, shelters, 6), ["A", "T"])
                second = CityNetwork()
                second.add_edge("A", "S", 1, 2, bidirectional=False)
                second.add_node("T")
                self.assertEqual(router.route(agent, second, shelters, 7), ["A", "S"])
                reverse = router.route(Agent(2, "S"), second, {"A": Shelter("A", 1)}, 8)
                self.assertIsNone(reverse)

    def test_astar_geometric_shortcut_and_missing_coordinates(self):
        network = CityNetwork()
        network.add_edge("start", "goal", 5, 2, bidirectional=False)
        network.add_edge("start", "far", 1, 2, bidirectional=False)
        network.add_edge("far", "goal", 1, 2, bidirectional=False)
        for node, x in (("start", 0), ("goal", 1), ("far", 1000)):
            network.set_position(node, x, 0)
        router = AStarRouter()
        agent, shelters = Agent(1, "start"), {"goal": Shelter("goal", 1)}
        self.assertEqual(router.route(agent, network, shelters, 0), ["start", "far", "goal"])
        network.add_node("no-coordinates")
        self.assertEqual(router.route(agent, network, shelters, 1), ["start", "far", "goal"])
        self.assertIsNone(router.route(agent, network, {}, 2))

    def test_static_optima_against_exhaustive_simple_paths(self):
        rng = random.Random(19)
        for case in range(20):
            network = CityNetwork()
            nodes = [str(i) for i in range(5)]
            for node in nodes:
                network.set_position(node, rng.random() * 20, rng.random() * 20)
            for u, v in itertools.permutations(nodes, 2):
                if rng.random() < 0.45:
                    network.add_edge(u, v, rng.randint(1, 9) / 3, 2, bidirectional=False)
            best = float("inf")
            for count in range(4):
                for middle in itertools.permutations(nodes[1:-1], count):
                    path = [nodes[0], *middle, nodes[-1]]
                    if all(v in network.neighbors(u) for u, v in zip(path, path[1:])):
                        best = min(best, distance(network, path))
            for router in (DijkstraRouter(), AStarRouter()):
                path = router.route(Agent(0, "0"), network, {"4": Shelter("4", 1)}, 0)
                with self.subTest(case=case, router=router.name):
                    self.assertAlmostEqual(distance(network, path), best)

    def test_dijkstra_tree_is_shared_by_many_origins(self):
        network = CityNetwork()
        for i in range(30):
            network.add_edge(str(i), "S", i + 1, 2, bidirectional=False)
        router = DijkstraRouter()
        for i in range(30):
            self.assertEqual(router.route(Agent(i, str(i)), network, {"S": Shelter("S", 30)}, 0), [str(i), "S"])
        self.assertEqual(router.stats.searches, 1)
        self.assertEqual(router.stats.cache_hits, 29)


class CongestionRoutingTests(unittest.TestCase):
    def test_hazard_aware_router_detours_and_invalidates_same_tick_cache(self):
        network = CityNetwork()
        network.add_edge("A", "B", 1, 100, bidirectional=False)
        network.add_edge("B", "S", 1, 100, bidirectional=False)
        network.add_edge("A", "C", 2, 100, bidirectional=False)
        network.add_edge("C", "S", 2, 100, bidirectional=False)
        agent, shelters = Agent(1, "A"), {"S": Shelter("S", 1)}
        router = HazardAwareRouter()

        self.assertEqual(router.route(agent, network, shelters, 0), ["A", "B", "S"])
        network.set_edge_hazard("A", "B", 2.0)
        network.set_edge_hazard("B", "S", 2.0)
        self.assertEqual(router.route(agent, network, shelters, 0), ["A", "C", "S"])
        configured = create_router("hazard-aware", hazard_weight=3.5)
        self.assertEqual(configured.name, "hazard-aware")
        self.assertEqual(configured.hazard_weight, 3.5)

    def test_live_costs_detour_and_tick_snapshot(self):
        network = CityNetwork()
        network.add_edge("A", "B", 1, 1, bidirectional=False)
        network.add_edge("B", "S", 1, 1, bidirectional=False)
        network.add_edge("A", "C", 1.3, 100, bidirectional=False)
        network.add_edge("C", "S", 1.3, 100, bidirectional=False)
        agent, shelters = Agent(1, "A"), {"S": Shelter("S", 5)}
        self.assertEqual(DijkstraRouter().route(agent, network, shelters, 0), ["A", "B", "S"])
        self.assertEqual(CongestionAwareRouter().route(agent, network, shelters, 0), ["A", "C", "S"])
        # Snapshot reuse must not depend on preceding agents entering a road.
        router = CongestionAwareRouter()
        first = router.route(agent, network, shelters, 0)
        network.enter("A", "C")
        self.assertEqual(router.route(agent, network, shelters, 0), first)
        self.assertEqual(router.stats.searches, 1)
        router.route(agent, network, shelters, 1)
        self.assertEqual(router.stats.searches, 2)

    def test_speed_specific_costs_and_full_road_is_waitable(self):
        network = CityNetwork()
        network.add_edge("A", "S", 1, 100, speed_limit=0.2, bidirectional=False)
        network.add_edge("A", "B", 1, 100, bidirectional=False)
        network.add_edge("B", "S", 1, 100, bidirectional=False)
        router, shelters = CongestionAwareRouter(), {"S": Shelter("S", 10)}
        self.assertEqual(router.route(Agent(1, "A", speed=0.1), network, shelters, 0), ["A", "S"])
        self.assertEqual(router.route(Agent(2, "A", speed=1), network, shelters, 0), ["A", "B", "S"])
        self.assertEqual(router.stats.searches, 2)
        full = CityNetwork()
        full.add_edge("A", "S", 1, 1, bidirectional=False)
        full.enter("A", "S")
        self.assertEqual(router.route(Agent(1, "A"), full, shelters, 1), ["A", "S"])

    def test_invalid_router_name(self):
        with self.assertRaises(ValueError):
            create_router("quantum-magic")


if __name__ == "__main__":
    unittest.main()
