import unittest

from exev.models import Agent, AgentStatus, Shelter
from exev.network import CityNetwork
from exev.scenarios import grid_scenario
from exev.simulation import EvacuationSimulation, SimulationConfig
from exev.routing import AStarRouter, CongestionAwareRouter, DijkstraRouter, HazardAwareRouter


class SimulationTests(unittest.TestCase):
    def test_movement_uses_tick_snapshot_independent_of_iteration_order(self):
        progress_values = []
        for reverse in (False, True):
            network = CityNetwork()
            network.add_edge("A", "B", 1, 2)
            near, far = Agent(0, "A"), Agent(1, "A")
            sim = EvacuationSimulation(network, [far, near] if reverse else [near, far], [Shelter("B", 2)])
            for agent in (near, far):
                agent.status = AgentStatus.TRAVELING
                agent.edge = ("A", "B")
                network.enter("A", "B")
            near.edge_progress = 0.99
            sim._advance_travelers()
            progress_values.append(far.edge_progress)
        self.assertAlmostEqual(progress_values[0], 1 / 3)
        self.assertEqual(progress_values[0], progress_values[1])

    def test_noncontiguous_id_priority_rotates(self):
        network = CityNetwork()
        network.add_edge("A", "B", 0.1, 1)
        agents = [Agent(0, "A"), Agent(2, "A")]
        sim = EvacuationSimulation(network, agents, [Shelter("B", 2)])
        sim.tick = 1
        sim.step()
        self.assertEqual(agents[1].status, AgentStatus.TRAVELING)
        self.assertEqual(agents[0].status, AgentStatus.WAITING)

    def test_shortest_and_congestion_routers_conserve_population_and_capacity(self):
        for router_type in (DijkstraRouter, AStarRouter, CongestionAwareRouter, HazardAwareRouter):
            network, agents, shelters = grid_scenario(5, 4, 75, seed=4)
            sim = EvacuationSimulation(network, agents, shelters, router_type())
            while sim.tick < 500 and sim._has_active_agents():
                sim.step()
                self.assertEqual(len(agents), sum(a.status in (AgentStatus.WAITING, AgentStatus.TRAVELING,
                                                            AgentStatus.EVACUATED, AgentStatus.STRANDED) for a in agents))
                self.assertEqual(sum(s.occupants for s in shelters),
                                 sum(a.status == AgentStatus.EVACUATED for a in agents))
                for edge in network.edges:
                    occupied = sum(a.edge == (edge.source, edge.target) for a in agents)
                    self.assertEqual(network.occupancy(edge.source, edge.target), occupied)
                    self.assertLessEqual(occupied, edge.capacity)
                self.assertTrue(all(s.occupants <= s.capacity for s in shelters))
            self.assertEqual(sim.result().evacuated, 75, router_type.__name__)

    def test_empty_population_capped_run_and_progress(self):
        network, agents, shelters = grid_scenario(2, 2, 0)
        self.assertEqual(EvacuationSimulation(network, agents, shelters).run().evacuation_rate, 1)
        network, agents, shelters = grid_scenario(4, 4, 10)
        ticks = []
        sim = EvacuationSimulation(network, agents, shelters, config=SimulationConfig(max_ticks=1))
        result = sim.run(progress=lambda current: ticks.append(current.tick), progress_interval=1)
        self.assertEqual(result.unfinished, 10)
        self.assertEqual(result.elapsed_ticks, 1)
        self.assertEqual(ticks[0], 0)
        self.assertEqual(ticks[-1], 1)
        self.assertGreaterEqual(result.wall_seconds, result.preparation_seconds + result.routing_seconds)

    def test_invalid_scenario_inputs(self):
        with self.assertRaises(ValueError):
            grid_scenario(agent_count=-1)
        with self.assertRaises(ValueError):
            SimulationConfig(max_ticks=0)
        with self.assertRaises(ValueError):
            Agent(0, "A", float("nan"))
        with self.assertRaises(ValueError):
            Shelter("A", -1)

    def test_agents_evacuate_to_shelter(self) -> None:
        network = CityNetwork()
        network.add_edge("A", "B", distance=2, capacity=2)
        agents = [Agent(1, "A"), Agent(2, "A", speed=0.5)]
        simulation = EvacuationSimulation(network, agents, [Shelter("B", 2)])

        result = simulation.run()

        self.assertEqual(result.evacuated, 2)
        self.assertEqual(result.stranded, 0)
        self.assertEqual(agents[0].status, AgentStatus.EVACUATED)
        self.assertGreater(agents[1].evacuated_at, agents[0].evacuated_at)
        self.assertGreater(result.speed_group_evacuation_gap, 0)
        self.assertGreaterEqual(result.p95_evacuation_time, result.p50_evacuation_time)

    def test_capacity_creates_queue(self) -> None:
        network = CityNetwork()
        network.add_edge("A", "B", distance=1, capacity=1)
        agents = [Agent(i, "A") for i in range(3)]

        result = EvacuationSimulation(network, agents, [Shelter("B", 3)]).run()

        self.assertEqual(result.evacuated, 3)
        self.assertGreater(result.mean_waiting_time, 0)
        self.assertEqual(result.total_waiting_time, sum(a.total_waiting_time for a in agents))
        self.assertAlmostEqual(
            result.total_distance_traveled, sum(a.distance_traveled for a in agents)
        )
        self.assertGreater(result.full_edge_ticks, 0)
        self.assertGreater(result.total_edge_occupancy_ticks, 0)
        self.assertGreater(result.mean_edge_utilization, 0)
        self.assertLessEqual(result.mean_edge_utilization, 1)
        self.assertEqual(result.hazard_exposure_gini, 0)

    def test_routing_reasons_count_unique_affected_agents(self) -> None:
        network = CityNetwork()
        network.add_edge("A", "B", distance=20, capacity=1)
        agents = [Agent(1, "A"), Agent(2, "A")]
        simulation = EvacuationSimulation(
            network, agents, [Shelter("B", 2)],
            config=SimulationConfig(max_ticks=15, reroute_wait_threshold=2),
        )
        simulation.run()
        self.assertEqual(simulation.routing_reasons["congestion delay"], 1)
        self.assertGreater(simulation.routing_decisions["congestion delay"], 1)

    def test_blocked_route_is_avoided(self) -> None:
        network = CityNetwork()
        network.add_edge("A", "B", distance=1, capacity=2)
        network.add_edge("B", "D", distance=1, capacity=2)
        network.add_edge("A", "C", distance=2, capacity=2)
        network.add_edge("C", "D", distance=2, capacity=2)
        network.set_blocked("B", "D")
        agent = Agent(1, "A")

        result = EvacuationSimulation(network, [agent], [Shelter("D", 1)]).run()

        self.assertEqual(result.evacuated, 1)
        self.assertEqual(agent.route, ["A", "C", "D"])
        self.assertAlmostEqual(agent.distance_traveled, 4)

    def test_unreachable_agent_is_stranded(self) -> None:
        network = CityNetwork()
        network.add_edge("A", "B", distance=1, capacity=1)
        network.set_blocked("A", "B")

        result = EvacuationSimulation(network, [Agent(1, "A")], [Shelter("B", 1)]).run()

        self.assertEqual(result.stranded, 1)
        self.assertEqual(result.evacuated, 0)

    def test_thousands_of_agents_complete(self) -> None:
        network, agents, shelters = grid_scenario(8, 8, 2_000, seed=3)
        result = EvacuationSimulation(
            network,
            agents,
            shelters,
            config=SimulationConfig(max_ticks=5_000),
        ).run()
        self.assertEqual(result.evacuated, 2_000)
        self.assertEqual(result.unfinished, 0)


if __name__ == "__main__":
    unittest.main()
