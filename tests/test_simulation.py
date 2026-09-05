import unittest

from evacsim.models import Agent, AgentStatus, Shelter
from evacsim.network import CityNetwork
from evacsim.scenarios import grid_scenario
from evacsim.simulation import EvacuationSimulation, SimulationConfig


class SimulationTests(unittest.TestCase):
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

    def test_capacity_creates_queue(self) -> None:
        network = CityNetwork()
        network.add_edge("A", "B", distance=1, capacity=1)
        agents = [Agent(i, "A") for i in range(3)]

        result = EvacuationSimulation(network, agents, [Shelter("B", 3)]).run()

        self.assertEqual(result.evacuated, 3)
        self.assertGreater(result.mean_waiting_time, 0)

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
