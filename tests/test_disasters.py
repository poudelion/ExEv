import unittest

from exev.disasters import (
    DisasterSchedule,
    EdgeHazardEvent,
    NodeHazardEvent,
    RoadStatusEvent,
    ShelterCapacityEvent,
    grid_disaster_schedule,
)
from exev.models import Agent, AgentStatus, Shelter
from exev.network import CityNetwork
from exev.scenarios import grid_scenario
from exev.simulation import EvacuationSimulation


class DisasterEventTests(unittest.TestCase):
    def test_tick_zero_closure_is_applied_before_initial_routing(self):
        network = CityNetwork()
        network.add_edge("A", "B", 1, 10)
        network.add_edge("B", "S", 1, 10)
        network.add_edge("A", "C", 2, 10)
        network.add_edge("C", "S", 2, 10)
        agent = Agent(0, "A")
        schedule = DisasterSchedule([
            RoadStatusEvent(0, "B", "S", blocked=True, bidirectional=True)
        ])

        result = EvacuationSimulation(
            network, [agent], [Shelter("S", 1)], disaster_schedule=schedule
        ).run()

        self.assertEqual(agent.route, ["A", "C", "S"])
        self.assertEqual(result.evacuated, 1)
        self.assertEqual(result.events_processed, 1)

    def test_agent_on_newly_closed_road_finishes_crossing(self):
        network = CityNetwork()
        network.add_edge("A", "S", 2, 10, bidirectional=False)
        agent = Agent(0, "A")
        schedule = DisasterSchedule([RoadStatusEvent(1, "A", "S")])

        result = EvacuationSimulation(
            network, [agent], [Shelter("S", 1)], disaster_schedule=schedule
        ).run()

        self.assertEqual(result.evacuated, 1)
        self.assertEqual(agent.status, AgentStatus.EVACUATED)
        self.assertTrue(network.edge("A", "S").blocked)

    def test_future_shelter_reopening_prevents_premature_stranding(self):
        network = CityNetwork()
        network.add_edge("A", "S", 1, 10, bidirectional=False)
        agent = Agent(0, "A")
        schedule = DisasterSchedule([
            ShelterCapacityEvent(0, "S", 0),
            ShelterCapacityEvent(2, "S", 1),
        ])

        result = EvacuationSimulation(
            network, [agent], [Shelter("S", 1)], disaster_schedule=schedule
        ).run()

        self.assertEqual(result.evacuated, 1)
        self.assertEqual(result.stranded, 0)
        self.assertGreaterEqual(agent.total_waiting_time, 2)
        self.assertEqual(result.events_processed, 2)

    def test_node_and_edge_exposure_are_accumulated(self):
        network = CityNetwork()
        network.add_edge("A", "S", 2, 10, bidirectional=False)
        agent = Agent(0, "A")
        schedule = DisasterSchedule([
            NodeHazardEvent(0, "A", 0.5),
            EdgeHazardEvent(0, "A", "S", 2.0),
        ])

        result = EvacuationSimulation(
            network, [agent], [Shelter("S", 1)], disaster_schedule=schedule
        ).run()

        self.assertGreater(agent.hazard_exposure, 0.5)
        self.assertEqual(result.total_hazard_exposure, agent.hazard_exposure)
        self.assertEqual(result.mean_hazard_exposure, agent.hazard_exposure)
        self.assertEqual(result.max_hazard_exposure, agent.hazard_exposure)

    def test_invalid_events_and_profiles_are_rejected(self):
        network = CityNetwork()
        network.add_edge("A", "S", 1, 1, bidirectional=False)
        with self.assertRaises(ValueError):
            RoadStatusEvent(-1, "A", "S")
        with self.assertRaises(ValueError):
            EdgeHazardEvent(0, "A", "S", float("nan"))
        with self.assertRaises(ValueError):
            DisasterSchedule([RoadStatusEvent(0, "missing", "S")]).validate(
                network, {"S": Shelter("S", 1)}
            )
        with self.assertRaises(ValueError):
            grid_disaster_schedule(3, 3, 10, "tornado")

    def test_synthetic_profiles_are_fresh_and_deterministic(self):
        first = grid_disaster_schedule(4, 3, 20, "fire")
        second = grid_disaster_schedule(4, 3, 20, "fire")
        self.assertGreater(first.event_count, 12)
        self.assertEqual(first.event_count, second.event_count)
        self.assertIsNot(first, second)

    def test_road_profile_closes_a_corridor_but_keeps_detours(self):
        network, _, shelter_list = grid_scenario(5, 5, 0)
        shelters = {shelter.node: shelter for shelter in shelter_list}
        schedule = grid_disaster_schedule(5, 5, 0, "road-closure")
        schedule.validate(network, shelters)

        self.assertEqual(schedule.apply_tick(10, network, shelters), 4)
        for row in (0, 1, 3, 4):
            self.assertTrue(network.edge(f"1,{row}", f"2,{row}").blocked)
        self.assertFalse(network.edge("1,2", "2,2").blocked)


if __name__ == "__main__":
    unittest.main()
