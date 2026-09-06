from __future__ import annotations

import unittest

from exev.models import Agent, Shelter
from exev.network import CityNetwork
from exev.qubo import (
    ExactQUBOSolver,
    QUBOModel,
    QUBOSimulatedAnnealingRouter,
    RouteOption,
    SimulatedAnnealingQUBOSolver,
    build_route_assignment_qubo,
)
from exev.simulation import EvacuationSimulation


class QUBOModelTests(unittest.TestCase):
    def test_squared_expression_and_exact_solver(self) -> None:
        model = QUBOModel()
        first = model.add_variable("first")
        second = model.add_variable("second")
        model.add_squared_expression({first: 1.0, second: 1.0}, offset=-1.0)
        self.assertEqual(model.energy((0, 0)), 1.0)
        self.assertEqual(model.energy((1, 0)), 0.0)
        self.assertEqual(model.energy((0, 1)), 0.0)
        self.assertEqual(model.energy((1, 1)), 1.0)

        solution = ExactQUBOSolver().solve(model)
        self.assertEqual(solution.energy, 0.0)
        self.assertIn(solution.sample, ((1, 0), (0, 1)))
        self.assertEqual(solution.iterations, 4)

    def test_seeded_annealer_preserves_a_known_best_sample(self) -> None:
        model = QUBOModel()
        first = model.add_variable("first")
        second = model.add_variable("second")
        model.add_squared_expression({first: 1.0}, offset=-1.0, weight=2.0)
        model.add_linear(second, 3.0)
        solver = SimulatedAnnealingQUBOSolver(sweeps=20, restarts=2, seed=11)
        first_solution = solver.solve(model, initial_sample=(1, 0))
        second_solution = solver.solve(model, initial_sample=(1, 0))
        self.assertEqual(first_solution, second_solution)
        self.assertEqual(first_solution.sample, (1, 0))
        self.assertEqual(first_solution.energy, 0.0)
        self.assertEqual(first_solution.iterations, 80)


class RouteAssignmentTests(unittest.TestCase):
    def test_exact_assignment_respects_capacity_and_minimizes_cost(self) -> None:
        options = {
            0: (
                RouteOption(0, "A", ("o0", "A"), 1.0),
                RouteOption(0, "B", ("o0", "B"), 5.0),
                RouteOption(0, None, (), 100.0),
            ),
            1: (
                RouteOption(1, "A", ("o1", "A"), 2.0),
                RouteOption(1, "B", ("o1", "B"), 1.0),
                RouteOption(1, None, (), 100.0),
            ),
        }
        problem = build_route_assignment_qubo(options, {"A": 1, "B": 1})
        solution = ExactQUBOSolver().solve(problem.model)
        selected = problem.selected_options(solution.sample)

        self.assertTrue(problem.is_feasible(solution.sample))
        self.assertEqual(solution.energy, 2.0)
        self.assertEqual(selected[0].shelter, "A")
        self.assertEqual(selected[1].shelter, "B")
        self.assertTrue(problem.is_feasible(problem.greedy_sample()))

    def test_router_runs_end_to_end_and_exports_diagnostics(self) -> None:
        network = CityNetwork()
        network.add_edge("o0", "x", 1, 1, bidirectional=False)
        network.add_edge("o1", "x", 1, 1, bidirectional=False)
        network.add_edge("x", "A", 1, 1, bidirectional=False)
        network.add_edge("o0", "B", 4, 2, bidirectional=False)
        network.add_edge("o1", "B", 4, 2, bidirectional=False)
        agents = [Agent(0, "o0"), Agent(1, "o1")]
        router = QUBOSimulatedAnnealingRouter(
            batch_size=2, sweeps=20, restarts=2, seed=5
        )
        result = EvacuationSimulation(
            network, agents, [Shelter("A", 1), Shelter("B", 1)], router=router
        ).run()

        self.assertEqual(result.evacuated, 2)
        self.assertEqual(result.stranded, 0)
        self.assertEqual(result.qubo_batches, 1)
        self.assertGreater(result.qubo_total_variables, 0)
        self.assertEqual(result.qubo_planned_flow, 2)
        self.assertGreater(result.qubo_annealing_iterations, 0)
        self.assertEqual(result.qubo_fallback_batches, 0)
        self.assertEqual({agent.current_node for agent in agents}, {"A", "B"})

    def test_multiple_candidates_can_split_agents_across_equal_routes(self) -> None:
        network = CityNetwork()
        network.add_edge("o", "a", 1, 2, bidirectional=False)
        network.add_edge("a", "s", 1, 2, bidirectional=False)
        network.add_edge("o", "b", 1, 2, bidirectional=False)
        network.add_edge("b", "s", 1, 2, bidirectional=False)
        agents = [Agent(0, "o"), Agent(1, "o")]
        router = QUBOSimulatedAnnealingRouter(
            batch_size=2, routes_per_shelter=2, sweeps=20, restarts=2, seed=3
        )
        result = EvacuationSimulation(
            network, agents, [Shelter("s", 2)], router=router
        ).run()

        self.assertEqual(result.evacuated, 2)
        self.assertEqual(
            {tuple(agent.route) for agent in agents},
            {("o", "a", "s"), ("o", "b", "s")},
        )

    def test_invalid_inputs_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            QUBOSimulatedAnnealingRouter(batch_size=0)
        with self.assertRaises(ValueError):
            QUBOSimulatedAnnealingRouter(routes_per_shelter=0)
        with self.assertRaises(ValueError):
            SimulatedAnnealingQUBOSolver(sweeps=0)
        with self.assertRaises(ValueError):
            RouteOption(1, None, ("not-empty",), 1.0)
        with self.assertRaises(ValueError):
            build_route_assignment_qubo(
                {1: (RouteOption(1, None, (), 1.0),)}, {}, congestion_weight=-1.0
            )


if __name__ == "__main__":
    unittest.main()
