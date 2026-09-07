from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from exev.models import Agent, Shelter
from exev.network import CityNetwork
from exev.qubo import (
    ExactQUBOSolver,
    QUBOModel,
    QUBOSimulatedAnnealingRouter,
    QUBOSolution,
    SimulatedAnnealingQUBOSolver,
)
from exev.qubo_benchmark import compare_qubo_backends
from exev.qubo_cli import main as qubo_main
from exev.qubo_io import (
    dumps_qubo,
    loads_qubo,
    qubo_fingerprint,
    qubo_from_dict,
    qubo_to_dict,
    read_qubo,
    write_qubo,
)
from exev.simulation import EvacuationSimulation


def one_hot_model() -> QUBOModel:
    model = QUBOModel(constant=0.25)
    first = model.add_variable("route α")
    second = model.add_variable("route β")
    model.add_squared_expression(
        {first: 1.0, second: 1.0},
        offset=-1.0,
        weight=3.0,
    )
    model.add_linear(first, 0.5)
    model.add_linear(second, 1.0)
    return model


class QUBOSerializationTests(unittest.TestCase):
    def test_name_based_json_round_trip_preserves_every_energy(self):
        model = one_hot_model()
        document = dumps_qubo(model)
        restored = loads_qubo(document)

        self.assertEqual(qubo_to_dict(restored), qubo_to_dict(model))
        self.assertEqual(qubo_fingerprint(restored), qubo_fingerprint(model))
        for sample in ((0, 0), (0, 1), (1, 0), (1, 1)):
            with self.subTest(sample=sample):
                self.assertEqual(restored.energy(sample), model.energy(sample))

    def test_file_round_trip_creates_parent_and_returns_absolute_path(self):
        model = one_hot_model()
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "nested" / "problem.json"
            written = write_qubo(model, destination)
            restored = read_qubo(destination)

        self.assertTrue(written.is_absolute())
        self.assertEqual(qubo_to_dict(restored), qubo_to_dict(model))

    def test_malformed_documents_are_rejected(self):
        valid = qubo_to_dict(one_hot_model())
        invalid_documents = [
            {**valid, "format": "other"},
            {**valid, "format_version": 99},
            {**valid, "vartype": "SPIN"},
            {**valid, "variables": ["x", "x"]},
            {**valid, "constant": float("nan")},
            {**valid, "linear": [{"variable": "missing", "bias": 1.0}]},
            {
                **valid,
                "quadratic": [{"first": "route α", "second": "route α", "bias": 1.0}],
            },
        ]
        for document in invalid_documents:
            with self.subTest(document=document), self.assertRaises(ValueError):
                qubo_from_dict(document)
        with self.assertRaises(ValueError):
            loads_qubo("{not-json")
        with self.assertRaises(ValueError):
            QUBOModel().add_variable("")


class QUBOBackendTests(unittest.TestCase):
    def test_exact_and_annealing_compare_on_the_identical_model(self):
        model = one_hot_model()
        report = compare_qubo_backends(
            model,
            {
                "exact-reference": ExactQUBOSolver(),
                "local-sa": SimulatedAnnealingQUBOSolver(
                    sweeps=20,
                    restarts=2,
                    seed=11,
                ),
            },
            initial_sample=(1, 0),
            feasibility=lambda sample: sum(sample) == 1,
        )

        self.assertEqual(report.qubo_fingerprint, qubo_fingerprint(model))
        self.assertEqual(report.variable_count, 2)
        self.assertEqual([row.backend for row in report.backends], ["exact-reference", "local-sa"])
        self.assertTrue(all(row.constraint_feasible for row in report.backends))
        self.assertTrue(all(row.is_best_observed for row in report.backends))
        self.assertTrue(all(row.energy_gap_from_best_observed == 0 for row in report.backends))
        self.assertTrue(all(row.solve_seconds >= 0 for row in report.backends))

    def test_inconsistent_backend_result_is_rejected(self):
        class BadBackend:
            def solve(self, model, initial_sample=None):
                return QUBOSolution((0,) * model.variable_count, -999.0, "bad", 1)

        with self.assertRaises(ValueError):
            compare_qubo_backends(one_hot_model(), {"bad": BadBackend()})
        with self.assertRaises(ValueError):
            compare_qubo_backends(one_hot_model(), {})

    def test_router_accepts_an_injected_solver_backend(self):
        class RecordingBackend:
            def __init__(self):
                self.calls = 0
                self.exact = ExactQUBOSolver()

            def solve(self, model, initial_sample=None):
                self.calls += 1
                return self.exact.solve(model, initial_sample)

        network = CityNetwork()
        network.add_edge("origin", "shelter", 1, 2, bidirectional=False)
        backend = RecordingBackend()
        router = QUBOSimulatedAnnealingRouter(batch_size=2, solver=backend)
        result = EvacuationSimulation(
            network,
            [Agent(0, "origin"), Agent(1, "origin")],
            [Shelter("shelter", 2)],
            router=router,
        ).run()

        self.assertEqual(result.evacuated, 2)
        self.assertEqual(backend.calls, 1)


class QUBOCliTests(unittest.TestCase):
    def invoke(self, *arguments: str) -> tuple[str, str]:
        stdout, stderr = io.StringIO(), io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            qubo_main(list(arguments))
        return stdout.getvalue(), stderr.getvalue()

    def test_cli_compares_a_serialized_problem(self):
        with tempfile.TemporaryDirectory() as directory:
            problem = write_qubo(one_hot_model(), Path(directory) / "problem.json")
            stdout, stderr = self.invoke(
                "--input",
                str(problem),
                "--initial-sample",
                "1,0",
                "--sweeps",
                "10",
                "--restarts",
                "1",
            )
            payload = json.loads(stdout)
        self.assertEqual(payload["metadata"]["package_version"], "0.6.0")
        self.assertEqual(payload["metadata"]["model_version"], "stage6-backends-v1")

        self.assertEqual(payload["variable_count"], 2)
        self.assertEqual(len(payload["backends"]), 2)
        self.assertEqual(
            {row["backend"] for row in payload["backends"]},
            {"exact", "simulated-annealing"},
        )
        self.assertEqual(stderr, "")

    def test_cli_rejects_duplicate_backends_and_bad_sample_length(self):
        with tempfile.TemporaryDirectory() as directory:
            problem = write_qubo(one_hot_model(), Path(directory) / "problem.json")
            invalid = (
                ["--input", str(problem), "--backends", "exact", "exact"],
                ["--input", str(problem), "--initial-sample", "1"],
            )
            for arguments in invalid:
                with self.subTest(arguments=arguments):
                    with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                        qubo_main(arguments)


if __name__ == "__main__":
    unittest.main()
