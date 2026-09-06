import csv
import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from evacsim.cli import main
from evacsim.experiments import ExperimentConfig, compare_routers, run_scenario
from evacsim.routing import ROUTER_NAMES


class CliTests(unittest.TestCase):
    def invoke(self, *arguments):
        stdout, stderr = io.StringIO(), io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            main(list(arguments))
        return stdout.getvalue(), stderr.getvalue()

    def test_single_json_and_progress_are_separate(self):
        stdout, stderr = self.invoke(
            "--agents", "10", "--width", "3", "--height", "3", "--progress-every", "2"
        )
        result = json.loads(stdout)
        self.assertEqual(result["total_agents"], 10)
        self.assertEqual(result["algorithm"], "dijkstra")
        self.assertEqual(result["evacuated"], 10)
        self.assertEqual(result["evacuation_rate"], 1.0)
        self.assertEqual(result["seed"], 7)
        self.assertEqual(result["model_version"], "stage2-synchronous-v1")
        self.assertIn("Preparing", stderr)
        self.assertIn("tick=0", stderr)
        self.assertIn("tick=2", stderr)
        self.assertIn("wall-clock seconds", stderr)

    def test_quiet_and_explicit_router(self):
        stdout, stderr = self.invoke(
            "--agents", "4", "--width", "2", "--height", "2", "--router", "astar", "--quiet"
        )
        self.assertEqual(json.loads(stdout)["algorithm"], "astar")
        self.assertEqual(stderr, "")

    def test_compare_exports_every_algorithm_and_seed_as_csv(self):
        stdout, stderr = self.invoke(
            "--agents", "4", "--width", "2", "--height", "2", "--compare",
            "--seeds", "2", "3", "--format", "csv", "--quiet",
        )
        records = list(csv.DictReader(io.StringIO(stdout)))
        self.assertEqual(len(records), 8)
        self.assertEqual(
            {(row["algorithm"], row["seed"]) for row in records},
            {(name, str(seed)) for name in ROUTER_NAMES for seed in (2, 3)},
        )
        self.assertTrue(all(int(row["total_agents"]) == 4 for row in records))
        self.assertTrue(all(int(row["evacuated"]) == 4 for row in records))
        self.assertTrue(all(row["python_version"] and row["model_version"] for row in records))
        self.assertEqual(stderr, "")

    def test_output_file_receives_json_instead_of_stdout(self):
        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "comparison.json"
            stdout, stderr = self.invoke(
                "--agents", "2", "--width", "2", "--height", "2", "--compare",
                "--seed", "19", "--output", str(output_path), "--quiet",
            )
            result = json.loads(output_path.read_text())
        self.assertEqual(stdout, "")
        self.assertEqual(stderr, "")
        self.assertEqual(result["seeds"], [19])
        self.assertEqual(result["configuration"]["agent_count"], 2)
        self.assertEqual(len(result["records"]), 4)

    def test_abbreviation_invalid_ranges_and_conflicting_options_are_rejected(self):
        invalid = [
            ["--agent", "10"], ["--agents", "-1"], ["--width", "1"],
            ["--height", "0"], ["--max-ticks", "0"], ["--reroute-wait", "0"],
            ["--progress-every", "0"], ["--router", "unknown"],
            ["--seeds", "1"], ["--compare", "--router", "astar"],
            ["--compare", "--seed", "1", "--seeds", "2"],
            ["--compare", "--seeds", "1", "1"],
        ]
        for arguments in invalid:
            with self.subTest(arguments=arguments):
                with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                    main(arguments)
                self.assertEqual(error.exception.code, 2)

    def test_zero_agents_and_truncated_runs_are_reported(self):
        stdout, _ = self.invoke("--agents", "0", "--quiet")
        empty = json.loads(stdout)
        self.assertEqual(empty["total_agents"], 0)
        self.assertEqual(empty["evacuation_rate"], 1.0)
        stdout, _ = self.invoke("--agents", "10", "--max-ticks", "1", "--quiet")
        truncated = json.loads(stdout)
        self.assertEqual(truncated["elapsed_ticks"], 1)
        self.assertGreater(truncated["unfinished"], 0)


class ExperimentTests(unittest.TestCase):
    def test_repeated_comparison_has_identical_non_timing_results(self):
        config = ExperimentConfig(width=3, height=3, agent_count=12)
        first = compare_routers(config, seeds=[3, 5])
        second = compare_routers(config, seeds=[3, 5])

        def without_timings(records):
            return [{key: value for key, value in row.items() if not key.endswith("_seconds")}
                    for row in records]

        self.assertEqual(without_timings(first["records"]), without_timings(second["records"]))
        self.assertEqual(len(first["records"]), 8)
        self.assertTrue(all(row["evacuated"] == 12 for row in first["records"]))

    def test_algorithm_order_does_not_leak_scenario_or_router_state(self):
        config = ExperimentConfig(width=3, height=3, agent_count=21)
        forward = compare_routers(config)["records"]
        reverse = compare_routers(config, algorithms=reversed(ROUTER_NAMES))["records"]

        def normalize(records):
            return {row["algorithm"]: {
                key: value for key, value in row.items() if not key.endswith("_seconds")
            } for row in records}

        self.assertEqual(normalize(forward), normalize(reverse))

    def test_direct_api_rejects_invalid_configuration(self):
        for changes in ({"agent_count": -1}, {"width": 1}, {"height": 1},
                        {"max_ticks": 0}, {"reroute_wait_threshold": 0}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                ExperimentConfig(**changes)
        with self.assertRaises(ValueError):
            compare_routers(ExperimentConfig(), seeds=[])
        with self.assertRaises(ValueError):
            compare_routers(ExperimentConfig(), seeds=[1, 1])
        with self.assertRaises(ValueError):
            run_scenario(ExperimentConfig(), progress_interval=0)


if __name__ == "__main__":
    unittest.main()
