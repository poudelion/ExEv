import csv
import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from exev.routing import ROUTER_NAMES
from exev.studies import StudyDefinition, plan_study, run_study
from exev.study_cli import main as study_main


class StudyPlannerTests(unittest.TestCase):
    def test_algorithm_aware_sweeps_do_not_duplicate_baselines(self):
        definition = StudyDefinition(
            map_sizes=((3, 3),),
            agent_counts=(5,),
            seeds=(1, 2),
            disaster_profiles=("none",),
            algorithms=("dijkstra", "hazard-aware", "qubo-sa"),
            hazard_weights=(0.0, 1.0),
            qubo_batch_sizes=(2, 4),
            qubo_routes_per_shelter=(1,),
            qubo_congestion_weights=(1.0,),
            qubo_sweeps=(2,),
            qubo_restarts=(1,),
        )

        plan = plan_study(definition)
        counts = {
            algorithm: sum(run.algorithm == algorithm for run in plan)
            for algorithm in definition.algorithms
        }

        dijkstra = [run for run in plan if run.algorithm == "dijkstra"]
        hazard_aware = [run for run in plan if run.algorithm == "hazard-aware"]
        qubo = [run for run in plan if run.algorithm == "qubo-sa"]
        self.assertEqual({run.config.qubo_batch_size for run in dijkstra}, {8})
        self.assertEqual({run.config.qubo_batch_size for run in hazard_aware}, {8})
        self.assertEqual({run.config.qubo_batch_size for run in qubo}, {2, 4})
        self.assertEqual({run.config.hazard_weight for run in dijkstra}, {1.0})
        self.assertEqual({run.config.hazard_weight for run in hazard_aware}, {0.0, 1.0})
        self.assertEqual(counts, {"dijkstra": 2, "hazard-aware": 4, "qubo-sa": 8})
        self.assertEqual(len(plan), 14)
        self.assertEqual(len({run.run_id for run in plan}), len(plan))
        self.assertEqual(plan, plan_study(definition))

    def test_definition_rejects_empty_duplicate_and_invalid_dimensions(self):
        invalid = (
            {"algorithms": ()},
            {"seeds": (1, 1)},
            {"map_sizes": ((1, 8),)},
            {"algorithms": ("unknown",)},
            {"hazard_weights": (-1.0,)},
            {"qubo_sweeps": (0,)},
        )
        for changes in invalid:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                StudyDefinition(**changes)


class StudyRunnerTests(unittest.TestCase):
    def test_study_writes_raw_summary_manifest_and_resumes(self):
        definition = StudyDefinition(
            map_sizes=((2, 2),),
            agent_counts=(4,),
            seeds=(11, 12),
            disaster_profiles=("none",),
            algorithms=("dijkstra",),
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "tiny.csv"
            first = run_study(definition, output)
            raw_before_resume = output.read_text(encoding="utf-8")
            second = run_study(definition, output)
            raw_after_resume = output.read_text(encoding="utf-8")

            records = list(csv.DictReader(io.StringIO(raw_after_resume)))
            summaries = list(
                csv.DictReader(
                    io.StringIO(first.summary_results.read_text(encoding="utf-8"))
                )
            )
            manifest = json.loads(first.manifest.read_text(encoding="utf-8"))

        self.assertEqual(first.planned_runs, 2)
        self.assertEqual(first.completed_runs, 2)
        self.assertEqual(first.skipped_runs, 0)
        self.assertEqual(second.completed_runs, 2)
        self.assertEqual(second.skipped_runs, 2)
        self.assertEqual(len(records), 2)
        self.assertEqual(len({record["run_id"] for record in records}), 2)
        self.assertEqual(raw_before_resume, raw_after_resume)
        self.assertEqual(summaries[0]["replicates"], "2")
        self.assertEqual(summaries[0]["complete_evacuations"], "2")
        self.assertIn("p95_evacuation_time_mean", summaries[0])
        self.assertIn("completion_time_gini_mean", summaries[0])
        self.assertIn("mean_edge_utilization_mean", summaries[0])
        self.assertEqual(manifest["planned_runs"], 2)
        self.assertEqual(manifest["completed_runs"], 2)
        self.assertEqual(manifest["definition"]["seeds"], [11, 12])

    def test_no_resume_protects_existing_results(self):
        definition = StudyDefinition(
            map_sizes=((2, 2),),
            agent_counts=(1,),
            seeds=(1,),
            disaster_profiles=("none",),
            algorithms=("dijkstra",),
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "existing.csv"
            output.write_text("do not overwrite\n", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                run_study(definition, output, resume=False)
            self.assertEqual(output.read_text(encoding="utf-8"), "do not overwrite\n")


class StudyCliTests(unittest.TestCase):
    def invoke(self, *arguments: str) -> tuple[str, str]:
        stdout, stderr = io.StringIO(), io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            study_main(list(arguments))
        return stdout.getvalue(), stderr.getvalue()

    def test_dry_run_expands_requested_grid_without_writing(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "planned.csv"
            stdout, stderr = self.invoke(
                "--dry-run",
                "--maps", "3x3",
                "--agents", "5", "10",
                "--seeds", "1", "2",
                "--disasters", "none",
                "--algorithms", "dijkstra", "hazard-aware",
                "--hazard-weights", "0", "1",
                "--output", str(output),
            )
            payload = json.loads(stdout)

        self.assertEqual(payload["planned_runs"], 12)
        self.assertFalse(output.exists())
        self.assertEqual(stderr, "")

    def test_cli_rejects_duplicate_values_and_bad_map(self):
        for arguments in (
            ["--dry-run", "--seeds", "1", "1"],
            ["--dry-run", "--maps", "8"],
            ["--dry-run", "--maps", "1x8"],
        ):
            with self.subTest(arguments=arguments):
                with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                    study_main(arguments)
                self.assertIn(error.exception.code, (1, 2))

    def test_default_router_list_is_current(self):
        definition = StudyDefinition()
        self.assertEqual(definition.algorithms, ROUTER_NAMES)


if __name__ == "__main__":
    unittest.main()
