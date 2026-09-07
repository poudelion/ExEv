from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from exev.experiments import ExperimentConfig
from exev.ui_cli import main
from exev.visualization import (
    build_dashboard_payload,
    build_osm_dashboard_payload,
    build_visualization_run,
)


class VisualizationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = ExperimentConfig(
            width=4, height=3, agent_count=20, seed=7,
            max_ticks=200, disaster_profile="fire",
        )

    def test_run_contains_topology_frames_and_final_state(self) -> None:
        run = build_visualization_run(
            self.config, "hazard-aware", frame_interval=3, max_rendered_agents=5
        )
        self.assertEqual(run["algorithm"], "hazard-aware")
        self.assertEqual(len(run["topology"]["nodes"]), 12)
        self.assertEqual(run["frames"][0]["tick"], 0)
        self.assertEqual(run["frames"][-1]["tick"], run["result"]["elapsed_ticks"])
        self.assertLessEqual(max(len(frame["agents"]) for frame in run["frames"]), 5)
        final = run["frames"][-1]["counts"]
        self.assertEqual(final["evacuated"] + final["stranded"], 20)
        self.assertTrue(any(frame["nodes"] for frame in run["frames"]))
        self.assertTrue(all(len(frame["shelters"]) == 2 for frame in run["frames"]))
        final_shelters = run["frames"][-1]["shelters"]
        self.assertEqual(sum(item["occupants"] for item in final_shelters), 20)
        self.assertIn("initial route", run["frames"][-1]["routing_reasons"])

    def test_payload_compares_fresh_algorithms(self) -> None:
        payload = build_dashboard_payload(
            self.config, ["dijkstra", "hazard-aware"], frame_interval=4
        )
        self.assertEqual(payload["schema_version"], 1)
        self.assertEqual(payload["metadata"]["package_version"], "0.8.0")
        self.assertEqual([run["algorithm"] for run in payload["runs"]],
                         ["dijkstra", "hazard-aware"])
        with self.assertRaises(ValueError):
            build_dashboard_payload(self.config, ["dijkstra", "dijkstra"])

    def test_osm_payload_animates_imported_geometry(self) -> None:
        source = Path(__file__).parents[1] / "examples" / "tiny-map.osm"
        payload = build_osm_dashboard_payload(
            source, ["dijkstra", "astar"], shelter_nodes=["3"],
            origin_nodes=["1", "4"], agent_count=12, frame_interval=20,
        )
        self.assertEqual(payload["map_type"], "openstreetmap")
        self.assertEqual(payload["configuration"]["import"]["imported_nodes"], 4)
        self.assertEqual(len(payload["runs"]), 2)
        self.assertTrue(all(run["result"]["evacuated"] == 12 for run in payload["runs"]))
        fire = build_osm_dashboard_payload(
            source, ["hazard-aware"], shelter_nodes=["3"],
            origin_nodes=["1", "4"], agent_count=4,
            disaster_profile="fire", frame_interval=20,
        )
        self.assertEqual(fire["configuration"]["disaster_profile"], "fire")
        self.assertGreater(fire["runs"][0]["result"]["events_processed"], 0)
        self.assertGreater(fire["runs"][0]["result"]["total_hazard_exposure"], 0)

    def test_cli_exports_dashboard_json(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "dashboard.json"
            main([
                "--agents", "10", "--width", "3", "--height", "3",
                "--disaster", "none", "--routers", "dijkstra",
                "--export", str(output),
            ])
            payload = json.loads(output.read_text())
            self.assertEqual(len(payload["runs"]), 1)
            self.assertEqual(payload["runs"][0]["result"]["total_agents"], 10)


if __name__ == "__main__":
    unittest.main()
