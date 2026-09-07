from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from exev.osm import load_osm_scenario
from exev.osm_cli import main


EXAMPLE = Path(__file__).parents[1] / "examples" / "tiny-map.osm"


class OSMImportTests(unittest.TestCase):
    def test_import_preserves_directions_distances_and_capacity(self) -> None:
        scenario = load_osm_scenario(
            EXAMPLE,
            shelter_nodes=["3"],
            origin_nodes=["1", "4"],
            agent_count=10,
            seed=7,
        )
        network = scenario.network
        self.assertEqual(scenario.stats.source_nodes, 4)
        self.assertEqual(scenario.stats.road_ways, 2)
        self.assertEqual(scenario.stats.imported_nodes, 4)
        self.assertTrue(network.has_edge("1", "2"))
        self.assertTrue(network.has_edge("2", "1"))
        self.assertTrue(network.has_edge("4", "2"))
        self.assertFalse(network.has_edge("2", "4"))
        self.assertGreater(network.edge("1", "2").distance, 80)
        self.assertEqual(network.edge("1", "2").capacity, 40)
        self.assertAlmostEqual(network.edge("1", "2").speed_limit, 11.176, places=3)
        self.assertTrue({agent.origin for agent in scenario.agents} <= {"1", "4"})
        self.assertEqual(scenario.shelters[0].capacity, 10)
        doubled = load_osm_scenario(
            EXAMPLE, shelter_nodes=["3"], origin_nodes=["1", "4"],
            agent_count=10, capacity_multiplier=2.0,
        )
        self.assertEqual(doubled.network.edge("1", "2").capacity, 80)

    def test_invalid_documents_and_selections_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "capacity_multiplier"):
            load_osm_scenario(
                EXAMPLE, shelter_nodes=["3"], origin_nodes=["1"],
                agent_count=1, capacity_multiplier=0,
            )
        with self.assertRaisesRegex(ValueError, "selected nodes"):
            load_osm_scenario(
                EXAMPLE, shelter_nodes=["missing"], origin_nodes=["1"], agent_count=1
            )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.osm"
            path.write_text("<osm><node>", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "invalid OSM XML"):
                load_osm_scenario(
                    path, shelter_nodes=["1"], origin_nodes=["1"], agent_count=1
                )

    def test_cli_runs_real_map_scenario(self) -> None:
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            main([
                "--input", str(EXAMPLE), "--shelter-nodes", "3",
                "--origin-nodes", "1", "4", "--agents", "12",
                "--router", "dijkstra",
            ])
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["metadata"]["package_version"], "0.8.0")
        self.assertEqual(payload["metadata"]["model_version"], "stage8-osm-v1")
        self.assertEqual(payload["configuration"]["distance_unit"], "meters")
        self.assertEqual(payload["result"]["evacuated"], 12)


if __name__ == "__main__":
    unittest.main()
