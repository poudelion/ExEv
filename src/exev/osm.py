"""Dependency-free OpenStreetMap XML import for real-road experiments."""

from __future__ import annotations

import random
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from math import asin, cos, isfinite, radians, sin, sqrt
from pathlib import Path

from .models import Agent, Shelter
from .network import CityNetwork

EARTH_RADIUS_METERS = 6_371_008.8
EXCLUDED_HIGHWAYS = {"construction", "proposed", "abandoned", "raceway", "services"}
DEFAULT_SPEED_KPH = {
    "motorway": 100.0, "trunk": 80.0, "primary": 50.0,
    "secondary": 40.0, "tertiary": 35.0, "residential": 25.0,
    "service": 15.0, "living_street": 10.0, "pedestrian": 5.0,
    "footway": 5.0, "path": 5.0, "steps": 3.0,
}
DEFAULT_LANES = {
    "motorway": 2, "trunk": 2, "primary": 2, "secondary": 2,
    "tertiary": 2, "residential": 1, "service": 1,
}


@dataclass(frozen=True, slots=True)
class OSMImportStats:
    source_nodes: int
    source_ways: int
    road_ways: int
    imported_nodes: int
    directed_edges: int
    bounds: tuple[float, float, float, float]


@dataclass(slots=True)
class OSMScenario:
    network: CityNetwork
    agents: list[Agent]
    shelters: list[Shelter]
    stats: OSMImportStats


def _haversine(first: tuple[float, float], second: tuple[float, float]) -> float:
    lat1, lon1 = map(radians, first)
    lat2, lon2 = map(radians, second)
    dlat, dlon = lat2 - lat1, lon2 - lon1
    value = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_METERS * asin(sqrt(value))


def _speed_meters_per_tick(value: str | None, highway: str) -> float:
    if value:
        match = re.match(r"^\s*(\d+(?:\.\d+)?)\s*(mph|km/h|kph)?\s*$", value.lower())
        if match:
            speed = float(match.group(1))
            if match.group(2) == "mph":
                speed *= 1.609344
            if speed > 0:
                return speed / 3.6
    return DEFAULT_SPEED_KPH.get(highway, 20.0) / 3.6


def _capacity(tags: dict[str, str], highway: str) -> int:
    lane_text = tags.get("lanes", "").split(";")[0]
    try:
        lanes = max(1, int(float(lane_text)))
    except ValueError:
        lanes = DEFAULT_LANES.get(highway, 1)
    return lanes * 20


def _direction(tags: dict[str, str], highway: str) -> int:
    oneway = tags.get("oneway", "").lower()
    if oneway == "-1":
        return -1
    if oneway in {"yes", "true", "1"}:
        return 1
    if oneway in {"no", "false", "0"}:
        return 0
    return 1 if highway == "motorway" or tags.get("junction") == "roundabout" else 0


def load_osm_scenario(
    path: Path,
    *,
    shelter_nodes: list[str],
    origin_nodes: list[str],
    agent_count: int,
    seed: int = 7,
    capacity_multiplier: float = 1.0,
) -> OSMScenario:
    """Load an OSM XML road graph and create a reproducible population."""
    if agent_count < 0:
        raise ValueError("agent_count must be non-negative")
    if not isfinite(capacity_multiplier) or capacity_multiplier <= 0:
        raise ValueError("capacity_multiplier must be a positive finite number")
    if not shelter_nodes or len(set(shelter_nodes)) != len(shelter_nodes):
        raise ValueError("shelter_nodes must be non-empty and unique")
    if not origin_nodes or len(set(origin_nodes)) != len(origin_nodes):
        raise ValueError("origin_nodes must be non-empty and unique")
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError as exc:
        raise ValueError(f"invalid OSM XML: {exc}") from exc
    if root.tag.rsplit("}", 1)[-1] != "osm":
        raise ValueError("OSM document root must be <osm>")

    coordinates: dict[str, tuple[float, float]] = {}
    for element in root:
        if element.tag.rsplit("}", 1)[-1] != "node":
            continue
        node_id = element.get("id")
        try:
            latitude, longitude = float(element.get("lat", "")), float(element.get("lon", ""))
        except ValueError as exc:
            raise ValueError(f"node {node_id!r} has invalid coordinates") from exc
        if not node_id or not isfinite(latitude) or not isfinite(longitude):
            raise ValueError("OSM nodes require finite id, latitude, and longitude")
        coordinates[node_id] = (latitude, longitude)
    if not coordinates:
        raise ValueError("OSM document contains no nodes")

    network = CityNetwork()
    road_ways = 0
    source_ways = 0
    used_nodes: set[str] = set()
    for way in root:
        if way.tag.rsplit("}", 1)[-1] != "way":
            continue
        source_ways += 1
        tags = {
            child.get("k", ""): child.get("v", "")
            for child in way
            if child.tag.rsplit("}", 1)[-1] == "tag"
        }
        highway = tags.get("highway")
        if (
            not highway or highway in EXCLUDED_HIGHWAYS
            or tags.get("area") == "yes" or tags.get("access") in {"no", "private"}
        ):
            continue
        references = [
            child.get("ref", "") for child in way
            if child.tag.rsplit("}", 1)[-1] == "nd"
        ]
        if len(references) < 2:
            continue
        missing = [reference for reference in references if reference not in coordinates]
        if missing:
            raise ValueError(f"way {way.get('id')!r} references missing node {missing[0]!r}")
        road_ways += 1
        direction = _direction(tags, highway)
        speed = _speed_meters_per_tick(tags.get("maxspeed"), highway)
        capacity = max(1, round(_capacity(tags, highway) * capacity_multiplier))
        pairs = list(zip(references, references[1:]))
        if direction == -1:
            pairs = [(target, source) for source, target in reversed(pairs)]
        for source, target in pairs:
            if source == target:
                continue
            distance = _haversine(coordinates[source], coordinates[target])
            if distance <= 0:
                continue
            network.add_edge(
                source, target, distance, capacity, speed,
                bidirectional=direction == 0,
            )
            used_nodes.update((source, target))
    if not network.edges:
        raise ValueError("OSM document contains no supported highway ways")

    mean_latitude = sum(coordinates[node][0] for node in used_nodes) / len(used_nodes)
    mean_longitude = sum(coordinates[node][1] for node in used_nodes) / len(used_nodes)
    cosine = cos(radians(mean_latitude))
    for node in used_nodes:
        latitude, longitude = coordinates[node]
        x = EARTH_RADIUS_METERS * radians(longitude - mean_longitude) * cosine
        y = -EARTH_RADIUS_METERS * radians(latitude - mean_latitude)
        network.set_position(node, x, y)

    unknown = (set(shelter_nodes) | set(origin_nodes)) - network.nodes
    if unknown:
        raise ValueError(f"selected nodes are not part of imported roads: {', '.join(sorted(unknown))}")
    capacity = (agent_count + len(shelter_nodes) - 1) // len(shelter_nodes)
    shelters = [Shelter(node, capacity) for node in shelter_nodes]
    rng = random.Random(seed)
    speeds = (0.65, 0.8, 1.0, 1.15)
    agents = [
        Agent(index, rng.choice(origin_nodes), speed=rng.choice(speeds))
        for index in range(agent_count)
    ]
    latitudes = [coordinates[node][0] for node in used_nodes]
    longitudes = [coordinates[node][1] for node in used_nodes]
    stats = OSMImportStats(
        source_nodes=len(coordinates), source_ways=source_ways,
        road_ways=road_ways, imported_nodes=len(network.nodes),
        directed_edges=len(network.edges),
        bounds=(min(latitudes), min(longitudes), max(latitudes), max(longitudes)),
    )
    return OSMScenario(network, agents, shelters, stats)
