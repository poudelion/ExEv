"""Deterministic disaster events and synthetic Stage 3 hazard profiles."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Protocol

from .models import Shelter
from .network import CityNetwork


DISASTER_PROFILES = ("none", "road-closure", "fire", "flood")


def _validate_tick(tick: int) -> None:
    if not isinstance(tick, int) or tick < 0:
        raise ValueError("event tick must be a non-negative integer")


def _validate_intensity(intensity: float) -> None:
    if not isfinite(intensity) or intensity < 0:
        raise ValueError("hazard intensity must be finite and non-negative")


class DisasterEvent(Protocol):
    tick: int

    def validate(self, network: CityNetwork, shelters: dict[str, Shelter]) -> None: ...

    def apply(self, network: CityNetwork, shelters: dict[str, Shelter]) -> None: ...


@dataclass(frozen=True, slots=True)
class RoadStatusEvent:
    tick: int
    source: str
    target: str
    blocked: bool = True
    bidirectional: bool = False

    def __post_init__(self) -> None:
        _validate_tick(self.tick)

    def validate(self, network: CityNetwork, shelters: dict[str, Shelter]) -> None:
        if not network.has_edge(self.source, self.target):
            raise ValueError(f"unknown road {self.source}->{self.target}")
        if self.bidirectional and not network.has_edge(self.target, self.source):
            raise ValueError(f"unknown reverse road {self.target}->{self.source}")

    def apply(self, network: CityNetwork, shelters: dict[str, Shelter]) -> None:
        network.set_blocked(self.source, self.target, self.blocked)
        if self.bidirectional and self.source != self.target:
            network.set_blocked(self.target, self.source, self.blocked)


@dataclass(frozen=True, slots=True)
class EdgeHazardEvent:
    tick: int
    source: str
    target: str
    intensity: float
    bidirectional: bool = False

    def __post_init__(self) -> None:
        _validate_tick(self.tick)
        _validate_intensity(self.intensity)

    def validate(self, network: CityNetwork, shelters: dict[str, Shelter]) -> None:
        if not network.has_edge(self.source, self.target):
            raise ValueError(f"unknown road {self.source}->{self.target}")
        if self.bidirectional and not network.has_edge(self.target, self.source):
            raise ValueError(f"unknown reverse road {self.target}->{self.source}")

    def apply(self, network: CityNetwork, shelters: dict[str, Shelter]) -> None:
        network.set_edge_hazard(self.source, self.target, self.intensity)
        if self.bidirectional and self.source != self.target:
            network.set_edge_hazard(self.target, self.source, self.intensity)


@dataclass(frozen=True, slots=True)
class NodeHazardEvent:
    tick: int
    node: str
    intensity: float

    def __post_init__(self) -> None:
        _validate_tick(self.tick)
        _validate_intensity(self.intensity)

    def validate(self, network: CityNetwork, shelters: dict[str, Shelter]) -> None:
        if self.node not in network.nodes:
            raise ValueError(f"unknown node {self.node}")

    def apply(self, network: CityNetwork, shelters: dict[str, Shelter]) -> None:
        network.set_node_hazard(self.node, self.intensity)


@dataclass(frozen=True, slots=True)
class ShelterCapacityEvent:
    tick: int
    node: str
    capacity: int

    def __post_init__(self) -> None:
        _validate_tick(self.tick)
        if not isinstance(self.capacity, int) or self.capacity < 0:
            raise ValueError("shelter capacity must be a non-negative integer")

    def validate(self, network: CityNetwork, shelters: dict[str, Shelter]) -> None:
        if self.node not in shelters:
            raise ValueError(f"unknown shelter {self.node}")

    def apply(self, network: CityNetwork, shelters: dict[str, Shelter]) -> None:
        shelters[self.node].set_capacity(self.capacity)


class DisasterSchedule:
    """One-shot, stably ordered schedule of deterministic events."""

    def __init__(self, events: list[DisasterEvent] | tuple[DisasterEvent, ...] = ()) -> None:
        indexed = list(enumerate(events))
        self._events = tuple(
            event for _, event in sorted(indexed, key=lambda item: (item[1].tick, item[0]))
        )
        self._cursor = 0
        self.events_processed = 0

    @property
    def has_pending(self) -> bool:
        return self._cursor < len(self._events)

    @property
    def event_count(self) -> int:
        return len(self._events)

    def validate(self, network: CityNetwork, shelters: dict[str, Shelter]) -> None:
        for event in self._events:
            event.validate(network, shelters)

    def apply_tick(
        self, tick: int, network: CityNetwork, shelters: dict[str, Shelter]
    ) -> int:
        if self.has_pending and self._events[self._cursor].tick < tick:
            raise RuntimeError("disaster schedule skipped an event tick")
        applied = 0
        while self.has_pending and self._events[self._cursor].tick == tick:
            self._events[self._cursor].apply(network, shelters)
            self._cursor += 1
            self.events_processed += 1
            applied += 1
        return applied


def grid_disaster_schedule(
    width: int, height: int, agent_count: int, profile: str = "none"
) -> DisasterSchedule:
    """Build a reproducible synthetic profile for a rectangular grid.

    Fire and flood are deliberately simple stress-test generators, not physical
    hazard models. They make exposure and topology change over time so routing
    algorithms can be compared under identical conditions.
    """
    if profile not in DISASTER_PROFILES:
        raise ValueError(f"unknown disaster profile: {profile}")
    if width < 2 or height < 2:
        raise ValueError("grid dimensions must be at least 2")
    if agent_count < 0:
        raise ValueError("agent_count must be non-negative")
    if profile == "none":
        return DisasterSchedule()

    middle_y = height // 2
    left_x = max(0, width // 2 - 1)
    right_x = left_x + 1
    bottleneck_closures = tuple(row for row in range(height) if row != middle_y)

    def crossing(row: int) -> tuple[str, str]:
        return f"{left_x},{row}", f"{right_x},{row}"

    if profile == "road-closure":
        closures = [
            RoadStatusEvent(10, *crossing(row), blocked=True, bidirectional=True)
            for row in bottleneck_closures
        ]
        reopenings = [
            RoadStatusEvent(30, *crossing(row), blocked=False, bidirectional=True)
            for row in bottleneck_closures
        ]
        return DisasterSchedule([*closures, *reopenings])

    events: list[DisasterEvent] = []
    if profile == "fire":
        origin_x, origin_y = 0, 0
        arrival: dict[tuple[int, int], int] = {}
        for y in range(height):
            for x in range(width):
                tick = 4 * (abs(x - origin_x) + abs(y - origin_y))
                arrival[(x, y)] = tick
                events.append(NodeHazardEvent(tick, f"{x},{y}", 1.0))
        for y in range(height):
            for x in range(width):
                if x + 1 < width:
                    tick = max(arrival[(x, y)], arrival[(x + 1, y)])
                    events.append(EdgeHazardEvent(
                        tick, f"{x},{y}", f"{x + 1},{y}", 0.7, bidirectional=True
                    ))
                if y + 1 < height:
                    tick = max(arrival[(x, y)], arrival[(x, y + 1)])
                    events.append(EdgeHazardEvent(
                        tick, f"{x},{y}", f"{x},{y + 1}", 0.7, bidirectional=True
                    ))
        # As an upper-left fire reaches the central corridor, crossings close
        # progressively from top to bottom. The bottom crossing remains open as
        # the final capacity-constrained detour.
        for row in range(height - 1):
            closure_tick = max(8, arrival[(left_x, row)])
            events.append(RoadStatusEvent(
                closure_tick, *crossing(row), blocked=True, bidirectional=True
            ))
        return DisasterSchedule(events)

    # Flood rises from the bottom, temporarily removing the lower shelter and
    # progressively increasing exposure on rows and their horizontal roads.
    shelter_capacity = (agent_count + 1) // 2
    lower_shelter = f"{width - 1},{height - 1}"
    events.extend([
        ShelterCapacityEvent(0, lower_shelter, 0),
        ShelterCapacityEvent(30, lower_shelter, shelter_capacity),
    ])
    for y in reversed(range(height)):
        tick = 5 * (height - 1 - y)
        for x in range(width):
            events.append(NodeHazardEvent(tick, f"{x},{y}", 0.8))
            if x + 1 < width:
                events.append(EdgeHazardEvent(
                    tick, f"{x},{y}", f"{x + 1},{y}", 0.5, bidirectional=True
                ))
    # Flooded corridor crossings close row by row. The top crossing remains open
    # as the final evacuation route in this synthetic profile.
    for row in range(1, height):
        closure_tick = 5 * (height - 1 - row)
        events.append(RoadStatusEvent(
            closure_tick, *crossing(row), blocked=True, bidirectional=True
        ))
    return DisasterSchedule(events)
