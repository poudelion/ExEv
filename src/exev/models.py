from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from math import isfinite


class AgentStatus(str, Enum):
    WAITING = "waiting"
    TRAVELING = "traveling"
    EVACUATED = "evacuated"
    STRANDED = "stranded"


@dataclass(frozen=True, slots=True)
class Edge:
    source: str
    target: str
    distance: float
    capacity: int
    speed_limit: float = 1.0
    blocked: bool = False

    def __post_init__(self) -> None:
        if not isfinite(self.distance) or self.distance <= 0:
            raise ValueError("edge distance must be positive")
        if not isinstance(self.capacity, int) or self.capacity <= 0:
            raise ValueError("edge capacity must be positive")
        if not isfinite(self.speed_limit) or self.speed_limit <= 0:
            raise ValueError("edge speed_limit must be positive")


@dataclass(slots=True)
class Shelter:
    node: str
    capacity: int
    occupants: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.capacity, int) or self.capacity < 0:
            raise ValueError("shelter capacity must be a non-negative integer")
        if not isinstance(self.occupants, int) or not 0 <= self.occupants <= self.capacity:
            raise ValueError("shelter occupants must be between zero and capacity")

    @property
    def available(self) -> int:
        return max(0, self.capacity - self.occupants)

    def admit(self) -> bool:
        if self.available == 0:
            return False
        self.occupants += 1
        return True

    def set_capacity(self, capacity: int) -> None:
        if not isinstance(capacity, int) or capacity < 0:
            raise ValueError("shelter capacity must be a non-negative integer")
        if capacity < self.occupants:
            raise ValueError("shelter capacity cannot be lower than current occupancy")
        self.capacity = capacity


@dataclass(slots=True)
class Agent:
    id: int
    origin: str
    speed: float = 1.0
    status: AgentStatus = AgentStatus.WAITING
    current_node: str = field(init=False)
    route: list[str] = field(default_factory=list)
    route_index: int = 0
    edge: tuple[str, str] | None = None
    edge_progress: float = 0.0
    started_at: int | None = None
    evacuated_at: int | None = None
    distance_traveled: float = 0.0
    waiting_time: int = 0
    total_waiting_time: int = 0
    hazard_exposure: float = 0.0

    def __post_init__(self) -> None:
        if not isfinite(self.speed) or self.speed <= 0:
            raise ValueError("agent speed must be positive")
        self.current_node = self.origin
