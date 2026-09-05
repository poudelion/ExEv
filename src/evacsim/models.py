from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


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
        if self.distance <= 0:
            raise ValueError("edge distance must be positive")
        if self.capacity <= 0:
            raise ValueError("edge capacity must be positive")
        if self.speed_limit <= 0:
            raise ValueError("edge speed_limit must be positive")


@dataclass(slots=True)
class Shelter:
    node: str
    capacity: int
    occupants: int = 0

    @property
    def available(self) -> int:
        return max(0, self.capacity - self.occupants)

    def admit(self) -> bool:
        if self.available == 0:
            return False
        self.occupants += 1
        return True


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

    def __post_init__(self) -> None:
        if self.speed <= 0:
            raise ValueError("agent speed must be positive")
        self.current_node = self.origin
