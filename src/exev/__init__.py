"""ExEv: a solver-agnostic evacuation research simulator."""

__version__ = "0.3.1"

from .disasters import (
    DISASTER_PROFILES,
    DisasterSchedule,
    EdgeHazardEvent,
    NodeHazardEvent,
    RoadStatusEvent,
    ShelterCapacityEvent,
    grid_disaster_schedule,
)
from .models import Agent, AgentStatus, Edge, Shelter
from .network import CityNetwork
from .flow import MinCostFlowRouter
from .routing import (
    AStarRouter,
    CongestionAwareRouter,
    DijkstraRouter,
    HazardAwareRouter,
    RoutingStats,
    create_router,
)
from .simulation import EvacuationSimulation, SimulationConfig, SimulationResult

__all__ = [
    "Agent",
    "AgentStatus",
    "CityNetwork",
    "DijkstraRouter",
    "DISASTER_PROFILES",
    "DisasterSchedule",
    "EdgeHazardEvent",
    "AStarRouter",
    "CongestionAwareRouter",
    "Edge",
    "EvacuationSimulation",
    "HazardAwareRouter",
    "Shelter",
    "SimulationConfig",
    "SimulationResult",
    "MinCostFlowRouter",
    "NodeHazardEvent",
    "RoadStatusEvent",
    "RoutingStats",
    "ShelterCapacityEvent",
    "create_router",
    "grid_disaster_schedule",
]
