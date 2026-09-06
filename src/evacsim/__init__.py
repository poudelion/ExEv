"""EvacSim: a solver-agnostic evacuation research simulator."""

__version__ = "0.2.0"

from .models import Agent, AgentStatus, Edge, Shelter
from .network import CityNetwork
from .flow import MinCostFlowRouter
from .routing import (
    AStarRouter,
    CongestionAwareRouter,
    DijkstraRouter,
    RoutingStats,
    create_router,
)
from .simulation import EvacuationSimulation, SimulationConfig, SimulationResult

__all__ = [
    "Agent",
    "AgentStatus",
    "CityNetwork",
    "DijkstraRouter",
    "AStarRouter",
    "CongestionAwareRouter",
    "Edge",
    "EvacuationSimulation",
    "Shelter",
    "SimulationConfig",
    "SimulationResult",
    "MinCostFlowRouter",
    "RoutingStats",
    "create_router",
]
