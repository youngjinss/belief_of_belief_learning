"""
Experiment 3: MiniGrid-LockedRoom-v0 with A* Agent Integration

This module implements A* agent for the MiniGrid-LockedRoom-v0 environment,
adapted from the ToMnetF_impl AgentStar class.

Components:
- astar_agent.py: A* agent implementation for MiniGrid
- generate_data.py: Data generation pipeline
- config.py: Configuration management
- run_experiment.py: Main experiment runner

Usage:
    python run_experiment.py test          # Run single test episode
    python run_experiment.py generate      # Generate dataset
    python run_experiment.py evaluate      # Analyze existing dataset
    python run_experiment.py all           # Run all modes
"""

from .config import Config
from .agent import MiniGridAStarAgent

__all__ = ["Config", "MiniGridAStarAgent"]
