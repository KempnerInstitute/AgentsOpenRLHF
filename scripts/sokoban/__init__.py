"""
Sokoban utilities and agent components.

This module provides:
- Constants (CELL_TYPES, Action)
- Environment (SokobanEnv, SokobanConfig)
- Data generation (SokobanGenerator, SokobanDeadlockGenerator)
- Evaluation (SokobanEvaluator)
- Utilities (parsing, prompts, etc.)
"""

from .sokoban_constants import (
    Action,
    CELL_TYPES,
    GRID_CONFIG,
    # Backward compatibility
    WALL,
    EMPTY,
    TARGET,
    BOX_ON_TARGET,
    BOX,
    PLAYER,
    PLAYER_ON_TARGET,
)

from .sokoban_data_gen import (
    SokobanConfig,
    SokobanEnv,
    SokobanSolver,
    SokobanGenerator,
    SokobanDeadlockGenerator,
    TrainingExample,
    TrainingTrace,
)

from .sokoban_evaluator import (
    SokobanEvaluator,
    aggregate_parallel_results,
    save_parallel_results,
)

# Import utility functions
from . import sokoban_utils

__all__ = [
    # Constants
    "Action",
    "CELL_TYPES",
    "GRID_CONFIG",
    "WALL",
    "EMPTY",
    "TARGET",
    "BOX_ON_TARGET",
    "BOX",
    "PLAYER",
    "PLAYER_ON_TARGET",
    # Environment
    "SokobanConfig",
    "SokobanEnv",
    "SokobanSolver",
    # Data generation
    "SokobanGenerator",
    "SokobanDeadlockGenerator",
    "TrainingExample",
    "TrainingTrace",
    # Evaluation
    "SokobanEvaluator",
    "aggregate_parallel_results",
    "save_parallel_results",
]
