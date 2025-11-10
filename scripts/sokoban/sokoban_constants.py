"""
Centralized configuration for Sokoban.
Single source of truth for:
- Cell type constants (WALL, EMPTY, BOX, etc.)
- Action definitions (UP, DOWN, LEFT, RIGHT)
- Grid symbol mappings and formatting
"""
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List


# Cell type constants for grid representation
@dataclass(frozen=True)
class CellTypes:
    """Cell type constants for grid representation.
    These values correspond to the internal representation in gym-sokoban.
    """
    WALL: int = 0
    EMPTY: int = 1
    TARGET: int = 2
    BOX_ON_TARGET: int = 3
    BOX: int = 4
    PLAYER: int = 5
    PLAYER_ON_TARGET: int = 6

# Global singleton instance
CELL_TYPES = CellTypes()

# Backward compatibility - individual constants
WALL = CELL_TYPES.WALL
EMPTY = CELL_TYPES.EMPTY
TARGET = CELL_TYPES.TARGET
BOX_ON_TARGET = CELL_TYPES.BOX_ON_TARGET
BOX = CELL_TYPES.BOX
PLAYER = CELL_TYPES.PLAYER
PLAYER_ON_TARGET = CELL_TYPES.PLAYER_ON_TARGET


class Action(Enum):
    """Sokoban actions with proper enum structure"""

    UP = ("up", 1)
    DOWN = ("down", 2)
    LEFT = ("left", 3)
    RIGHT = ("right", 4)

    def __init__(self, action_name: str, action_id: int):
        self.action_name = action_name
        self.action_id = action_id

    @classmethod
    def from_id(cls, action_id: int) -> "Action":
        """Get Action enum from action ID"""
        id_map = {action.action_id: action for action in cls}
        return id_map.get(action_id)

    @classmethod
    def from_name(cls, action_name: str) -> "Action":
        """Get Action enum from action name (case-insensitive)"""
        name_lower = action_name.lower()
        for action in cls:
            if action.action_name.lower() == name_lower:
                return action
        return None

    @classmethod
    def all_actions(cls) -> List["Action"]:
        """Get list of all actions"""
        return list(cls)


@dataclass
class SokobanGridConfig:
    """Single source of truth for Sokoban grid symbols and mappings"""

    # Primary mapping: environment int -> display symbol
    ENV_TO_SYMBOL: Dict[int, str] = None

    # Symbol descriptions for prompts
    SYMBOL_TO_NAME: Dict[str, str] = None

    def __post_init__(self):
        if self.ENV_TO_SYMBOL is None:
            self.ENV_TO_SYMBOL = {
                0: "#",  # wall
                1: "_",  # empty
                2: "O",  # target
                3: "√",  # box on target
                4: "X",  # box
                5: "P",  # player
                6: "S",  # player on target
            }

        if self.SYMBOL_TO_NAME is None:
            self.SYMBOL_TO_NAME = {
                "#": "wall",
                "_": "empty",
                "O": "target",
                "√": "box on target",
                "X": "box",
                "P": "player",
                "S": "player on target",
            }

    @property
    def symbol_to_env(self) -> Dict[str, int]:
        """Derive reverse mapping: symbol -> environment int"""
        mapping = {symbol: env_id for env_id, symbol in self.ENV_TO_SYMBOL.items()}
        # Special case: "S" (player on target) maps to player value 5
        mapping["S"] = 5
        return mapping

    @property
    def env_to_symbol(self) -> Dict[int, str]:
        """Direct access to primary mapping"""
        return self.ENV_TO_SYMBOL

    @property
    def symbol_to_name(self) -> Dict[str, str]:
        """Direct access to symbol descriptions"""
        return self.SYMBOL_TO_NAME

    def get_legend_string(self) -> str:
        """Generate legend string for prompts"""
        return ", ".join(f"{symbol} = {name}" for symbol, name in self.SYMBOL_TO_NAME.items())


# Global singleton instance
GRID_CONFIG = SokobanGridConfig()
