"""
Utility functions for Sokoban agents and evaluation.

Provides helper functions for:
- Action parsing and validation
- Grid/state parsing and formatting
- Prompt generation
- Response parsing
"""

import re
import numpy as np
from typing import List, Optional, Tuple
from .sokoban_constants import Action, CELL_TYPES, GRID_CONFIG
from .sokoban_data_gen import SokobanConfig, SokobanEnv


def parse_action(response: str) -> Optional[Action]:
    """
    Extract single action from <answer> tag in response.

    Args:
        response: Model response string

    Returns:
        Action enum if valid action found, None otherwise
    """
    match = re.search(r"<answer>\s*(\w+)\s*</answer>", response, re.IGNORECASE)
    if match:
        action_name = match.group(1).strip()
        return Action.from_name(action_name)
    return None


def parse_action_sequence(response: str) -> List[int]:
    """
    Extract action sequence from model's <think> reasoning.

    Args:
        response: Model response string containing <think> tags

    Returns:
        List of action IDs extracted from "Step X: action_name" patterns
    """
    actions = []

    # Extract content within <think> tags
    think_match = re.search(
        r"<think>(.*?)</think>", response, re.DOTALL | re.IGNORECASE
    )
    if think_match:
        reasoning = think_match.group(1).strip()

        # Find all "Step X: action_name" patterns
        step_matches = re.findall(r"Step \d+: (\w+)", reasoning, re.IGNORECASE)

        for action_name in step_matches:
            action = Action.from_name(action_name.strip())
            if action:
                actions.append(action.action_id)

    return actions


def assert_valid_grid(grid_str: str) -> None:
    """
    Assert that grid string contains exactly one player.

    Args:
        grid_str: String representation of grid

    Raises:
        AssertionError: If grid doesn't have exactly 1 player
    """
    player_count = grid_str.count('P') + grid_str.count('S')
    assert player_count == 1, f"Grid must have exactly 1 player, found {player_count}"


def make_prompt(state_representation: str) -> str:
    """
    Generate standard Sokoban prompt for a given state.

    Args:
        state_representation: Text representation of current grid state

    Returns:
        Formatted prompt string with instructions and state
    """
    legend = GRID_CONFIG.get_legend_string()

    return f"""<|im_start|>user
Symbols:
{legend}

Rules:
- Push all boxes (X) onto targets (O).
- Avoid walls (#).

Valid answers:
<answer>up</answer> | <answer>down</answer> | <answer>left</answer> | <answer>right</answer>

Grid:
{state_representation}

What action should you take next?

Decide the next action:
Always output: <answer>[your answer]</answer> with no extra text. Strictly follow this format.<|im_end|>
<|im_start|>assistant
<think>
"""


def make_prompt_no_tool(state_str: str) -> str:
    return f"""
Grid:
{state_str}

What action should you take next?

Decide the next action:
Always output: <answer> [your answer] </answer> with no extra text.
Strictly follow this format. <|im_end|>
<|im_start|>assistant
<think>
"""


def extract_grid_from_prompt(prompt: str) -> Optional[str]:
    """
    Extract grid representation from model response if present.

    Args:
        prompt: Model response that may contain a grid

    Returns:
        Grid string if found, None otherwise
    """
    valid_symbols = {'#', '_', 'O', '√', 'X', 'P', 'S'}
    lines = prompt.split('\n')
    grid_lines = []
    in_grid = False

    for line in lines:
        stripped_line = line.strip()
        if not stripped_line:
            if in_grid:
                break
            continue

        # Check if line contains only valid Sokoban symbols
        cleaned_line = stripped_line.replace(' ', '')
        if cleaned_line and all(c in valid_symbols for c in cleaned_line):
            grid_lines.append(cleaned_line)
            in_grid = True
        elif in_grid:
            break

    return '\n'.join(grid_lines) if grid_lines else None


def parse_map_to_arrays(
    map_str: str
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[Tuple[int, int]]]:
    """
    Parse map string directly to numpy arrays for environment initialization.

    Args:
        map_str: Raw map string with newline-separated rows

    Returns:
        (room_state, room_fixed, player_pos): Parsed arrays and position, or (None, None, None) on error
    """
    # Split and normalize lines
    lines = [line.rstrip() for line in map_str.strip().split("\n") if line.strip()]
    if not lines:
        print("Error: Empty map string")
        return None, None, None

    height = len(lines)
    max_width = max(len(line) for line in lines)

    # Enforce square map
    if height != max_width:
        print(f"Error: Map is not square ({height}x{max_width})")
        return None, None, None

    side = height
    lines = [line.ljust(side, "_") for line in lines]  # Pad lines to square

    # Parse to arrays
    room_state = np.zeros((side, side), dtype=int)
    room_fixed = np.zeros((side, side), dtype=int)
    player_pos = None

    vocab_to_env = GRID_CONFIG.symbol_to_env

    for i in range(side):
        for j in range(side):
            char = lines[i][j]
            room_state[i, j] = vocab_to_env.get(char, CELL_TYPES.EMPTY)

            if char == "#":
                room_fixed[i, j] = CELL_TYPES.WALL
            elif char in ("O", "√", "S"):
                room_fixed[i, j] = CELL_TYPES.TARGET
            else:
                room_fixed[i, j] = CELL_TYPES.EMPTY

            if char in ("P", "S"):
                player_pos = (i, j)

    if player_pos is None:
        print("Error: Player position not found")
        return None, None, None

    return room_state, room_fixed, player_pos


def create_env_from_map(map_str: str, max_steps: int) -> Optional[SokobanEnv]:
    """Create and initialize environment from map string with safe state injection."""
    try:
        # Parse map string directly to arrays
        room_state, room_fixed, player_pos = parse_map_to_arrays(map_str)
        if room_state is None:
            return None

        # Get side length from room_state shape (maps are square)
        side = room_state.shape[0]

        # Count boxes from numpy array
        box_count = np.count_nonzero(room_state == CELL_TYPES.BOX) + \
                    np.count_nonzero(room_state == CELL_TYPES.BOX_ON_TARGET)
        box_count = max(box_count, 1)  # ensure at least 1 box

        # Create Sokoban config and env
        config = SokobanConfig(
            dim_room=(side, side),
            num_boxes=box_count,
            max_steps=max_steps,
        )
        env = SokobanEnv(config)
        env.reset()

        # Set state safely using in-place update
        env.set_state(room_state, room_fixed, player_pos)

        return env

    except Exception as e:
        print(f"Error creating environment from map: {e}")
        import traceback
        traceback.print_exc()
        return None
