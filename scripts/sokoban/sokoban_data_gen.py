import argparse
import copy
import json
import marshal
import random
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from tqdm import tqdm

import numpy as np
from gym_sokoban.envs.sokoban_env import SokobanEnv as GymSokobanEnv
from .sokoban_constants import (
    Action,
    GRID_CONFIG,
    WALL,
    EMPTY,
    TARGET,
    BOX_ON_TARGET,
    BOX,
    PLAYER,
    PLAYER_ON_TARGET
)

# Algorithm constants (tuning parameters for algorithms)
MAX_BFS_DEPTH = 100
U_TRAP_MAX_ATTEMPTS = 50
BOX_ORDERING_MAX_ATTEMPTS = 100
FROZEN_STATE_MAX_ATTEMPTS = 100


@dataclass
class TrainingTrace:
    """Single step in a training sequence"""
    state_representation: str
    action: Action
    reasoning: str
    step: int


@dataclass
class TrainingExample:
    """Complete training example with metadata"""

    initial_state: str
    trace: List[TrainingTrace]
    optimal_path: Optional[List[Action]] = None
    deadlock_type: Optional[str] = None

    @property
    def metadata(self) -> Dict[str, Any]:
        return {
            "grid_size": self._extract_grid_size(),
            "optimal_path_length": len(self.optimal_path) if self.optimal_path else 0,
            "num_boxes": self._count_boxes(),
            "action_name": self.trace[0].action.action_name if self.trace[0].action else None,
            "action_id": self.trace[0].action.action_id if self.trace[0].action else None,
            "deadlock_type": self.deadlock_type
        }

    def _extract_grid_size(self) -> Tuple[int, int]:
        if self.initial_state:
            lines = self.initial_state.strip().split("\n")
            return (len(lines), len(lines[0]) if lines else 0)
        return (0, 0)

    def _count_boxes(self) -> int:
        if self.initial_state:
            return self.initial_state.count("X") + self.initial_state.count("√")
        return 0


@dataclass
class SokobanConfig:
    """Configuration for Sokoban environment and training"""

    dim_room: Tuple[int, int] = (6, 6)
    max_steps: int = 100
    num_boxes: int = 2
    search_depth: int = 300
    dim_x: Optional[int] = None
    dim_y: Optional[int] = None
    render_mode: str = "text"

    # Dataset generation config
    dataset_seed_start: int = 1000
    max_retries: int = 10

    deadlock_distribution: Optional[Dict[str, float]] = field(
        default_factory=lambda: {
            "corner": 0.4,
            "wall": 0.2,
            "u_trap": 0.2,
            "ordering": 0.1,
            "frozen": 0.1,
        }
    )

    def __post_init__(self):
        if self.dim_x is not None and self.dim_y is not None:
            self.dim_room = (self.dim_x, self.dim_y)
            delattr(self, "dim_x")
            delattr(self, "dim_y")

    @property
    def grid_lookup(self) -> Dict[int, str]:
        """Get environment int to symbol mapping"""
        return GRID_CONFIG.env_to_symbol

    @property
    def grid_vocab(self) -> Dict[str, str]:
        """Get symbol to name mapping"""
        return GRID_CONFIG.symbol_to_name


class SokobanSolver:
    """BFS solver for Sokoban puzzles with performance optimizations"""

    # Class-level constants for direction mapping
    MOVES = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    ACTIONS = [Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT]

    def __init__(self, config: SokobanConfig):
        self.config = config

    def solve_with_states(
        self, room_fixed: np.ndarray, initial_state: np.ndarray
    ) -> Tuple[List[Action], List[np.ndarray]]:
        """Get solution with all intermediate states using BFS"""
        queue = deque([(copy.deepcopy(initial_state), [])])
        explored_states = set()

        while queue:
            current_room_state, path = queue.popleft()
            if len(path) > MAX_BFS_DEPTH:
                return [], []

            # Use tobytes() for faster hashing than marshal.dumps()
            state_tohash = current_room_state.tobytes()
            if state_tohash in explored_states:
                continue
            explored_states.add(state_tohash)

            if self._is_solved(current_room_state):
                return self._reconstruct_solution_path(initial_state, room_fixed, path)

            # Try each direction
            for move, action in zip(self.MOVES, self.ACTIONS):
                new_state = self._try_move(current_room_state, room_fixed, move, action)
                if new_state is not None:
                    queue.append((new_state, path + [action]))

        return [], []

    def _is_solved(self, room_state: np.ndarray) -> bool:
        """Check if puzzle is solved (no boxes not on targets)"""
        player_positions = np.argwhere(room_state == PLAYER)
        if len(player_positions) == 0:
            return False

        boxes_not_on_target = np.argwhere(room_state == BOX)
        return len(boxes_not_on_target) == 0

    def _try_move(
        self,
        room_state: np.ndarray,
        room_fixed: np.ndarray,
        move: Tuple[int, int]
    ) -> Optional[np.ndarray]:
        """Try to make a move and return new state if valid"""
        player_positions = np.argwhere(room_state == PLAYER)
        if len(player_positions) == 0:
            return None
        player_pos = tuple(player_positions[0])

        boxes_on_target = set(map(tuple, np.argwhere(room_state == BOX_ON_TARGET)))
        boxes_not_on_target = set(map(tuple, np.argwhere(room_state == BOX)))
        boxes = boxes_on_target | boxes_not_on_target

        new_player_pos = (player_pos[0] + move[0], player_pos[1] + move[1])

        # Check bounds and walls
        if (
            new_player_pos[0] < 0
            or new_player_pos[0] >= room_fixed.shape[0]
            or new_player_pos[1] < 0
            or new_player_pos[1] >= room_fixed.shape[1]
            or room_fixed[new_player_pos] == WALL
        ):
            return None

        new_room_state = copy.deepcopy(room_state)

        # Handle box pushing
        if new_player_pos in boxes:
            if not self._can_push_box(new_player_pos, move, room_fixed, boxes):
                return None
            self._push_box(new_room_state, room_fixed, new_player_pos, move)

        # Move player
        new_room_state[player_pos] = room_fixed[player_pos]
        new_room_state[new_player_pos] = PLAYER

        return new_room_state

    def _can_push_box(
        self,
        box_pos: Tuple[int, int],
        move: Tuple[int, int],
        room_fixed: np.ndarray,
        boxes: set,
    ) -> bool:
        """Check if box can be pushed in given direction"""
        new_box_pos = (box_pos[0] + move[0], box_pos[1] + move[1])

        return not (
            room_fixed[new_box_pos] == WALL
            or new_box_pos in boxes
            or new_box_pos[0] < 0
            or new_box_pos[0] >= room_fixed.shape[0]
            or new_box_pos[1] < 0
            or new_box_pos[1] >= room_fixed.shape[1]
        )

    def _push_box(
        self,
        room_state: np.ndarray,
        room_fixed: np.ndarray,
        box_pos: Tuple[int, int],
        move: Tuple[int, int],
    ):
        """Push box and update room state"""
        new_box_pos = (box_pos[0] + move[0], box_pos[1] + move[1])

        room_state[box_pos] = room_fixed[box_pos]
        if room_fixed[new_box_pos] == TARGET:
            room_state[new_box_pos] = BOX_ON_TARGET
        else:
            room_state[new_box_pos] = BOX

    def _reconstruct_solution_path(
        self,
        initial_state: np.ndarray,
        room_fixed: np.ndarray,
        action_path: List[Action],
    ) -> Tuple[List[Action], List[np.ndarray]]:
        """Reconstruct all states in the solution path"""
        states = [copy.deepcopy(initial_state)]
        current_state = copy.deepcopy(initial_state)

        for action in action_path:
            current_state = self._apply_action(current_state, room_fixed, action)
            states.append(copy.deepcopy(current_state))

        return action_path, states

    def _apply_action(
        self, state: np.ndarray, room_fixed: np.ndarray, action: Action
    ) -> np.ndarray:
        """Apply action to state and return new state"""
        move = self.MOVES[self.ACTIONS.index(action)]
        new_state = self._try_move(state, room_fixed, move, action)

        return new_state if new_state is not None else state


class SokobanEnv(GymSokobanEnv):
    """Wrapper around gym-sokoban with enhanced functionality"""
    def __init__(self, config: SokobanConfig):
        self.config = config
        self.solver = SokobanSolver(config)
        super().__init__(
            dim_room=config.dim_room,
            max_steps=config.max_steps,
            num_boxes=config.num_boxes,
        )
        self.action_sequence = []
        self.solution_states = []

    def reset_env(self, seed: Optional[int] = None) -> str:
        """Reset environment and find solution"""
        retries = 0
        while retries < 10:  # Reasonable retry limit
            try:
                if seed is not None:
                    super().seed(seed + retries)

                super().reset()
                self.action_sequence, self.solution_states = (
                    self.solver.solve_with_states(self.room_fixed, self.room_state)
                )

                if self.action_sequence:  # Solution found
                    return self.render_text()

            except Exception as e:
                last_error = e

            retries += 1

        raise RuntimeError(
            f"Failed to generate solvable puzzle after 10 retries.\nlast error: {last_error}"
        )

    def render_text(self) -> str:
        """Convert room state to text using configured symbols"""
        return self._state_to_text(self.room_state)

    def _state_to_text(self, room_state: np.ndarray) -> str:
        """Convert room state array to text representation"""
        room = room_state.copy()
        room = np.where((room == PLAYER) & (self.room_fixed == TARGET), PLAYER_ON_TARGET, room)

        text_lines = []
        for row in room:
            line = "".join(self.config.grid_lookup.get(cell, "?") for cell in row)
            text_lines.append(line)
        return "\n".join(text_lines)
    
    def set_state(self, room_state: np.ndarray, room_fixed: np.ndarray, player_pos: Tuple[int, int]):
        if room_state.shape != self.room_state.shape:
            raise ValueError(f"room_state shape mismatch: expected {self.room_state.shape}, got {room_state.shape}")
        if room_fixed.shape != self.room_fixed.shape:
            raise ValueError(f"room_fixed shape mismatch: expected {self.room_fixed.shape}, got {room_fixed.shape}")

        self.room_state[:] = room_state
        self.room_fixed[:] = room_fixed
        self.player_position[:] = player_pos



class ReasoningGenerator:
    """Generate strategic reasoning for Sokoban moves"""

    def __init__(self, config: SokobanConfig):
        self.config = config

    def generate_reasoning(
        self,
        action_sequence: List[Action],
        solution_states: List[np.ndarray],
        env: SokobanEnv,
    ) -> str:
        """Generate step-by-step reasoning for solution path"""
        if not action_sequence:
            return (
                "<think>\nNo solution found.\n</think>\n<answer>No valid move</answer>"
            )

        reasoning_steps = ["<think>"]

        for i, action in enumerate(action_sequence):
            step_reasoning = self._generate_step_reasoning(
                i, action, solution_states, env
            )
            reasoning_steps.extend(step_reasoning)

            # Check if puzzle is solved after this step
            if self._is_puzzle_complete(i + 1, solution_states, env):
                reasoning_steps.append("All boxes pushed onto targets.")
                break

        reasoning_steps.append("</think>")

        # First action as the answer
        first_action = action_sequence[0]
        reasoning_steps.append(f"<answer>{first_action.action_name.lower()}</answer>")

        return "\n".join(reasoning_steps)

    def _generate_step_reasoning(
        self,
        step_index: int,
        action: Action,
        solution_states: List[np.ndarray],
        env: SokobanEnv,
    ) -> List[str]:
        """Generate reasoning for a single step"""
        step_num = step_index + 1
        action_name = action.action_name

        if step_index < len(solution_states):
            state_info = self._analyze_state(solution_states[step_index], env)
            purpose = self._determine_move_purpose(state_info, action)
        else:
            purpose = "Continue optimal solution"

        return [f"Step {step_num}: {action_name.lower()}", f"{purpose}."]

    def _analyze_state(self, room_state: np.ndarray, env: SokobanEnv) -> Dict[str, Any]:
        """Analyze current state to extract strategic information"""
        player_positions = np.argwhere(room_state == PLAYER)
        player_pos = tuple(player_positions[0]) if len(player_positions) > 0 else None

        boxes = list(map(tuple, np.argwhere(room_state == BOX)))
        targets = list(map(tuple, np.argwhere(env.room_fixed == TARGET)))
        boxes_on_target = list(map(tuple, np.argwhere(room_state == BOX_ON_TARGET)))

        return {
            "player_pos": player_pos,
            "boxes": boxes,
            "targets": targets,
            "boxes_on_target": boxes_on_target,
            "room_state": room_state,
            "room_fixed": env.room_fixed,
        }

    def _determine_move_purpose(
        self, state_info: Dict[str, Any], action: Action
    ) -> str:
        """Determine the strategic purpose of a move"""
        if not state_info["player_pos"]:
            return "Continue optimal solution"

        will_push, box_pos, new_box_pos = self._will_push_box(state_info, action)

        if will_push and new_box_pos:
            if self._is_target_position(new_box_pos, state_info):
                return "Push box onto target"
            elif self._moves_box_closer_to_target(box_pos, new_box_pos, state_info):
                return "Push box closer to target"
            else:
                return "Push box to setup position"
        else:
            return "Move to push box"

    def _will_push_box(
        self, state_info: Dict[str, Any], action: Action
    ) -> Tuple[bool, Optional[Tuple], Optional[Tuple]]:
        """Check if action will push a box"""
        if not state_info["player_pos"]:
            return False, None, None

        moves = {
            Action.UP: (-1, 0),
            Action.DOWN: (1, 0),
            Action.LEFT: (0, -1),
            Action.RIGHT: (0, 1),
        }
        move = moves[action]

        player_pos = state_info["player_pos"]
        new_player_pos = (player_pos[0] + move[0], player_pos[1] + move[1])

        if new_player_pos in state_info["boxes"]:
            box_pos = new_player_pos
            new_box_pos = (box_pos[0] + move[0], box_pos[1] + move[1])
            return True, box_pos, new_box_pos

        return False, None, None

    def _is_target_position(
        self, pos: Tuple[int, int], state_info: Dict[str, Any]
    ) -> bool:
        """Check if position is a target"""
        return pos in state_info["targets"]

    def _moves_box_closer_to_target(
        self,
        old_pos: Tuple[int, int],
        new_pos: Tuple[int, int],
        state_info: Dict[str, Any],
    ) -> bool:
        """Check if move brings box closer to any target"""
        if not state_info["targets"]:
            return False

        closest_target = min(
            state_info["targets"], key=lambda t: self._manhattan_distance(old_pos, t)
        )
        old_distance = self._manhattan_distance(old_pos, closest_target)
        new_distance = self._manhattan_distance(new_pos, closest_target)

        return new_distance < old_distance

    def _manhattan_distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> int:
        """Calculate Manhattan distance between two positions"""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def _is_puzzle_complete(
        self, step_index: int, solution_states: List[np.ndarray], env: SokobanEnv
    ) -> bool:
        """Check if puzzle is complete after given step"""
        if step_index >= len(solution_states):
            return False

        state_info = self._analyze_state(solution_states[step_index], env)
        return len(state_info["boxes"]) == 0


class SokobanGenerator:
    """Generate SFT training data with strategic reasoning"""

    def __init__(self, config: SokobanConfig = None):
        self.config = config or SokobanConfig()
        self.reasoning_generator = ReasoningGenerator(self.config)

    def generate_puzzle_data(
        self, seed: Optional[int] = None, every_k: Optional[int] = None
    ) -> Optional[TrainingExample]:
        """Generate puzzle and create training examples"""
        try:
            env = SokobanEnv(self.config)
            env.reset_env(seed=seed)

            if not env.action_sequence or len(env.solution_states) < 2:
                return None

            return self._create_training_example(env, every_k)

        except Exception as e:
            print(f"Error generating puzzle data: {e}")
            return None

    def _create_training_example(
        self, env: SokobanEnv, every_k: Optional[int] = None
    ) -> List[TrainingExample]:
        """Create full and partially solved training examples from solved puzzle"""
        examples = []
        num_steps = len(env.action_sequence)

        for i in range(num_steps):
            current_state_text = env._state_to_text(env.solution_states[i])
            remaining_actions = env.action_sequence[i:]
            remaining_states = env.solution_states[i:]

            reasoning = self.reasoning_generator.generate_reasoning(
                remaining_actions, remaining_states, env
            )

            trace_item = TrainingTrace(
                state_representation=current_state_text,
                action=env.action_sequence[i],
                reasoning=reasoning,
                step=i,
            )

            if i == 0:
                examples.append(TrainingExample(
                    initial_state=current_state_text,
                    trace=[trace_item],
                    optimal_path=remaining_actions,
                ))
            elif every_k and (i % every_k == 0):
                examples.append(TrainingExample(
                    initial_state=current_state_text,
                    trace=[trace_item],
                    optimal_path=remaining_actions,
                ))

        return examples

    def generate_multiple_examples(
        self, count: int, sizes: List[int], every_k: int = None
    ) -> List[TrainingExample]:
        """Generate multiple training examples across different sizes"""
        examples = []
        successful = 0
        attempts = count * 3

        with tqdm(total=count, desc="Generating Sokoban examples") as pbar:
            for i in range(attempts):
                if successful >= count:
                    break

                size = sizes[i % len(sizes)]

                config = SokobanConfig(
                    dim_room=(size, size),
                    num_boxes=min(3, max(1, size // 3)),
                    max_steps=size * size,
                    dataset_seed_start=self.config.dataset_seed_start + i,
                )

                generator = SokobanGenerator(config)
                example_list = generator.generate_puzzle_data(seed=config.dataset_seed_start + i, every_k=every_k)

                if example_list:
                    for example in example_list:
                        examples.append(example)
                        successful += 1
                        pbar.update(1)
                        if successful >= count:
                            break

        print(f"Successfully generated {len(examples)}/{count} examples")
        return examples


class SokobanDeadlockGenerator:
    """Optimized Sokoban deadlock generator with performance improvements via caching"""

    def __init__(self, config: SokobanConfig):
        self.config = config
        self.rng = np.random.default_rng(config.dataset_seed_start)

        # Pre-compute valid positions cache for better performance
        self._valid_positions_cache = {}
        self._corner_cache = {}
        self._wall_cache = {}

    def _sample_deadlock_type(self) -> str:
        """Randomly selects a deadlock type based on configured distribution probabilities"""
        types = list(self.config.deadlock_distribution.keys())
        probs = list(self.config.deadlock_distribution.values())
        return self.rng.choice(types, p=probs)

    def _get_valid_positions(self, room_fixed: np.ndarray, room_state: np.ndarray) -> List[Tuple[int, int]]:
        """Cache valid positions to avoid recomputing"""
        cache_key = room_fixed.tobytes()
        if cache_key not in self._valid_positions_cache:
            h, w = room_fixed.shape
            valid_pos = [
                (i, j) for i in range(1, h-1) for j in range(1, w-1)
                if room_fixed[i, j] == EMPTY  # floor space
            ]
            self._valid_positions_cache[cache_key] = valid_pos

        # Filter by current room state
        return [(i, j) for i, j in self._valid_positions_cache[cache_key]
                if room_state[i, j] == EMPTY]

    def _get_corner_positions(self, room_fixed: np.ndarray) -> List[Tuple[int, int]]:
        """Pre-compute and cache corner positions"""
        cache_key = room_fixed.tobytes()
        if cache_key not in self._corner_cache:
            h, w = room_fixed.shape
            corners = []
            for i in range(1, h-1):
                for j in range(1, w-1):
                    if room_fixed[i, j] != EMPTY:
                        continue

                    # Corner if 2 adjacent walls
                    adjacent_walls = (
                        (room_fixed[i+1, j] == WALL and room_fixed[i, j+1] == WALL) or
                        (room_fixed[i-1, j] == WALL and room_fixed[i, j+1] == WALL) or
                        (room_fixed[i+1, j] == WALL and room_fixed[i, j-1] == WALL) or
                        (room_fixed[i-1, j] == WALL and room_fixed[i, j-1] == WALL)
                    )

                    if adjacent_walls:
                        corners.append((i, j))

            self._corner_cache[cache_key] = corners
        return self._corner_cache[cache_key]

    def _get_wall_positions(self, room_fixed: np.ndarray) -> List[Tuple[int, int]]:
        """Pre-compute and cache wall-adjacent positions"""
        cache_key = room_fixed.tobytes()
        if cache_key not in self._wall_cache:
            h, w = room_fixed.shape
            wall_adjacent = []
            for i in range(1, h-1):
                for j in range(1, w-1):
                    if room_fixed[i, j] != EMPTY:
                        continue

                    # Check if adjacent to at least one wall
                    if any(room_fixed[i+di, j+dj] == WALL for di, dj in [(-1,0), (1,0), (0,-1), (0,1)]):
                        wall_adjacent.append((i, j))

            self._wall_cache[cache_key] = wall_adjacent
        return self._wall_cache[cache_key]

    def _place_box_in_corner(self, room_state: np.ndarray, room_fixed: np.ndarray) -> bool:
        """Optimized corner box placement using pre-computed positions"""
        corners = self._get_corner_positions(room_fixed)

        # Filter corners that are currently empty and not targets
        available_corners = [
            pos for pos in corners
            if room_state[pos] == EMPTY and room_fixed[pos] != TARGET
        ]

        if not available_corners:
            return False

        # Pick random corner
        pos = available_corners[self.rng.integers(0, len(available_corners))]
        room_state[pos] = BOX
        return True

    def _place_box_against_wall(self, room_state: np.ndarray, room_fixed: np.ndarray) -> bool:
        """Optimized wall box placement"""
        wall_positions = self._get_wall_positions(room_fixed)

        # Filter available positions
        available = [
            pos for pos in wall_positions
            if room_state[pos] == EMPTY and room_fixed[pos] != TARGET
        ]

        if not available:
            return False

        pos = available[self.rng.integers(0, len(available))]
        room_state[pos] = BOX
        return True

    def _make_u_trap(self, room_state: np.ndarray, room_fixed: np.ndarray) -> bool:
        """Optimized U-trap creation with better position selection"""
        h, w = room_state.shape
        candidates = []

        # Pre-filter U-trap candidates
        for i in range(1, h-1):
            for j in range(1, w-1):
                if room_fixed[i, j] != EMPTY or room_state[i, j] != EMPTY:
                    continue

                # Check for U-shape patterns
                horizontal_u = (room_fixed[i-1, j] == WALL and room_fixed[i+1, j] == WALL and
                              room_fixed[i, j-1] == EMPTY and room_fixed[i, j+1] == EMPTY)
                vertical_u = (room_fixed[i, j-1] == WALL and room_fixed[i, j+1] == WALL and
                            room_fixed[i-1, j] == EMPTY and room_fixed[i+1, j] == EMPTY)

                if horizontal_u or vertical_u:
                    candidates.append((i, j))

        if not candidates:
            return False

        pos = candidates[self.rng.integers(0, len(candidates))]
        room_state[pos] = BOX
        return True

    def _block_box_ordering(self, room_state: np.ndarray, room_fixed: np.ndarray) -> bool:
        """Optimized box ordering deadlock"""
        h, w = room_state.shape

        # Find pairs of adjacent empty spaces
        for i in range(1, h-1):
            for j in range(1, w-2):  # Leave room for j+1
                pos1, pos2 = (i, j), (i, j+1)

                if (room_fixed[pos1] == EMPTY and room_state[pos1] == EMPTY and
                    room_fixed[pos2] == EMPTY and room_state[pos2] == EMPTY and
                    room_fixed[i, j-1] == WALL and room_fixed[i, j+2] == WALL):

                    room_state[pos1] = BOX
                    room_state[pos2] = BOX
                    return True

        return False

    def _make_frozen_state(self, room_state: np.ndarray, room_fixed: np.ndarray) -> bool:
        """Optimized 2x2 frozen box creation"""
        h, w = room_state.shape

        # Check 2x2 blocks
        for i in range(1, h-2):
            for j in range(1, w-2):
                positions = [(i, j), (i, j+1), (i+1, j), (i+1, j+1)]

                if all(room_fixed[pos] == EMPTY and room_state[pos] == EMPTY for pos in positions):
                    for pos in positions:
                        room_state[pos] = BOX
                    return True

        return False

    def _inject_deadlock(self, deadlock_type: str, room_state: np.ndarray, room_fixed: np.ndarray) -> bool:
        """Optimized deadlock injection without retry loop"""
        if deadlock_type == "corner":
            return self._place_box_in_corner(room_state, room_fixed)
        elif deadlock_type == "wall":
            return self._place_box_against_wall(room_state, room_fixed)
        elif deadlock_type == "u_trap":
            return self._make_u_trap(room_state, room_fixed)
        elif deadlock_type == "ordering":
            return self._block_box_ordering(room_state, room_fixed)
        elif deadlock_type == "frozen":
            return self._make_frozen_state(room_state, room_fixed)

        return False

    def _quick_unsolvable_check(self, room_state: np.ndarray, room_fixed: np.ndarray) -> bool:
        """Quick heuristic check if state is likely unsolvable without full BFS"""
        # Count boxes and targets
        boxes = np.count_nonzero(room_state == BOX)
        boxes_on_target = np.count_nonzero(room_state == BOX_ON_TARGET)
        targets = np.count_nonzero(room_fixed == TARGET)

        # Basic checks
        if boxes + boxes_on_target > targets:
            return True  # More boxes than targets

        # Check for obvious corner deadlocks
        h, w = room_state.shape
        box_positions = list(zip(*np.where(room_state == BOX)))

        for box_pos in box_positions:
            i, j = box_pos
            if room_fixed[i, j] != TARGET:  # Not on target
                # Check if in corner
                wall_count = sum([
                    room_fixed[i+1, j] == WALL if i+1 < h else True,
                    room_fixed[i-1, j] == WALL if i-1 >= 0 else True,
                    room_fixed[i, j+1] == WALL if j+1 < w else True,
                    room_fixed[i, j-1] == WALL if j-1 >= 0 else True
                ])
                if wall_count >= 2:
                    return True  # Box in corner, not on target

        return False

    def _generate_deadlock_reasoning(self, deadlock_type: str) -> str:
        """Generate explanation for why a particular deadlock type has no solution"""
        deadlock_explanations = {
            "corner": "Box is trapped in corner with no target",
            "wall": "Box is against wall with no path to targets",
            "u_trap": "Box is in U-shaped trap with no escape",
            "ordering": "Boxes block each other's path to targets",
            "frozen": "Boxes are mutually frozen and cannot move"
        }
        base_explanation = deadlock_explanations.get(deadlock_type, "Deadlock detected")
        return f"<think>\n{base_explanation}. No solution possible.\n</think>\n<answer>no valid move</answer>"

    def create_deadlock_example(self, seed: Optional[int] = None) -> Optional[TrainingExample]:
        """Optimized deadlock example creation"""
        try:
            env = SokobanEnv(self.config)
            env.reset_env(seed=seed)

            room = env.room_state.copy()
            deadlock_type = self._sample_deadlock_type()

            # Try to inject deadlock
            success = self._inject_deadlock(deadlock_type, room, env.room_fixed)
            if not success:
                return None

            # Quick unsolvable check first
            if not self._quick_unsolvable_check(room, env.room_fixed):
                # If quick check passes, do full verification
                actions, _ = env.solver.solve_with_states(env.room_fixed, room)
                if actions:  # Still solvable
                    return None

            # Create example
            state_text = env._state_to_text(room)
            reasoning = self._generate_deadlock_reasoning(deadlock_type)

            trace = [TrainingTrace(
                state_representation=state_text,
                action=None,
                reasoning=reasoning,
                step=0,
            )]

            return TrainingExample(
                initial_state=state_text,
                trace=trace,
                deadlock_type=deadlock_type
            )

        except Exception:
            return None

    def generate_multiple_examples(self, count: int, sizes: List[int]) -> List[TrainingExample]:
        """Generate examples with better batching and early termination"""
        examples = []
        successful = 0

        # Reduce attempts multiplier since we're more efficient now
        max_attempts = count * 2  # Reduced from 3x

        with tqdm(total=count, desc="Generating deadlock examples") as pbar:
            for i in range(max_attempts):
                if successful >= count:
                    break

                size = sizes[i % len(sizes)]

                config = SokobanConfig(
                    dim_room=(size, size),
                    num_boxes=min(2, max(1, size // 4)),  # Fewer boxes for faster generation
                    max_steps=size * size,
                    dataset_seed_start=self.config.dataset_seed_start + i,
                )

                generator = SokobanDeadlockGenerator(config)
                example = generator.create_deadlock_example(seed=config.dataset_seed_start + i)

                if example:
                    examples.append(example)
                    successful += 1
                    pbar.update(1)

        print(f"Successfully generated {len(examples)}/{count} deadlock examples")
        return examples


@dataclass
class DatasetConfig:
    """Configuration for dataset generation"""

    count: int
    grid_sizes: List[int]
    output_file: str


class SokobanBuilder:
    """Service for generating complete datasets"""

    def __init__(self, base_config: SokobanConfig):
        self.base_config = base_config

    def _format_training_instance(
        self, trace_item: TrainingTrace = None, metadata: Dict[str, Any] = None,
        deadlock_state: str = None, deadlock_type: str = None
    ) -> Dict[str, Any]:
        """Format training examples into prompt-response pairs"""
        legend = ", ".join(f"{k} = {v}" for k, v in self.base_config.grid_vocab.items())

        state_representation = deadlock_state if deadlock_state else trace_item.state_representation

        prompt = f"""<|im_start|>user
You are a Sokoban solver.

Sokoban Quick Guide
Goal: Push all boxes (X) onto targets (O).

Symbols:
{legend}

Rules:

Push boxes (can't pull).
Avoid walls (#).
Answers:
<answer> Up </answer> | <answer> Down </answer> | <answer> Left </answer> | <answer> Right </answer>

Rewards:
Move: -0.1
Box on target: +1.0
All boxes placed: +10.0

[Cumulative Observations]:

{state_representation}

Decide the next action:
Always output: <answer> [your answer] </answer> with no extra text. Strictly follow this format. <|im_end|>
<|im_start|>assistant
"""

        return {
            "prompt": prompt,
            "response": trace_item.reasoning,
            "map": trace_item.state_representation,
            "metadata": metadata,
        }


def run_demo(args):
    """Run demo of sample puzzles and traces"""
    print("=== Sokoban Dataset Generation Demo ===\n")

    # Create a small config for demo
    config = SokobanConfig(
        dim_room=(6, 6),
        num_boxes=2,
        max_steps=50,
        dataset_seed_start=args.seed if args.seed else 1000,
    )

    generator = SokobanGenerator(config)
    examples = generator.generate_puzzle_data(seed=config.dataset_seed_start, every_k=args.every_k)

    for example in examples:
        # Create formatted training instance
        first_trace = example.trace[0]
        builder = SokobanBuilder(config)
        formatted_instance = builder._format_training_instance(
            first_trace, example.metadata
        )

        print("Sample puzzle and reasoning:")
        print("=" * 50)
        print("MAP:")
        print(formatted_instance["map"])
        print("\nRESPONSE:")
        print(formatted_instance["response"])
        print("=" * 50)
    else:
        print("Failed to generate demo puzzle")


def generate_dataset_split(
    split_name: str,
    num_examples: int,
    num_deadlocks: int,
    sizes: List[int],
    args,
    base_config: SokobanConfig,
    every_k: Optional[int] = None,
):
    """Generate examples for a single dataset split"""
    if num_examples <= 0 and num_deadlocks <= 0:
        return []

    print(f"\nGenerating {split_name} split ({num_examples} solvable + {num_deadlocks} unsolvable):")

    generator = SokobanGenerator(base_config)
    all_examples = generator.generate_multiple_examples(num_examples, sizes, every_k)

    deadlock_generator = SokobanDeadlockGenerator(base_config)
    all_deadlocks = deadlock_generator.generate_multiple_examples(num_deadlocks, sizes)

    all_data = all_examples + all_deadlocks

    random.shuffle(all_data)

    if all_data:
        output_file = f"{args.output}_{split_name}.jsonl"
        print(f"Saving  to {output_file}")

        with open(output_file, "w") as f:
            for example in all_data:
                if example.trace:  # solvable example
                    first_trace_item = example.trace[0]
                    formatted_instance = SokobanBuilder(base_config)._format_training_instance(
                        trace_item=first_trace_item,
                        metadata=example.metadata
                    )
                else:  # deadlock example
                    formatted_instance = SokobanBuilder(base_config)._format_training_instance(
                        metadata=example.metadata,
                        deadlock_state=example.initial_state,
                        deadlock_type=example.deadlock_type
                    )
                f.write(json.dumps(formatted_instance) + "\n")

    return all_data


def main():
    """Main function with argument parsing"""
    parser = argparse.ArgumentParser(description="Generate Sokoban training datasets")
    parser.add_argument(
        "--train", type=int, default=0, help="Number of training examples"
    )
    parser.add_argument(
        "--val", type=int, default=0, help="Number of validation examples"
    )
    parser.add_argument("--test", type=int, default=0, help="Number of test examples")
    parser.add_argument(
        "--every-k",
        type=int,
        default=None,
        help="Add intermediate puzzles every k steps in the solution trace to training data",
    )

    parser.add_argument("--train-deadlocks", type=int, default=0, help="Number of training deadlock examples")
    parser.add_argument("--val-deadlocks", type=int, default=0, help="Number of validation deadlock examples")

    parser.add_argument(
        "--train-sizes",
        nargs="+",
        type=int,
        default=[6, 7, 8],
        help="Training data map size(s)",
    )
    parser.add_argument(
        "--val-sizes",
        nargs="+",
        type=int,
        default=[6, 7, 8],
        help="Validation data map size(s)",
    )
    parser.add_argument(
        "--test-sizes",
        nargs="+",
        type=int,
        default=[9, 10],
        help="Test data map size(s)",
    )
    parser.add_argument(
        "--output", type=str, default="sokoban", help="Base output filename prefix"
    )
    parser.add_argument(
        "--num-boxes",
        type=int,
        default=1,
        help="Number of boxes (overrides size-based scaling)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--demo", action="store_true", help="Run demo of sample puzzles and traces"
    )

    args = parser.parse_args()

    # Set up base configuration - use seed from args if provided
    base_config = SokobanConfig(dataset_seed_start=args.seed if args.seed else 1000)

    # Run demo if requested
    if args.demo:
        run_demo(args)
        if not any([args.train, args.val, args.test]):
            return

    # Generate each split
    generate_dataset_split("train", args.train, args.train_deadlocks, args.train_sizes, args, base_config, every_k=args.every_k)
    generate_dataset_split("val", args.val, args.val_deadlocks, args.val_sizes, args, base_config, every_k=None)
    generate_dataset_split("test", args.test, 0, args.test_sizes, args, base_config, every_k=None)

    print(f"\nDataset generation complete! Files saved with prefix '{args.output}'")


if __name__ == "__main__":
    main()
