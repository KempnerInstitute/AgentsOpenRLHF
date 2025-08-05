import argparse
import json
import random
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class Position:
    row: int
    col: int

    def __add__(self, other: "Position") -> "Position":
        return Position(self.row + other.row, self.col + other.col)


@dataclass
class Grid:
    """Pure data structure representing the frozen lake grid"""

    grid_data: List[List[str]]
    start_pos: Position
    goal_pos: Position

    @property
    def size(self) -> int:
        return len(self.grid_data)

    def get_cell(self, pos: Position) -> Optional[str]:
        if self.is_valid_position(pos):
            return self.grid_data[pos.row][pos.col]
        return None

    def is_valid_position(self, pos: Position) -> bool:
        return 0 <= pos.row < self.size and 0 <= pos.col < self.size

    def is_hole(self, pos: Position) -> bool:
        return self.get_cell(pos) == "H"

    def is_frozen(self, pos: Position) -> bool:
        return self.get_cell(pos) == "F"

    def to_string(self) -> str:
        return "\n".join([" ".join(row) for row in self.grid_data])

    def to_string_with_player(self, player_pos: Position) -> str:
        display_grid = [row[:] for row in self.grid_data]
        if player_pos != self.start_pos and player_pos != self.goal_pos:
            display_grid[player_pos.row][player_pos.col] = "@"
        return "\n".join([" ".join(row) for row in display_grid])


class Action(Enum):
    LEFT = ("Left", Position(0, -1), 0)
    DOWN = ("Down", Position(1, 0), 1)
    RIGHT = ("Right", Position(0, 1), 2)
    UP = ("Up", Position(-1, 0), 3)

    def __init__(self, name: str, delta: Position, gym_id: int):
        self.action_name = name
        self.delta = delta
        self.gym_id = gym_id

    @classmethod
    def from_name(cls, name: str) -> "Action":
        name_map = {action.action_name.upper(): action for action in cls}
        return name_map.get(name.upper())

    @classmethod
    def from_gym_id(cls, gym_id: int) -> "Action":
        id_map = {action.gym_id: action for action in cls}
        return id_map.get(gym_id)

    @classmethod
    def all_actions(cls) -> List["Action"]:
        return list(cls)


class GameState:
    """Represents the current state of the game"""

    def __init__(self, grid: Grid, position: Position):
        self.grid = grid
        self.position = position

    def get_valid_actions(self) -> List[Action]:
        valid_actions = []
        for action in Action.all_actions():
            next_pos = self.position + action.delta
            if self.grid.is_valid_position(next_pos) and not self.grid.is_hole(
                next_pos
            ):
                valid_actions.append(action)
        return valid_actions

    def apply_action(self, action: Action) -> Optional["GameState"]:
        next_pos = self.position + action.delta
        if action in self.get_valid_actions():
            return GameState(self.grid, next_pos)
        return None

    def is_goal_reached(self) -> bool:
        return self.position == self.grid.goal_pos

    def manhattan_distance_to_goal(self) -> int:
        return abs(self.position.row - self.grid.goal_pos.row) + abs(
            self.position.col - self.grid.goal_pos.col
        )


class BFSPathFinder:
    """Breadth-First Search implementation for finding optimal path"""

    def find_optimal_path(self, grid: Grid) -> List[Action]:
        queue = deque([(grid.start_pos, [])])
        visited = {grid.start_pos}

        while queue:
            current_pos, path = queue.popleft()

            if current_pos == grid.goal_pos:
                return path

            for action in Action.all_actions():
                next_pos = current_pos + action.delta

                if (
                    grid.is_valid_position(next_pos)
                    and not grid.is_hole(next_pos)
                    and next_pos not in visited
                ):

                    visited.add(next_pos)
                    queue.append((next_pos, path + [action]))

        return []  # No path found


class BFSReasoningStrategy:
    """Generates BFS-style reasoning explanations"""

    def generate_reasoning(
        self, game_state: GameState, action: Action, optimal_path: List[Action]
    ) -> str:
        # Generate complete BFS reasoning for the entire optimal path
        bfs_steps = []
        current_pos = game_state.position

        for i, path_action in enumerate(optimal_path):
            next_pos = current_pos + path_action.delta
            reasons = []

            # Distance reasoning
            current_dist = self._manhattan_distance(
                current_pos, game_state.grid.goal_pos
            )
            next_dist = self._manhattan_distance(next_pos, game_state.grid.goal_pos)

            if next_dist < current_dist:
                reasons.append(
                    f"gets closer to the goal ({current_dist} → {next_dist})"
                )
            elif next_dist == current_dist:
                reasons.append("maintains distance to goal")
            else:
                reasons.append("increases distance but necessary for optimal path")

            # Safety analysis
            holes_nearby = self._count_adjacent_holes(next_pos, game_state.grid)
            if holes_nearby == 0:
                reasons.append("no holes nearby (safe)")
            elif holes_nearby == 1:
                reasons.append("1 adjacent hole (moderately safe)")
            else:
                reasons.append(f"{holes_nearby} adjacent holes (risky)")

            # Goal proximity
            if next_pos == game_state.grid.goal_pos:
                reasons.append("reaches the goal")
            elif next_dist == 1:
                reasons.append("close to goal")

            # Alternative moves analysis
            temp_state = GameState(game_state.grid, current_pos)
            valid_moves = temp_state.get_valid_actions()
            if len(valid_moves) <= 1:
                reasons.append("no other safe moves available")
            elif len(valid_moves) == 2:
                reasons.append("limited safe move options")

            # Add this step to BFS sequence
            step_reasoning = ". ".join(reasons) + "."
            bfs_steps.append(
                f"Step {i + 1}: action {path_action.action_name}. \nReason: {step_reasoning}"
            )

            # Move to next position
            current_pos = next_pos

        # Add final goal achievement
        bfs_steps.append("Reach goal")

        # Format: complete BFS reasoning in think, but answer with current step action
        complete_bfs = "\n".join(bfs_steps)
        return f"<think>\n{complete_bfs}\n</think>\n\n<answer>{action.action_name}</answer>"

    def _manhattan_distance(self, pos1: Position, pos2: Position) -> int:
        return abs(pos1.row - pos2.row) + abs(pos1.col - pos2.col)

    def _count_adjacent_holes(self, position: Position, grid: Grid) -> int:
        count = 0
        for direction in [
            Position(0, 1),
            Position(0, -1),
            Position(1, 0),
            Position(-1, 0),
        ]:
            neighbor = position + direction
            if grid.is_valid_position(neighbor) and grid.is_hole(neighbor):
                count += 1
        return count


@dataclass
class GridGeneratorConfig:
    grid_size: int
    hole_density: float
    max_attempts: int = 100
    random_seed: Optional[int] = None


class GridGenerator:
    """Generates random solvable grids"""

    def __init__(self, config: GridGeneratorConfig, path_finder: BFSPathFinder):
        self.config = config
        self.path_finder = path_finder
        if config.random_seed is not None:
            random.seed(config.random_seed)

    def generate_solvable_grid(self) -> Grid:
        for attempt in range(self.config.max_attempts):
            grid = self._generate_random_grid()
            if self._is_solvable(grid):
                return grid

        raise Exception(
            f"Failed to generate solvable grid after {self.config.max_attempts} attempts"
        )

    def _generate_random_grid(self) -> Grid:
        size = self.config.grid_size
        grid_data = [["F" for _ in range(size)] for _ in range(size)]

        # Randomly place start and goal
        all_positions = [Position(r, c) for r in range(size) for c in range(size)]
        start_pos, goal_pos = random.sample(all_positions, 2)

        grid_data[start_pos.row][start_pos.col] = "S"
        grid_data[goal_pos.row][goal_pos.col] = "G"

        # Add holes
        available_positions = [
            pos for pos in all_positions if pos not in [start_pos, goal_pos]
        ]
        num_holes = int(len(available_positions) * self.config.hole_density)
        hole_positions = random.sample(available_positions, num_holes)

        for hole_pos in hole_positions:
            grid_data[hole_pos.row][hole_pos.col] = "H"

        return Grid(grid_data, start_pos, goal_pos)

    def _is_solvable(self, grid: Grid) -> bool:
        path = self.path_finder.find_optimal_path(grid)
        return len(path) > 0


@dataclass
class TrainingTrace:
    state_representation: str
    action: Action
    reasoning: str
    step: int


@dataclass
class TrainingExample:
    grid: Grid
    trace: List[TrainingTrace]
    optimal_path: List[Action]

    @property
    def metadata(self) -> Dict[str, Any]:
        return {
            "grid_size": self.grid.size,
            "optimal_path_length": len(self.optimal_path),
            "action_name": self.trace[0].action.action_name if self.trace else None,
            "action_int": self.trace[0].action.gym_id if self.trace else None,
        }


class FrozenLakeGenerator:
    """Service for generating training examples"""

    def __init__(
        self, path_finder: BFSPathFinder, reasoning_strategy: BFSReasoningStrategy
    ):
        self.path_finder = path_finder
        self.reasoning_strategy = reasoning_strategy

    def generate_training_example(
        self, grid_size: int, hole_density: float = 0.3
    ) -> TrainingExample:
        config = GridGeneratorConfig(grid_size, hole_density)
        grid_generator = GridGenerator(config, self.path_finder)

        grid = grid_generator.generate_solvable_grid()
        optimal_path = self.path_finder.find_optimal_path(grid)

        trace = []
        current_state = GameState(grid, grid.start_pos)

        for step_index, action in enumerate(optimal_path):
            reasoning = self.reasoning_strategy.generate_reasoning(
                current_state, action, optimal_path
            )

            trace.append(
                TrainingTrace(
                    state_representation=current_state.grid.to_string_with_player(
                        current_state.position
                    ),
                    action=action,
                    reasoning=reasoning,
                    step=step_index,
                )
            )

            current_state = current_state.apply_action(action)
            if current_state and current_state.is_goal_reached():
                break

        return TrainingExample(grid, trace, optimal_path)

    def generate_multiple_examples(
        self, count: int, grid_sizes: List[int], hole_density: float = 0.3
    ) -> List[TrainingExample]:
        examples = []
        for i in range(count):
            size = grid_sizes[i % len(grid_sizes)]
            example = self.generate_training_example(size, hole_density)
            examples.append(example)
        return examples

def make_prompt(str_representation):
    return f"""<|im_start|>user
You are walking on a frozen lake.

FrozenLake Quick Guide
Goal: Reach the goal (G)

Symbols:
S Start | F Frozen | H Hole | G Goal

Rules:
- Avoid falling into holes (H)
- Frozen tiles are slippery

Answers:
<answer> Up </answer> | <answer> Down </answer> | <answer> Left </answer> | <answer> Right </answer>

Rewards:
Fall into hole: 0
Reach goal: +1.0

Grid:
{str_representation}

What action should you take next?

Decide the next action:
Always output: <answer> [your answer] </answer> with no extra text.
Strictly follow this format. <|im_end|>
<|im_start|>assistant
<think>"""

def format_training_instance(
    trace_item: TrainingTrace, metadata: Dict[str, Any]
) -> Dict[str, Any]:
    """Formats training examples into prompt-response pairs"""
    prompt = make_prompt(trace_item.state_representation)

    return {"prompt": prompt, "response": trace_item.reasoning, "metadata": metadata}


@dataclass
class DatasetConfig:
    count: int
    grid_sizes: List[int]
    output_file: str


class DatasetBuilder:
    """Service for generating complete datasets"""

    def __init__(self, training_example_service: FrozenLakeGenerator):
        self.training_example_service = training_example_service

    def generate_dataset_splits(
        self,
        train_config: DatasetConfig,
        val_config: DatasetConfig,
        test_config: DatasetConfig,
        hole_density: float = 0.3,
    ):
        if train_config.count > 0:
            print(f"Generating {train_config.count} training examples...")
            train_examples = self.training_example_service.generate_multiple_examples(
                train_config.count, train_config.grid_sizes, hole_density
            )
            self._write_examples_to_file(train_examples, train_config.output_file)

        if val_config.count > 0:
            print(f"Generating {val_config.count} validation examples...")
            val_examples = self.training_example_service.generate_multiple_examples(
                val_config.count, val_config.grid_sizes, hole_density
            )
            self._write_examples_to_file(val_examples, val_config.output_file)

        if test_config.count > 0:
            print(f"Generating {test_config.count} test examples...")
            test_examples = self.training_example_service.generate_multiple_examples(
                test_config.count, test_config.grid_sizes, hole_density
            )
            self._write_examples_to_file(test_examples, test_config.output_file)

        print("Dataset generation complete!")

    def _write_examples_to_file(self, examples: List[TrainingExample], filename: str):
        import json

        with open(filename, "w") as f:
            for example in examples:
                if example.trace:
                    first_trace_item = example.trace[0]
                    formatted_instance = format_training_instance(
                        first_trace_item, example.metadata
                    )
                    f.write(json.dumps(formatted_instance) + "\n")


def load_config(config_path: str) -> Optional[Dict[str, Any]]:
    """Load configuration from JSON file"""
    try:
        with open(config_path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return None


def create_dataset_configs(
    train_count: int,
    val_count: int,
    test_count: int,
    output_prefix: str,
    train_sizes: List[int],
    val_sizes: List[int],
    test_sizes: List[int],
) -> Dict[str, DatasetConfig]:
    """Create dataset configuration objects"""
    return {
        "train": DatasetConfig(
            train_count, train_sizes, f"{output_prefix}_train.jsonl"
        ),
        "val": DatasetConfig(val_count, val_sizes, f"{output_prefix}_val.jsonl"),
        "test": DatasetConfig(test_count, test_sizes, f"{output_prefix}_test.jsonl"),
    }


def run_demo(example_generator, demo_count, hole_density):
    """Run demonstration mode showing sample grids and solutions."""
    print("\n=== Demo Examples ===")
    examples = example_generator.generate_multiple_examples(
        demo_count, [4, 6, 8], hole_density
    )

    for i, example in enumerate(examples, 1):
        print(f"\nExample {i} ({example.grid.size}x{example.grid.size} grid):")
        print("Grid:")
        print(example.grid.to_string())
        print(f"\nOptimal solution ({len(example.trace)} steps):")

        if example.trace:
            trace_item = example.trace[0]
            print(f"First action: {trace_item.action.action_name}")
            print(f"Reasoning: {trace_item.reasoning}")

        print("-" * 60)


def generate_datasets(example_generator, args):
    """Generate training, validation, and test datasets."""
    dataset_configs = create_dataset_configs(
        args.train,
        args.val,
        args.test,
        args.output,
        args.train_sizes,
        args.val_sizes,
        args.test_sizes,
    )

    dataset_builder = DatasetBuilder(example_generator)
    dataset_builder.generate_dataset_splits(
        dataset_configs["train"],
        dataset_configs["val"],
        dataset_configs["test"],
        args.density,
    )


def main():
    parser = argparse.ArgumentParser(description="Frozen Lake RLHF Dataset Generator")
    parser.add_argument(
        "--train", type=int, default=0, help="Number of training examples"
    )
    parser.add_argument(
        "--val", type=int, default=0, help="Number of validation examples"
    )
    parser.add_argument("--test", type=int, default=0, help="Number of test examples")
    parser.add_argument(
        "--train-sizes",
        nargs="+",
        type=int,
        default=[4, 5, 6, 7, 8],
        help="training data map size(s)",
    )
    parser.add_argument(
        "--val-sizes",
        nargs="+",
        type=int,
        default=[4, 5, 6, 7, 8],
        help="validation data map size(s)",
    )
    parser.add_argument(
        "--test-sizes",
        nargs="+",
        type=int,
        default=[9, 10],
        help="test data map size(s)",
    )
    parser.add_argument(
        "--output", type=str, default="frozen_lake", help="Base output filename prefix"
    )
    parser.add_argument("--density", type=float, default=0.3, help="Hole density")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument(
        "--demo", type=int, default=0, help="Run demo of sample grids and traces"
    )

    args = parser.parse_args()

    # Load config
    config = load_config("/n/holylfs06/LABS/kempner_undergrads/Lab/ellenma/openrlhf-proj/AgentsOpenRLHF/scripts/info.json")
    if config:
        print(f"Loaded config for: {config.get('env_name', 'Unknown')}")
    else:
        print("info.json not found, using default config")

    # Set up components
    path_finder = BFSPathFinder()
    reasoning_strategy = BFSReasoningStrategy()
    example_generator = FrozenLakeGenerator(path_finder, reasoning_strategy)

    # Run demo if requested
    if args.demo > 0:
        run_demo(example_generator, args.demo, args.density)

    # Generate datasets if requested
    if args.train > 0 or args.val > 0 or args.test > 0:
        generate_datasets(example_generator, args)


if __name__ == "__main__":
    main()
