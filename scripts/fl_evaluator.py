import argparse
import json
import re
from collections import defaultdict
from typing import Dict, List

import gymnasium as gym


class FrozenLakeEvaluator:
    def __init__(self):
        # Action mapping for gym FrozenLake
        self.action_name_to_int = {"Left": 0, "Down": 1, "Right": 2, "Up": 3}
        self.action_int_to_name = {0: "Left", 1: "Down", 2: "Right", 3: "Up"}

    def evaluate_reasoning_paths(self, data_file: str) -> Dict:
        """Evaluate if model's reasoning leads to successful goal completion"""

        results = {
            "successful_paths": 0,
            "total_paths": 0,
            "success_rate": 0.0,
            "grid_size_breakdown": defaultdict(lambda: {"success": 0, "total": 0}),
            "errors": [],
        }

        data = self._load_jsonl(data_file)

        for i, item in enumerate(data):
            try:
                # Extract prompt and response from the same item
                prompt = item.get("prompt", item.get("input", ""))
                response = item.get("response", item.get("output", ""))

                if not prompt or not response:
                    results["errors"].append(f"Item {i}: Missing prompt or response")
                    results["total_paths"] += 1
                    continue

                # Extract the full reasoning from model output
                reasoning = self._extract_reasoning({"response": response})

                # Extract action sequence from reasoning
                actions = self._extract_action_sequence(reasoning)

                # Get the grid from prompt
                grid = self._extract_grid_from_prompt(prompt)

                # Simulate the path
                success = self._simulate_path(grid, actions)

                results["successful_paths"] += int(success)
                results["total_paths"] += 1

                # Track by grid size
                grid_size = (
                    len(grid)
                    if grid
                    else item.get("metadata", {}).get("grid_size", "unknown")
                )
                results["grid_size_breakdown"][grid_size]["success"] += int(success)
                results["grid_size_breakdown"][grid_size]["total"] += 1

                if not success:
                    results["errors"].append(
                        f"Path {i}: Failed to reach goal with actions {actions}"
                    )

            except Exception as e:
                results["errors"].append(f"Path {i}: Error - {str(e)}")
                results["total_paths"] += 1  # Still count failed attempts
                continue

        results["success_rate"] = (
            results["successful_paths"] / results["total_paths"]
            if results["total_paths"] > 0
            else 0.0
        )
        return results

    def _load_jsonl(self, filepath: str) -> List[Dict]:
        """Load JSONL file"""
        data = []
        with open(filepath, "r") as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line.strip()))
        return data

    def _extract_reasoning(self, model_output: Dict) -> str:
        """Extract reasoning from model output"""
        # Check different possible keys where reasoning might be stored
        if "response" in model_output:
            response_text = model_output["response"]
        elif "reasoning" in model_output:
            response_text = model_output["reasoning"]
        elif "output" in model_output:
            response_text = model_output["output"]
        else:
            # If no specific key, try the whole output
            response_text = str(model_output)

        # Extract content within <think> tags
        think_match = re.search(
            r"<think>(.*?)</think>", response_text, re.DOTALL | re.IGNORECASE
        )
        if think_match:
            return think_match.group(1).strip()

        # If no <think> tags, return the whole response
        return response_text

    def _extract_action_sequence(self, reasoning: str) -> List[int]:
        """Extract action sequence from BFS reasoning"""
        actions = []

        # Find all "Step X: action ActionName" patterns (your new format)
        step_matches = re.findall(r"Step \d+: action (\w+)", reasoning, re.IGNORECASE)

        for action_name in step_matches:
            action_name = action_name.strip().title()  # Normalize case
            if action_name in self.action_name_to_int:
                actions.append(self.action_name_to_int[action_name])
            else:
                print(f"Warning: Unknown action name '{action_name}'")

        return actions

    def _extract_grid_from_prompt(self, prompt: str) -> List[List[str]]:
        """Extract grid from prompt text"""
        try:
            # Find the grid section
            if "Grid:" in prompt:
                grid_section = prompt.split("Grid:")[1]
            else:
                raise ValueError("No Grid: section found in prompt")

            # Stop at the next question
            if "What action" in grid_section:
                grid_section = grid_section.split("What action")[0]

            grid_section = grid_section.strip()
            lines = [line.strip() for line in grid_section.split("\n") if line.strip()]

            grid = []
            for line in lines:
                if " " in line:  # Grid rows should have spaces between cells
                    row = line.split()
                    if row and all(
                        cell in ["S", "F", "H", "G", "@"] for cell in row
                    ):  # Valid symbols
                        grid.append(row)

            # Validate grid is square and not empty
            if (
                grid
                and len(grid) == len(grid[0])
                and all(len(row) == len(grid[0]) for row in grid)
            ):
                return grid
            else:
                raise ValueError(
                    f"Invalid grid structure: {len(grid)} rows, varying column lengths"
                )

        except Exception as e:
            raise ValueError(f"Grid extraction from prompt failed: {e}")

    def _simulate_path(self, grid: List[List[str]], actions: List[int]) -> bool:
        """Simulate the action sequence on the grid and check if goal is reached"""
        if not grid or not actions:
            return False

        try:
            # Create gym environment from grid
            env = self._create_gym_env_from_grid(grid)

            state = env.reset()
            if isinstance(state, tuple):
                state = state[0]

            for action in actions:
                if action not in [0, 1, 2, 3]:  # Valid gym actions
                    env.close()
                    return False

                state, reward, done, truncated, info = env.step(action)

                if done:
                    env.close()
                    return reward > 0  # Success if reward > 0 (reached goal)

            env.close()
            return False  # Didn't finish/reach goal

        except Exception as e:
            print(f"Simulation error: {e}")
            return False

    def _create_gym_env_from_grid(self, grid: List[List[str]]):
        """Create gym environment from grid (using S, F, H, G format)"""
        desc = []
        for row in grid:
            row_string = ""
            for cell in row:
                if cell == "@":  # Player marker, treat as start
                    row_string += "S"
                elif cell in ["S", "F", "H", "G"]:  # Valid gym symbols
                    row_string += cell
                else:
                    raise ValueError(f"Unknown grid symbol: {cell}")
            desc.append(row_string)

        return gym.make(
            "FrozenLake-v1", desc=desc, is_slippery=False, render_mode="rgb_array"
        )

    def print_detailed_results(self, results: Dict):
        """Print detailed evaluation results"""
        print("\n=== FrozenLake Evaluation Results ===")
        print(f"Total paths evaluated: {results['total_paths']}")
        print(f"Successful paths: {results['successful_paths']}")
        print(f"Success rate: {results['success_rate']:.2%}")

        print("\n=== Breakdown by Grid Size ===")
        for size, stats in results["grid_size_breakdown"].items():
            success_rate = (
                stats["success"] / stats["total"] if stats["total"] > 0 else 0
            )
            print(
                f"Grid {size}x{size}: {stats['success']}/{stats['total']} ({success_rate:.2%})"
            )

        if results["errors"]:
            print("\n=== Errors (first 10) ===")
            for error in results["errors"][:10]:
                print(f"  {error}")
            if len(results["errors"]) > 10:
                print(f"  ... and {len(results['errors']) - 10} more errors")

    def save_results(self, results: Dict, output_path: str):
        """Save results to JSON file"""
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate FrozenLake model responses")
    parser.add_argument(
        "--data", type=str, help="Path to JSONL file containing prompts and responses"
    )
    parser.add_argument(
        "--output", type=str, help="Path to save evaluation results as JSON"
    )

    args = parser.parse_args()

    evaluator = FrozenLakeEvaluator()

    try:
        print(f"Loading model responses from: {args.data}")

        results = evaluator.evaluate_reasoning_paths(args.data)

        evaluator.print_detailed_results(results)

        if args.output:
            evaluator.save_results(results, args.output)

    except FileNotFoundError as e:
        print(f"Error: File not found - {e}")
        return 1
    except Exception as e:
        print(f"Error during evaluation: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
