import argparse
import json
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os
from collections import defaultdict
from typing import Dict, List, Optional
from tqdm import tqdm

import numpy as np

from .sokoban_data_gen import SokobanConfig, SokobanEnv
from .sokoban_constants import Action, GRID_CONFIG, CELL_TYPES
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from .sokoban_utils import make_prompt, parse_action, parse_map_to_arrays, create_env_from_map


class SokobanEvaluator:
    def __init__(self, max_steps: int = 100):
        self.max_steps = max_steps

        # Use centralized grid config
        self.vocab_to_env = GRID_CONFIG.symbol_to_env
    
    def _check_corner_deadlock(self, env) -> bool:
        state = env.room_state
        fixed = env.room_fixed
        h, w = state.shape

        for i in range(h):
            for j in range(w):
                if state[i, j] == CELL_TYPES.BOX:
                    if (
                        (self._is_wall(state, i - 1, j) and self._is_wall(state, i, j - 1)) or
                        (self._is_wall(state, i - 1, j) and self._is_wall(state, i, j + 1)) or
                        (self._is_wall(state, i + 1, j) and self._is_wall(state, i, j - 1)) or
                        (self._is_wall(state, i + 1, j) and self._is_wall(state, i, j + 1))
                    ):
                        if fixed[i, j] != CELL_TYPES.TARGET:
                            return True
        return False

    def _is_wall(self, state: np.ndarray, i: int, j: int) -> bool:
        if 0 <= i < state.shape[0] and 0 <= j < state.shape[1]:
            return state[i, j] == CELL_TYPES.WALL
        return True  # treat out-of-bounds as wall

    def _build_result(
        self, puzzle_id: int, success: bool, steps: int,
        actions: List[int], logs: List[Dict], reason: Optional[str] = None
    ) -> Dict:
        """Helper to build consistent result dictionaries."""
        result = {
            "puzzle_id": puzzle_id,
            "success": success,
            "steps": steps,
            "actions": actions,
            "log": logs
        }
        if reason:
            result["reason"] = reason
        return result

    def interactive_evaluate(self, map_str: str, model_fn, puzzle_id: int, max_steps: Optional[int] = None) -> Dict:
        env = create_env_from_map(map_str, self.max_steps)
        if env is None:
            return {"puzzle_id": puzzle_id, "error": "Invalid map"}

        max_steps = max_steps or self.max_steps
        logs = []
        actions_taken = []

        for step_num in range(max_steps):
            state_str = env.render_text()
            prompt = make_prompt(state_str)

            raw_response = model_fn(prompt)
            action_id = parse_action(raw_response)

            logs.append({
                "puzzle_id": puzzle_id,
                "step": step_num,
                "prompt": prompt,
                "response": raw_response
            })

            if action_id is None:
                return self._build_result(
                    puzzle_id, False, step_num, actions_taken, logs,
                    f"Could not extract action from response: '{raw_response}'"
                )

            actions_taken.append(action_id)
            _, _, done, info = env.step(action_id)

            if self._check_corner_deadlock(env):
                return self._build_result(
                    puzzle_id, False, step_num + 1, actions_taken, logs,
                    "Corner deadlock detected"
                )

            if done:
                success = info.get("all_boxes_on_target", False)
                return self._build_result(
                    puzzle_id, success, step_num + 1, actions_taken, logs
                )

        # Max steps reached - check final state
        success = env._check_if_all_boxes_on_target()
        return self._build_result(
            puzzle_id, success, max_steps, actions_taken, logs,
            "Max steps reached"
        )


    def _simulate_puzzle(
        self, env: SokobanEnv, actions: List[int], results: Dict
    ) -> bool:
        """Simulate the action sequence and return success status"""
        if not actions:
            return False

        try:
            valid_action_ids = [a.action_id for a in Action.all_actions()]
            for step_num, action in enumerate(actions):
                if action not in valid_action_ids:
                    results["invalid_action_count"] += 1
                    return False

                # Execute action
                obs, reward, done, info = env.step(action)

                if done:
                    # Check if successfully solved
                    success = info.get("all_boxes_on_target", False)
                    return success

            # Check final state even if not done
            final_success = env._check_if_all_boxes_on_target()
            return final_success

        except Exception as e:
            import traceback

            traceback.print_exc()
            return False

    def print_detailed_results(self, results: Dict):
        """Print detailed evaluation results"""
        print("\n=== Sokoban Evaluation Results ===")
        print(f"Total puzzles evaluated: {results['total_puzzles']}")
        print(f"Successful puzzles: {results['successful_puzzles']}")
        print(f"Success rate: {results['success_rate']:.2%}")
        if results.get("deadlock_total", 0) > 0:
            d_success = results["deadlock_success"]
            d_total = results["deadlock_total"]
            print(f"\nCorrectly identified deadlocks: {d_success}/{d_total} (rate: {d_success / d_total:.2%})")

        # Add error summary
        if results["errors"]:
            print(f"Total errors: {len(results['errors'])}")

        if results["invalid_action_count"] > 0:
            print(f"total invalid actions: {results['invalid_action_count']}")

    def save_results(self, results: Dict, output_path: str):
        """Save results to JSON file"""
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to: {output_path}")

    def evaluate_batch_parallel(
        self,
        test_data: List[Dict],
        model_fn,
        num_workers: int = 4,
        max_steps: Optional[int] = None
    ) -> List[tuple[Dict, Dict]]:
        """
        Evaluate multiple puzzles in parallel using multiprocessing.

        Args:
            test_data: List of puzzle dictionaries with 'map' and optional 'metadata'
            model_fn: Function that takes prompt string and returns response
            num_workers: Number of parallel workers (default: 4)
            max_steps: Maximum steps per puzzle (default: uses self.max_steps)

        Returns:
            List of (result, metadata) tuples

        Note:
            This uses threading with a shared model_fn, which works well for
            CPU-bound inference or when the model handles concurrency internally.
            For GPU-based models, ensure model_fn is thread-safe.
        """
        max_steps = max_steps or self.max_steps

        # Prepare arguments for each puzzle
        puzzle_args = [
            (puzzle_data, idx, max_steps)
            for idx, puzzle_data in enumerate(test_data)
        ]

        # Use ThreadPoolExecutor since model_fn is shared
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            # Create partial function with model_fn bound
            def worker(args):
                puzzle_data, puzzle_id, max_steps_arg = args
                return _evaluate_single_puzzle_worker(
                    puzzle_data, puzzle_id, model_fn, max_steps_arg
                )

            results = list(tqdm(
                executor.map(worker, puzzle_args),
                total=len(puzzle_args),
                desc="Evaluating puzzles in parallel"
            ))

        return results


# Module-level worker function for multiprocessing/threading
def _evaluate_single_puzzle_worker(
    puzzle_data: Dict,
    puzzle_id: int,
    model_fn,
    max_steps: int
) -> tuple[Dict, Dict]:
    """
    Worker function to evaluate a single puzzle.

    Args:
        puzzle_data: Dictionary with 'map' key and optional 'metadata'
        puzzle_id: Puzzle identifier
        model_fn: Function that takes prompt and returns response
        max_steps: Maximum steps for this puzzle

    Returns:
        (result_dict, metadata_dict) tuple
    """
    try:
        evaluator = SokobanEvaluator(max_steps=max_steps)
        result = evaluator.interactive_evaluate(
            puzzle_data["map"],
            model_fn,
            puzzle_id,
            max_steps
        )
        return result, puzzle_data.get("metadata", {})

    except Exception as e:
        return {
            "puzzle_id": puzzle_id,
            "error": f"Evaluation error: {str(e)}",
            "success": False,
            "steps": 0,
            "actions": [],
            "log": []
        }, puzzle_data.get("metadata", {})


def aggregate_parallel_results(
    results: List[tuple[Dict, Dict]]
) -> Dict:
    """
    Aggregate results from parallel evaluation into summary statistics.

    Args:
        results: List of (result, metadata) tuples from parallel evaluation

    Returns:
        Dictionary with aggregated statistics including:
        - total_puzzles, successful_puzzles, success_rate
        - grid_size_breakdown, box_count_breakdown
        - errors
    """
    total_puzzles = len(results)
    successful_puzzles = 0
    grid_size_breakdown = defaultdict(lambda: {"success": 0, "total": 0})
    box_count_breakdown = defaultdict(lambda: {"success": 0, "total": 0})
    errors = []
    all_logs = []

    for result, metadata in results:
        puzzle_id = result.get("puzzle_id", 0)

        # Track errors
        if "error" in result:
            errors.append(f"Puzzle {puzzle_id}: {result['error']}")
            continue

        # Collect logs
        for log_entry in result.get("log", []):
            all_logs.append(log_entry)

        # Check success once for all tracking
        is_success = result.get("success", False)
        if is_success:
            successful_puzzles += 1

        # Grid size breakdown
        grid_size = metadata.get("grid_size", [0, 0])
        if isinstance(grid_size, list) and len(grid_size) == 2:
            size_str = f"{grid_size[0]}x{grid_size[1]}"
        else:
            size_str = str(grid_size)

        grid_size_breakdown[size_str]["total"] += 1
        if is_success:
            grid_size_breakdown[size_str]["success"] += 1

        # Box count breakdown
        num_boxes = metadata.get("num_boxes", 0)
        box_count_breakdown[num_boxes]["total"] += 1
        if is_success:
            box_count_breakdown[num_boxes]["success"] += 1

    # Calculate success rate
    success_rate = successful_puzzles / total_puzzles if total_puzzles > 0 else 0

    return {
        "total_puzzles": total_puzzles,
        "successful_puzzles": successful_puzzles,
        "success_rate": success_rate,
        "grid_size_breakdown": dict(grid_size_breakdown),
        "box_count_breakdown": dict(box_count_breakdown),
        "errors": errors,
        "logs": all_logs
    }


def save_parallel_results(
    results: List[tuple[Dict, Dict]],
    interaction_output: str,
    summary_output: str
):
    """
    Save parallel evaluation results to files.

    Args:
        results: List of (result, metadata) tuples
        interaction_output: Path to save detailed interaction logs (JSONL)
        summary_output: Path to save summary statistics (JSONL)
    """
    # Aggregate statistics
    summary = aggregate_parallel_results(results)

    # Save interaction logs
    with open(interaction_output, "w") as f:
        for log_entry in summary["logs"]:
            f.write(json.dumps(log_entry) + "\n")

    # Save summary
    with open(summary_output, "w") as f:
        # Write overall summary header
        header = {
            "solved": summary["successful_puzzles"],
            "total": summary["total_puzzles"],
            "success_rate": summary["success_rate"],
            "grid_size_breakdown": summary["grid_size_breakdown"],
            "box_count_breakdown": summary["box_count_breakdown"]
        }
        f.write(json.dumps(header) + "\n")

        # Write individual puzzle results
        for result, metadata in results:
            puzzle_id = result.get("puzzle_id", 0)
            actions = result.get("actions", [])

            if "error" in result:
                line = f"Puzzle {puzzle_id}: ERROR - {result['error']}"
            elif result.get("success", False):
                line = f"Puzzle {puzzle_id}: SOLVED in {result.get('steps', 0)} steps with actions {actions}"
            else:
                reason = result.get("reason", "unknown")
                line = f"Puzzle {puzzle_id}: FAILED after {result.get('steps', 0)} steps (reason: {reason})"

            f.write(line + "\n")

    print(f"Saved interaction logs to: {interaction_output}")
    print(f"Saved summary to: {summary_output}")
    print(f"Overall success rate: {summary['success_rate']:.2%}")
    print(f"Successful: {summary['successful_puzzles']}/{summary['total_puzzles']}")

