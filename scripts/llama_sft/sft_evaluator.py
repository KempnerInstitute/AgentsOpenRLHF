"""
sft_evaluator.py

Evaluator class for benchmarking SFT Llama performance on FrozenLake navigation tasks.

This script:
- Loads JSONL datasets with FrozenLake problems and expected actions
- Prompts the SFT model to predict next actions for each grid state
- Parses model output and compares against ground truth actions
- Logs detailed model responses, extracted actions, and correctness
- Computes and saves accuracy metrics by grid size

Designed specifically for SFT Llama 8B evaluation on structured navigation data.
"""

import json
import re
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
from sft_llm_policy import SFTLLMPolicy

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SFTEvaluator:
    """Evaluator class for SFT Llama evaluation on FrozenLake tasks."""
    
    def __init__(self):
        """Initialize the SFT evaluator."""
        # Action mappings from your dataset
        self.action_mappings = {
            "up": 3, "down": 1, "left": 0, "right": 2,
            "u": 3, "d": 1, "l": 0, "r": 2
        }
        self.int_to_action = {3: "Up", 1: "Down", 0: "Left", 2: "Right"}
        
        # Results tracking
        self.results = {
            "by_grid_size": {},
            "detailed_results": [],
            "summary": {}
        }
    
    def load_dataset(self, dataset_path: str) -> List[Dict[str, Any]]:
        """Load JSONL dataset of FrozenLake problems."""
        dataset = []
        dataset_file = Path(dataset_path)
        
        if not dataset_file.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")
        
        with open(dataset_file, 'r') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    data = json.loads(line.strip())
                    dataset.append(data)
                except json.JSONDecodeError as e:
                    logger.warning(f"Skipping malformed line {line_num}: {e}")
        
        logger.info(f"Loaded {len(dataset)} problems from {dataset_path}")
        return dataset
    
    def _extract_grid_from_prompt(self, prompt: str) -> Optional[str]:
        """Extract grid representation from prompt for logging."""
        grid_match = re.search(r'Grid:\n((?:[SFHG ]+\n?)+)', prompt)
        if grid_match:
            return grid_match.group(1).strip()
        return None
    
    def _parse_action_response(self, response: str) -> Dict[str, Any]:
        """Parse action from model response."""
        # Look for <answer> tags first (primary format)
        answer_pattern = r'<answer>\s*([^<]*?)\s*</answer>'
        answer_matches = re.findall(answer_pattern, response, re.IGNORECASE)
        
        if answer_matches:
            # Take the last answer if multiple exist
            action_text = answer_matches[-1].strip().lower()
        else:
            # Fallback: look for common action words at the end
            action_words = ['up', 'down', 'left', 'right', 'u', 'd', 'l', 'r']
            words = response.lower().split()
            action_text = None
            
            # Find the last occurrence of an action word
            for word in reversed(words):
                clean_word = re.sub(r'[^\w]', '', word)
                if clean_word in action_words:
                    action_text = clean_word
                    break
        
        # Convert to action integer
        parsed_action = None
        if action_text and action_text in self.action_mappings:
            parsed_action = self.action_mappings[action_text]
        
        return {
            "raw_response": response,
            "parsed_action": parsed_action,
            "parsing_success": parsed_action is not None
        }
    
    def _evaluate_single_problem(self, problem: Dict[str, Any], policy: SFTLLMPolicy) -> Dict[str, Any]:
        """Evaluate model on a single FrozenLake problem."""
        prompt = problem["prompt"]
        correct_action = problem["metadata"]["action_int"]
        correct_action_name = problem["metadata"]["action_name"]
        grid_size = problem["metadata"]["grid_size"]
        
        # Get model response
        response = policy.respond(prompt)
        
        # Parse the response
        parsed = self._parse_action_response(response)
        
        # Check correctness
        correct = (parsed["parsed_action"] == correct_action) if parsed["parsing_success"] else False
        
        # Extract grid for logging
        grid_repr = self._extract_grid_from_prompt(prompt)
        
        result = {
            "grid_size": grid_size,
            "correct_action": correct_action,
            "correct_action_name": correct_action_name,
            "parsed_action": parsed["parsed_action"],
            "parsed_action_name": self.int_to_action.get(parsed["parsed_action"], "Unknown") if parsed["parsed_action"] is not None else "None",
            "correct": correct,
            "parsing_success": parsed["parsing_success"],
            "model_raw_response": parsed["raw_response"],
            "grid_repr": grid_repr,
            "optimal_path_length": problem["metadata"].get("optimal_path_length", "Unknown")
        }
        
        return result
    
    def run_evaluation(self, policy: SFTLLMPolicy, dataset_path: str) -> Dict[str, Any]:
        """Main evaluation loop."""
        logger.info("Starting SFT model evaluation...")
        
        # Load dataset
        dataset = self.load_dataset(dataset_path)
        
        # Track results by grid size
        size_results = {}
        
        # Evaluate each problem
        for i, problem in enumerate(dataset):
            result = self._evaluate_single_problem(problem, policy)
            self.results["detailed_results"].append(result)
            
            grid_size = result["grid_size"]
            if grid_size not in size_results:
                size_results[grid_size] = {"correct": 0, "total": 0, "parsing_failures": 0}
            
            size_results[grid_size]["total"] += 1
            if result["correct"]:
                size_results[grid_size]["correct"] += 1
            if not result["parsing_success"]:
                size_results[grid_size]["parsing_failures"] += 1
            
            # Log progress
            if (i + 1) % 50 == 0:
                logger.info(f"Evaluated {i + 1}/{len(dataset)} problems")
            
            # Detailed logging for first few problems of each size
            if size_results[grid_size]["total"] <= 3:
                logger.info(f"\n--- Problem {i+1} ({grid_size}x{grid_size}) ---")
                logger.info(f"Grid:\n{result['grid_repr']}")
                logger.info(f"Correct Action: {result['correct_action_name']} ({result['correct_action']})")
                logger.info(f"Parsed Action: {result['parsed_action_name']} ({result['parsed_action']})")
                logger.info(f"Model response: {result['model_raw_response'][:200]}...")
                logger.info(f"Result: {'CORRECT' if result['correct'] else 'INCORRECT'}")
        
        # Calculate summary statistics
        self.results["by_grid_size"] = {}
        total_correct = 0
        total_problems = 0
        
        for size, stats in size_results.items():
            accuracy = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
            parse_rate = (stats["total"] - stats["parsing_failures"]) / stats["total"] if stats["total"] > 0 else 0
            
            self.results["by_grid_size"][size] = {
                "accuracy": accuracy,
                "correct": stats["correct"],
                "total": stats["total"],
                "parsing_success_rate": parse_rate,
                "parsing_failures": stats["parsing_failures"]
            }
            
            total_correct += stats["correct"]
            total_problems += stats["total"]
        
        # Overall summary
        overall_accuracy = total_correct / total_problems if total_problems > 0 else 0
        self.results["summary"] = {
            "overall_accuracy": overall_accuracy,
            "total_correct": total_correct,
            "total_problems": total_problems,
            "grid_sizes_tested": list(size_results.keys())
        }
        
        # Log final results
        logger.info(f"\n=== Evaluation Complete ===")
        logger.info(f"Overall Accuracy: {overall_accuracy:.2%} ({total_correct}/{total_problems})")
        
        for size in sorted(size_results.keys()):
            stats = self.results["by_grid_size"][size]
            logger.info(f"Grid {size}x{size}: {stats['accuracy']:.2%} ({stats['correct']}/{stats['total']})")
        
        return self.results
    
    def save_results(self, base_filepath: str) -> None:
        """Save evaluation results to separate JSON files."""
        base_path = Path(base_filepath)
        base_name = base_path.stem
        base_dir = base_path.parent
        
        # Save summary results (lightweight)
        summary_results = {
            "by_grid_size": self.results["by_grid_size"],
            "summary": self.results["summary"]
        }
        summary_filepath = base_dir / f"{base_name}_summary.json"
        with open(summary_filepath, 'w') as f:
            json.dump(summary_results, f, indent=2)
        logger.info(f"Summary results saved to {summary_filepath}")
        
        # Save detailed results (includes all raw responses)
        detailed_filepath = base_dir / f"{base_name}_detailed.json"
        with open(detailed_filepath, 'w') as f:
            json.dump(self.results, f, indent=2)
        logger.info(f"Detailed results saved to {detailed_filepath}")
        
        return str(summary_filepath), str(detailed_filepath)