"""
sft_evaluator.py

Evaluator class for benchmarking SFT Llama vs Base Llama performance on FrozenLake navigation tasks.

This script:
- Loads JSONL datasets with FrozenLake problems and expected actions
- Prompts both base and SFT models to predict next actions for each grid state
- Parses model outputs and compares against ground truth actions
- Logs detailed model responses, extracted actions, and correctness
- Computes and saves accuracy metrics by grid size for both models side-by-side

Designed for comparative evaluation of base vs fine-tuned Llama 8B models.
"""

import json
import re
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
from sft_llm_policy import SFTLLMPolicy
from base_llm_policy import BaseLLMPolicy

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ComparisonEvaluator:
    """Evaluator class for comparing SFT and Base Llama models on FrozenLake tasks."""
    
    def __init__(self):
        """Initialize the comparison evaluator."""
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
    
    def _evaluate_single_problem(self, problem: Dict[str, Any], 
                                base_policy: BaseLLMPolicy, 
                                sft_policy: SFTLLMPolicy) -> Dict[str, Any]:
        """Evaluate both base and SFT models on a single FrozenLake problem."""
        prompt = problem["prompt"]
        correct_action = problem["metadata"]["action_int"]
        correct_action_name = problem["metadata"]["action_name"]
        grid_size = problem["metadata"]["grid_size"]
        
        # Get responses from both models
        base_response = base_policy.respond(prompt)
        sft_response = sft_policy.respond(prompt)
        
        # Parse both responses
        base_parsed = self._parse_action_response(base_response)
        sft_parsed = self._parse_action_response(sft_response)
        
        # Check correctness for both
        base_correct = (base_parsed["parsed_action"] == correct_action) if base_parsed["parsing_success"] else False
        sft_correct = (sft_parsed["parsed_action"] == correct_action) if sft_parsed["parsing_success"] else False
        
        # Extract grid for logging
        grid_repr = self._extract_grid_from_prompt(prompt)
        
        result = {
            "grid_size": grid_size,
            "correct_action": correct_action,
            "correct_action_name": correct_action_name,
            "grid_repr": grid_repr,
            "optimal_path_length": problem["metadata"].get("optimal_path_length", "Unknown"),
            
            # Base model results
            "base_parsed_action": base_parsed["parsed_action"],
            "base_parsed_action_name": self.int_to_action.get(base_parsed["parsed_action"], "Unknown") if base_parsed["parsed_action"] is not None else "None",
            "base_correct": base_correct,
            "base_parsing_success": base_parsed["parsing_success"],
            "base_raw_response": base_parsed["raw_response"],
            
            # SFT model results
            "sft_parsed_action": sft_parsed["parsed_action"],
            "sft_parsed_action_name": self.int_to_action.get(sft_parsed["parsed_action"], "Unknown") if sft_parsed["parsed_action"] is not None else "None",
            "sft_correct": sft_correct,
            "sft_parsing_success": sft_parsed["parsing_success"],
            "sft_raw_response": sft_parsed["raw_response"],
        }
        
        return result
    
    def run_evaluation(self, base_policy: BaseLLMPolicy, sft_policy: SFTLLMPolicy, 
                      dataset_path: str) -> Dict[str, Any]:
        """Main evaluation loop comparing base and SFT models."""
        logger.info("Starting base vs SFT model comparison evaluation...")
        
        # Load dataset
        dataset = self.load_dataset(dataset_path)
        
        # Track results by grid size for both models
        size_results = {}
        
        # Evaluate each problem
        for i, problem in enumerate(dataset):
            result = self._evaluate_single_problem(problem, base_policy, sft_policy)
            self.results["detailed_results"].append(result)
            
            grid_size = result["grid_size"]
            if grid_size not in size_results:
                size_results[grid_size] = {
                    "base": {"correct": 0, "total": 0, "parsing_failures": 0},
                    "sft": {"correct": 0, "total": 0, "parsing_failures": 0}
                }
            
            # Update base model stats
            size_results[grid_size]["base"]["total"] += 1
            if result["base_correct"]:
                size_results[grid_size]["base"]["correct"] += 1
            if not result["base_parsing_success"]:
                size_results[grid_size]["base"]["parsing_failures"] += 1
            
            # Update SFT model stats
            size_results[grid_size]["sft"]["total"] += 1
            if result["sft_correct"]:
                size_results[grid_size]["sft"]["correct"] += 1
            if not result["sft_parsing_success"]:
                size_results[grid_size]["sft"]["parsing_failures"] += 1
            
            # Log progress
            if (i + 1) % 50 == 0:
                logger.info(f"Evaluated {i + 1}/{len(dataset)} problems")
            
            # Detailed logging for first few problems of each size
            if size_results[grid_size]["base"]["total"] <= 3:
                logger.info(f"\n--- Problem {i+1} ({grid_size}x{grid_size}) ---")
                logger.info(f"Grid:\n{result['grid_repr']}")
                logger.info(f"Correct Action: {result['correct_action_name']} ({result['correct_action']})")
                logger.info(f"Base Model: {result['base_parsed_action_name']} ({result['base_parsed_action']}) - {'✓' if result['base_correct'] else '✗'}")
                logger.info(f"SFT Model: {result['sft_parsed_action_name']} ({result['sft_parsed_action']}) - {'✓' if result['sft_correct'] else '✗'}")
                logger.info(f"Base response: {result['base_raw_response'][:150]}...")
                logger.info(f"SFT response: {result['sft_raw_response'][:150]}...")
        
        # Calculate summary statistics for both models
        self.results["by_grid_size"] = {}
        base_total_correct = 0
        sft_total_correct = 0
        total_problems = 0
        
        for size, stats in size_results.items():
            base_accuracy = stats["base"]["correct"] / stats["base"]["total"] if stats["base"]["total"] > 0 else 0
            sft_accuracy = stats["sft"]["correct"] / stats["sft"]["total"] if stats["sft"]["total"] > 0 else 0
            
            base_parse_rate = (stats["base"]["total"] - stats["base"]["parsing_failures"]) / stats["base"]["total"] if stats["base"]["total"] > 0 else 0
            sft_parse_rate = (stats["sft"]["total"] - stats["sft"]["parsing_failures"]) / stats["sft"]["total"] if stats["sft"]["total"] > 0 else 0
            
            self.results["by_grid_size"][size] = {
                "base": {
                    "accuracy": base_accuracy,
                    "correct": stats["base"]["correct"],
                    "total": stats["base"]["total"],
                    "parsing_success_rate": base_parse_rate,
                    "parsing_failures": stats["base"]["parsing_failures"]
                },
                "sft": {
                    "accuracy": sft_accuracy,
                    "correct": stats["sft"]["correct"],
                    "total": stats["sft"]["total"],
                    "parsing_success_rate": sft_parse_rate,
                    "parsing_failures": stats["sft"]["parsing_failures"]
                }
            }
            
            base_total_correct += stats["base"]["correct"]
            sft_total_correct += stats["sft"]["correct"]
            total_problems += stats["base"]["total"]  # Fixed: accumulate instead of overwrite
        
        # Overall summary
        base_overall_accuracy = base_total_correct / total_problems if total_problems > 0 else 0
        sft_overall_accuracy = sft_total_correct / total_problems if total_problems > 0 else 0
        
        self.results["summary"] = {
            "base": {
                "overall_accuracy": base_overall_accuracy,
                "total_correct": base_total_correct,
                "total_problems": total_problems
            },
            "sft": {
                "overall_accuracy": sft_overall_accuracy,
                "total_correct": sft_total_correct,
                "total_problems": total_problems
            },
            "improvement": sft_overall_accuracy - base_overall_accuracy,
            "grid_sizes_tested": list(size_results.keys())
        }
        
        # Log final results
        logger.info(f"\n=== Comparison Evaluation Complete ===")
        logger.info(f"Base Model Accuracy: {base_overall_accuracy:.2%} ({base_total_correct}/{total_problems})")
        logger.info(f"SFT Model Accuracy: {sft_overall_accuracy:.2%} ({sft_total_correct}/{total_problems})")
        logger.info(f"Improvement: {self.results['summary']['improvement']:.2%}")
        
        for size in sorted(size_results.keys()):
            base_stats = self.results["by_grid_size"][size]["base"]
            sft_stats = self.results["by_grid_size"][size]["sft"]
            logger.info(f"Grid {size}x{size}: Base {base_stats['accuracy']:.2%} vs SFT {sft_stats['accuracy']:.2%}")
        
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