"""
sft_evaluator.py

Evaluator class for benchmarking SFT Llama vs Base Llama performance on FrozenLake navigation tasks.

This script:
- Loads JSONL datasets with FrozenLake problems and expected actions
- Prompts both base and SFT models to predict complete action sequences for grid navigation
- Tests proposed solutions in actual Gymnasium FrozenLake environments
- Evaluates based on successful goal completion rather than optimal path matching
- Computes and saves success rates by grid size for both models side-by-side

Designed for comparative evaluation of base vs fine-tuned Llama 8B models using trajectory completion.
"""

import json
import re
import logging
import gymnasium as gym
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import wandb
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime
from sft_llm_policy import SFTLLMPolicy
from base_llm_policy import BaseLLMPolicy

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TrajectoryEvaluator:
    """Evaluator class for comparing SFT and Base Llama models on FrozenLake trajectory completion."""
    
    def __init__(self, use_wandb: bool = True, wandb_project: str = "frozenlake-evaluation"):
        """Initialize the trajectory evaluator."""
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
        
        # Weights & Biases setup
        self.use_wandb = use_wandb
        if self.use_wandb:
            try:
                wandb.init(
                    project=wandb_project,
                    name=f"trajectory_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    tags=["trajectory-completion", "base-vs-sft"]
                )
                logger.info("Initialized Weights & Biases logging")
            except Exception as e:
                logger.warning(f"Failed to initialize W&B: {e}. Continuing without W&B logging.")
                self.use_wandb = False
    
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
    
    def _extract_grid_from_prompt(self, prompt: str) -> Optional[List[str]]:
        """Extract grid representation from prompt and convert to map format (matching their approach)."""
        try:
            # Find the grid section
            if "Grid:" in prompt:
                grid_section = prompt.split("Grid:")[1]
            else:
                return None

            # Stop at the next question or section
            if "What action" in grid_section:
                grid_section = grid_section.split("What action")[0]
            elif "TASK:" in grid_section:
                grid_section = grid_section.split("TASK:")[0]
            elif "Plan your" in grid_section:
                grid_section = grid_section.split("Plan your")[0]

            grid_section = grid_section.strip()
            lines = [line.strip() for line in grid_section.split("\n") if line.strip()]

            # Convert spaced grid format to list of strings
            grid_desc = []
            for line in lines:
                if " " in line:  # Grid rows should have spaces between cells
                    row = line.split()
                    if row and all(cell in ["S", "F", "H", "G", "@"] for cell in row):  # Valid symbols
                        grid_desc.append("".join(row))  # Join without spaces for gym

            # Validate grid is square and not empty
            if (grid_desc and len(grid_desc) == len(grid_desc[0]) and 
                all(len(row) == len(grid_desc[0]) for row in grid_desc)):
                return grid_desc
            else:
                return None

        except Exception as e:
            logger.error(f"Grid extraction from prompt failed: {e}")
            return None
    
    def _generate_trajectory_prompt(self, grid_desc: List[str]) -> str:
        """Generate a simple prompt for complete trajectory (similar to their approach)."""
        grid_str = '\n'.join([' '.join(row) for row in grid_desc])  # Add spaces like theirs
        
        prompt = f"""You are walking on a frozen lake.

FrozenLake Quick Guide
Goal: Reach the goal (G)

Symbols:
S Start | F Frozen | H Hole | G Goal

Rules:
- Avoid falling into holes (H)
- You need to navigate from S to G

Grid:
{grid_str}

Plan your complete path from S to G. Think step by step about each move.

Output your reasoning in <think></think> tags, then provide your final answer.

Example reasoning format:
<think>
Step 1: action Right. Reason: gets closer to goal
Step 2: action Down. Reason: reaches the goal
</think>

<answer>Right Down</answer>"""

        return prompt
    
    def _parse_trajectory_response(self, response: str) -> Dict[str, Any]:
        """Parse action sequence from response (matching their exact approach)."""
        
        # Extract reasoning from <think> tags (their primary method)
        think_match = re.search(r'<think>(.*?)</think>', response, re.DOTALL | re.IGNORECASE)
        reasoning = think_match.group(1).strip() if think_match else ""
        
        # Extract actions from reasoning using their exact method
        actions = []
        action_words = []
        
        if reasoning:
            # Find all "Step X: action ActionName" patterns (their exact format)
            step_matches = re.findall(r'Step \d+: action (\w+)', reasoning, re.IGNORECASE)
            
            for action_name in step_matches:
                action_name = action_name.strip().title()  # Their normalization
                action_name_lower = action_name.lower()
                if action_name_lower in self.action_mappings:
                    action_words.append(action_name)
                    actions.append(self.action_mappings[action_name_lower])
                else:
                    logger.warning(f"Unknown action name '{action_name}'")
        
        # Fallback: try to extract from <answer> tags if no reasoning found
        if not actions:
            answer_pattern = r'<answer>\s*([^<]*?)\s*(?:</answer>|$)'
            answer_matches = re.findall(answer_pattern, response, re.IGNORECASE | re.DOTALL)
            
            if answer_matches:
                answer_text = answer_matches[-1].strip()
                # Split by spaces (their simple approach)
                parts = answer_text.split()
                
                for part in parts:
                    clean_part = re.sub(r'[^\w]', '', part).lower()
                    if clean_part in self.action_mappings:
                        action_words.append(clean_part.title())
                        actions.append(self.action_mappings[clean_part])
        
        # Their approach: no safety limit on actions, let it fail naturally if too long
        
        return {
            "raw_response": response,
            "reasoning": reasoning,
            "trajectory_text": " ".join(action_words),
            "action_words": action_words,
            "valid_actions": actions,
            "parsing_success": len(actions) > 0,
            "trajectory_length": len(actions)
        }
    
    def _execute_single_step_policy(self, map_desc: List[str], policy, max_steps: int = 50) -> Dict[str, Any]:
        """Execute policy step-by-step, asking for one action at a time."""
        try:
            # Create gym environment
            env = gym.make('FrozenLake-v1', desc=map_desc, is_slippery=False, render_mode=None)
            observation, info = env.reset()
            
            actions_taken = []
            action_words = []
            total_reward = 0
            steps_taken = 0
            success = False
            
            for step in range(max_steps):
                # Get current state info
                current_pos = self._get_position_from_observation(observation, map_desc)
                
                # Generate single-step prompt
                prompt = self._generate_single_step_prompt(map_desc, current_pos)
                
                # Get model response
                response = policy.respond(prompt)
                
                # Parse single action
                parsed = self._parse_single_action_response(response)
                
                if not parsed["parsing_success"]:
                    break
                
                action = parsed["action"]
                action_words.append(parsed["action_word"])
                actions_taken.append(action)
                
                # Execute action in environment
                observation, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                steps_taken += 1
                
                if terminated or truncated:
                    success = (reward > 0)
                    break
            
            env.close()
            
            return {
                "success": success,
                "total_reward": total_reward,
                "steps_taken": steps_taken,
                "actions_taken": actions_taken,
                "action_words": action_words,
                "completed": terminated or truncated if 'terminated' in locals() else False,
                "max_steps_reached": steps_taken >= max_steps
            }
            
        except Exception as e:
            logger.error(f"Error in single-step execution: {e}")
            return {
                "success": False,
                "total_reward": 0,
                "steps_taken": 0,
                "actions_taken": [],
                "action_words": [],
                "completed": False,
                "max_steps_reached": False,
                "error": str(e)
            }
    
    def _get_position_from_observation(self, observation: int, map_desc: List[str]) -> tuple:
        """Convert gym observation to grid position."""
        rows, cols = len(map_desc), len(map_desc[0])
        row = observation // cols
        col = observation % cols
        return (row, col)
    
    def _parse_single_action_response(self, response: str) -> Dict[str, Any]:
        """Parse single action from response (matches SFT training format)."""
        # Look for <answer> tags
        answer_pattern = r'<answer>\s*([^<]*?)\s*(?:</answer>|$)'
        answer_matches = re.findall(answer_pattern, response, re.IGNORECASE)
        
        action_text = ""
        if answer_matches:
            action_text = answer_matches[-1].strip().lower()
        else:
            # Fallback: look for action words
            for word in ['up', 'down', 'left', 'right']:
                if word in response.lower():
                    action_text = word
                    break
        
        # Convert to action integer
        action = None
        if action_text in self.action_mappings:
            action = self.action_mappings[action_text]
        
        return {
            "raw_response": response,
            "action_word": action_text.title() if action_text else "",
            "action": action,
            "parsing_success": action is not None
        }
    
    def _evaluate_single_problem(self, problem: Dict[str, Any], 
                                base_policy: BaseLLMPolicy, 
                                sft_policy: SFTLLMPolicy) -> Dict[str, Any]:
        """Evaluate both base and SFT models on a single FrozenLake problem."""
        # Extract grid from the original prompt
        grid_desc = self._extract_grid_from_prompt(problem["prompt"])
        if not grid_desc:
            logger.error("Could not extract grid from prompt")
            return None
        
        grid_size = problem["metadata"]["grid_size"]
        
        # Use same trajectory-based approach for both models (like their code)
        trajectory_prompt = self._generate_trajectory_prompt(grid_desc)
        
        # Get responses from both models
        base_response = base_policy.respond(trajectory_prompt)
        sft_response = sft_policy.respond(trajectory_prompt)
        
        # Parse both responses using the same method
        base_parsed = self._parse_trajectory_response(base_response)
        sft_parsed = self._parse_trajectory_response(sft_response)
        
        # Test trajectories in environment
        base_result = None
        sft_result = None
        
        if base_parsed["parsing_success"] and base_parsed["valid_actions"]:
            base_result = self._test_trajectory_in_env(grid_desc, base_parsed["valid_actions"])
        
        if sft_parsed["parsing_success"] and sft_parsed["valid_actions"]:
            sft_result = self._test_trajectory_in_env(grid_desc, sft_parsed["valid_actions"])
        
        result = {
            "grid_size": grid_size,
            "grid_desc": grid_desc,
            "optimal_path_length": problem["metadata"].get("optimal_path_length", "Unknown"),
            
            # Base model results
            "base_reasoning": base_parsed.get("reasoning", ""),
            "base_trajectory_text": base_parsed["trajectory_text"],
            "base_action_words": base_parsed["action_words"],
            "base_valid_actions": base_parsed["valid_actions"],
            "base_parsing_success": base_parsed["parsing_success"],
            "base_trajectory_length": base_parsed["trajectory_length"],
            "base_raw_response": base_parsed["raw_response"],
            "base_env_result": base_result,
            "base_success": base_result["success"] if base_result else False,
            
            # SFT model results
            "sft_reasoning": sft_parsed.get("reasoning", ""),
            "sft_trajectory_text": sft_parsed["trajectory_text"],
            "sft_action_words": sft_parsed["action_words"],
            "sft_valid_actions": sft_parsed["valid_actions"],
            "sft_parsing_success": sft_parsed["parsing_success"],
            "sft_trajectory_length": sft_parsed["trajectory_length"],
            "sft_raw_response": sft_parsed["raw_response"],
            "sft_env_result": sft_result,
            "sft_success": sft_result["success"] if sft_result else False,
        }
        
        return result
    
    def _generate_trajectory_prompt(self, grid_desc: List[str]) -> str:
        """Generate a prompt asking for complete trajectory solution (for base model)."""
        grid_str = '\n'.join(grid_desc)
        
        prompt = f"""You are solving a frozen lake navigation puzzle. Find the path from S to G.

SYMBOLS:
- S = Start (your current position)
- F = Frozen surface (safe to walk on)  
- H = Hole (dangerous - avoid!)
- G = Goal (destination)

ACTIONS: Up, Down, Left, Right

GRID:
{grid_str}

TASK: Navigate from S to G using the shortest safe path.

Give me ONLY the sequence of moves separated by |

EXAMPLE FORMAT: 
<answer>Right | Down | Right</answer>

YOUR SOLUTION:
<answer>"""

        return prompt
    
    def _test_trajectory_in_env(self, map_desc: List[str], actions: List[int], max_steps: int = 100) -> Dict[str, Any]:
        """Test action sequence in actual FrozenLake environment."""
        try:
            # Create gym environment with the specific map
            env = gym.make('FrozenLake-v1', desc=map_desc, is_slippery=False, render_mode=None)
            
            observation, info = env.reset()
            total_reward = 0
            steps_taken = 0
            success = False
            
            for action in actions:
                if steps_taken >= max_steps:
                    break
                    
                observation, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                steps_taken += 1
                
                if terminated or truncated:
                    success = (reward > 0)  # Success if reached goal
                    break
            
            env.close()
            
            return {
                "success": success,
                "total_reward": total_reward,
                "steps_taken": steps_taken,
                "completed": terminated or truncated if 'terminated' in locals() else False,
                "max_steps_reached": steps_taken >= max_steps
            }
            
        except Exception as e:
            logger.error(f"Error testing trajectory in environment: {e}")
            return {
                "success": False,
                "total_reward": 0,
                "steps_taken": 0,
                "completed": False,
                "max_steps_reached": False,
                "error": str(e)
            }
    
    def run_evaluation(self, base_policy: BaseLLMPolicy, sft_policy: SFTLLMPolicy, 
                      dataset_path: str) -> Dict[str, Any]:
        """Main evaluation loop comparing base and SFT models on trajectory completion."""
        logger.info("Starting base vs SFT model trajectory completion evaluation...")
        
        # Load dataset
        dataset = self.load_dataset(dataset_path)
        
        # Track results by grid size for both models
        size_results = {}
        
        # Evaluate each problem
        for i, problem in enumerate(dataset):
            result = self._evaluate_single_problem(problem, base_policy, sft_policy)
            if result is None:
                continue
                
            self.results["detailed_results"].append(result)
            
            grid_size = result["grid_size"]
            if grid_size not in size_results:
                size_results[grid_size] = {
                    "base": {"success": 0, "total": 0, "parsing_failures": 0},
                    "sft": {"success": 0, "total": 0, "parsing_failures": 0}
                }
            
            # Update base model stats
            size_results[grid_size]["base"]["total"] += 1
            if result["base_success"]:
                size_results[grid_size]["base"]["success"] += 1
            if not result["base_parsing_success"]:
                size_results[grid_size]["base"]["parsing_failures"] += 1
            
            # Update SFT model stats
            size_results[grid_size]["sft"]["total"] += 1
            if result["sft_success"]:
                size_results[grid_size]["sft"]["success"] += 1
            if not result["sft_parsing_success"]:
                size_results[grid_size]["sft"]["parsing_failures"] += 1
            
            # Log progress with success rates
            if (i + 1) % 25 == 0:
                current_base_successes = sum(r["base_success"] for r in self.results["detailed_results"])
                current_sft_successes = sum(r["sft_success"] for r in self.results["detailed_results"])
                base_rate = current_base_successes / (i + 1) * 100
                sft_rate = current_sft_successes / (i + 1) * 100
                logger.info(f"Evaluated {i + 1}/{len(dataset)} problems - Base: {base_rate:.1f}% | SFT: {sft_rate:.1f}%")
            
            # Enhanced logging conditions
            should_log_detailed = (
                size_results[grid_size]["base"]["total"] <= 2 or  # First 2 of each size
                (i < 100 and (not result['base_success'] or not result['sft_success'])) or  # Extended early failure window
                (i % 200 == 0)  # Every 200th problem for ongoing monitoring
            )
            
            if should_log_detailed:
                base_actions_display = result['base_action_words'][:5] if len(result['base_action_words']) > 5 else result['base_action_words']
                if len(result['base_action_words']) > 5:
                    base_actions_display = base_actions_display + [f"...+{len(result['base_action_words'])-5} more"]
                
                logger.info(f"\n--- Problem {i+1} ({grid_size}x{grid_size}) ---")
                logger.info(f"Grid: {' '.join(result['grid_desc'])}")
                logger.info(f"Base Model: {base_actions_display} -> {'SUCCESS' if result['base_success'] else 'FAILED'}")
                logger.info(f"SFT Model: {result['sft_action_words']} -> {'SUCCESS' if result['sft_success'] else 'FAILED'}")
                
                # Always log raw responses for early problems to debug
                if i < 20:
                    logger.info(f"Base raw response: {result['base_raw_response'][:150]}...")
                    logger.info(f"SFT raw response: {result['sft_raw_response'][:150]}...")
            
            # Log to W&B every 10 problems
            if self.use_wandb and (i + 1) % 10 == 0:
                current_base_rate = size_results[grid_size]["base"]["success"] / size_results[grid_size]["base"]["total"]
                current_sft_rate = size_results[grid_size]["sft"]["success"] / size_results[grid_size]["sft"]["total"]
                
                wandb.log({
                    "problems_evaluated": i + 1,
                    f"base_success_rate_{grid_size}x{grid_size}": current_base_rate,
                    f"sft_success_rate_{grid_size}x{grid_size}": current_sft_rate,
                    f"improvement_{grid_size}x{grid_size}": current_sft_rate - current_base_rate,
                    "step": i + 1
                })
        
        # Calculate summary statistics for both models
        self.results["by_grid_size"] = {}
        base_total_success = 0
        sft_total_success = 0
        total_problems = 0
        
        for size, stats in size_results.items():
            base_success_rate = stats["base"]["success"] / stats["base"]["total"] if stats["base"]["total"] > 0 else 0
            sft_success_rate = stats["sft"]["success"] / stats["sft"]["total"] if stats["sft"]["total"] > 0 else 0
            
            base_parse_rate = (stats["base"]["total"] - stats["base"]["parsing_failures"]) / stats["base"]["total"] if stats["base"]["total"] > 0 else 0
            sft_parse_rate = (stats["sft"]["total"] - stats["sft"]["parsing_failures"]) / stats["sft"]["total"] if stats["sft"]["total"] > 0 else 0
            
            self.results["by_grid_size"][size] = {
                "base": {
                    "success_rate": base_success_rate,
                    "successful": stats["base"]["success"],
                    "total": stats["base"]["total"],
                    "parsing_success_rate": base_parse_rate,
                    "parsing_failures": stats["base"]["parsing_failures"]
                },
                "sft": {
                    "success_rate": sft_success_rate,
                    "successful": stats["sft"]["success"],
                    "total": stats["sft"]["total"],
                    "parsing_success_rate": sft_parse_rate,
                    "parsing_failures": stats["sft"]["parsing_failures"]
                }
            }
            
            base_total_success += stats["base"]["success"]
            sft_total_success += stats["sft"]["success"]
            total_problems += stats["base"]["total"]
        
        # Overall summary
        base_overall_success_rate = base_total_success / total_problems if total_problems > 0 else 0
        sft_overall_success_rate = sft_total_success / total_problems if total_problems > 0 else 0
        
        self.results["summary"] = {
            "base": {
                "overall_success_rate": base_overall_success_rate,
                "total_successful": base_total_success,
                "total_problems": total_problems
            },
            "sft": {
                "overall_success_rate": sft_overall_success_rate,
                "total_successful": sft_total_success,
                "total_problems": total_problems
            },
            "improvement": sft_overall_success_rate - base_overall_success_rate,
            "grid_sizes_tested": list(size_results.keys())
        }
        
        # Log final results
        logger.info(f"\n=== Trajectory Completion Evaluation Complete ===")
        logger.info(f"Base Model Success Rate: {base_overall_success_rate:.2%} ({base_total_success}/{total_problems})")
        logger.info(f"SFT Model Success Rate: {sft_overall_success_rate:.2%} ({sft_total_success}/{total_problems})")
        logger.info(f"Improvement: {self.results['summary']['improvement']:.2%}")
        
        for size in sorted(size_results.keys()):
            base_stats = self.results["by_grid_size"][size]["base"]
            sft_stats = self.results["by_grid_size"][size]["sft"]
            logger.info(f"Grid {size}x{size}: Base {base_stats['success_rate']:.2%} vs SFT {sft_stats['success_rate']:.2%}")
        
        # Log final results to W&B
        if self.use_wandb:
            # Log overall metrics
            wandb.log({
                "final/base_overall_success_rate": base_overall_success_rate,
                "final/sft_overall_success_rate": sft_overall_success_rate,
                "final/overall_improvement": self.results['summary']['improvement'],
                "final/total_problems": total_problems
            })
            
            # Log per-grid-size metrics
            for size in sorted(size_results.keys()):
                base_stats = self.results["by_grid_size"][size]["base"]
                sft_stats = self.results["by_grid_size"][size]["sft"]
                wandb.log({
                    f"final/base_success_rate_{size}x{size}": base_stats['success_rate'],
                    f"final/sft_success_rate_{size}x{size}": sft_stats['success_rate'],
                    f"final/improvement_{size}x{size}": sft_stats['success_rate'] - base_stats['success_rate'],
                    f"final/problems_tested_{size}x{size}": base_stats['total']
                })
        
        return self.results
    
    def plot_results(self, save_path: Optional[str] = None) -> str:
        """Create a publication-quality plot comparing model performance by grid size."""
        
        # Set up the plot style
        plt.style.use('default')  # Use default instead of seaborn for better control
        fig, ax = plt.subplots(figsize=(10, 6))  # More reasonable aspect ratio
        
        # Extract data for plotting
        grid_sizes = sorted(self.results["by_grid_size"].keys())
        base_success_rates = []
        sft_success_rates = []
        base_errors = []
        sft_errors = []
        
        for size in grid_sizes:
            base_stats = self.results["by_grid_size"][size]["base"]
            sft_stats = self.results["by_grid_size"][size]["sft"]
            
            base_rate = base_stats["success_rate"] * 100  # Convert to percentage
            sft_rate = sft_stats["success_rate"] * 100
            
            # Calculate 95% confidence intervals (Wilson score interval)
            base_n = base_stats["total"]
            sft_n = sft_stats["total"]
            
            base_ci = self._calculate_confidence_interval(base_stats["successful"], base_n)
            sft_ci = self._calculate_confidence_interval(sft_stats["successful"], sft_n)
            
            base_success_rates.append(base_rate)
            sft_success_rates.append(sft_rate)
            base_errors.append(base_ci)
            sft_errors.append(sft_ci)
        
        # Create the plot with better spacing
        x_pos = np.arange(len(grid_sizes))  # Use arange for better spacing
        width = 0.35
        
        # Plot bars with error bars
        base_bars = ax.bar(x_pos - width/2, base_success_rates, width, 
                          yerr=base_errors, capsize=4, 
                          label='Base Llama 8B', color='#2E8B57', alpha=0.8,
                          edgecolor='black', linewidth=0.5)
        sft_bars = ax.bar(x_pos + width/2, sft_success_rates, width,
                         yerr=sft_errors, capsize=4,
                         label='SFT Llama 8B', color='#FF6B35', alpha=0.8,
                         edgecolor='black', linewidth=0.5)
        
        # Customize the plot
        ax.set_xlabel('Grid Size', fontsize=12, fontweight='bold')
        ax.set_ylabel('Success Rate (%)', fontsize=12, fontweight='bold')
        ax.set_title('FrozenLake Trajectory Completion:\nBase vs SFT Model Performance', 
                    fontsize=14, fontweight='bold', pad=20)
        
        # Set x-axis
        ax.set_xticks(x_pos)
        ax.set_xticklabels([f'{size}×{size}' for size in grid_sizes], fontsize=10)
        
        # Set y-axis
        max_rate = max(max(base_success_rates), max(sft_success_rates))
        ax.set_ylim(0, max(max_rate * 1.2, 20))  # At least 20% for readability
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0f}%'))
        
        # Add grid for better readability
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_axisbelow(True)
        
        # Legend
        ax.legend(fontsize=11, loc='upper right', framealpha=0.9)
        
        # Add value labels on bars (only if they're visible)
        for i, (base_bar, sft_bar) in enumerate(zip(base_bars, sft_bars)):
            height_base = base_bar.get_height()
            height_sft = sft_bar.get_height()
            
            # Only add labels if the bar is visible (height > 1%)
            if height_base > 1:
                ax.text(base_bar.get_x() + base_bar.get_width()/2., 
                       height_base + base_errors[i] + 0.5,
                       f'{height_base:.1f}%', ha='center', va='bottom', 
                       fontsize=9, fontweight='bold')
            
            if height_sft > 1:
                ax.text(sft_bar.get_x() + sft_bar.get_width()/2., 
                       height_sft + sft_errors[i] + 0.5,
                       f'{height_sft:.1f}%', ha='center', va='bottom', 
                       fontsize=9, fontweight='bold')
        
        # Add summary statistics as text box
        base_overall = self.results["summary"]["base"]["overall_success_rate"] * 100
        sft_overall = self.results["summary"]["sft"]["overall_success_rate"] * 100
        improvement = self.results["summary"]["improvement"] * 100
        
        summary_text = f'Overall Results:\nBase: {base_overall:.1f}%\nSFT: {sft_overall:.1f}%\nΔ: {improvement:+.1f}%'
        
        # Position the text box better
        ax.text(0.02, 0.98, summary_text, transform=ax.transAxes, fontsize=10,
               verticalalignment='top', ha='left',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
        
        # Adjust layout to prevent clipping
        plt.tight_layout()
        
        # Save the plot
        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = f"trajectory_comparison_{timestamp}.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()  # Close the figure to free memory
        logger.info(f"Plot saved to {save_path}")
        
        # Log plot to W&B
        if self.use_wandb:
            wandb.log({"trajectory_comparison_plot": wandb.Image(save_path)})
        
        return save_path
    
    def _calculate_confidence_interval(self, successes: int, total: int, confidence: float = 0.95) -> float:
        """Calculate Wilson score confidence interval for success rate."""
        if total == 0:
            return 0
        
        try:
            from scipy import stats
            z = stats.norm.ppf((1 + confidence) / 2)
        except ImportError:
            # Fallback to normal approximation if scipy not available
            z = 1.96  # 95% confidence
        
        p = successes / total
        
        denominator = 1 + z**2 / total
        centre_adjusted_probability = (p + z**2 / (2 * total)) / denominator
        adjusted_standard_deviation = np.sqrt((p * (1 - p) + z**2 / (4 * total)) / total) / denominator
        
        lower_bound = centre_adjusted_probability - z * adjusted_standard_deviation
        upper_bound = centre_adjusted_probability + z * adjusted_standard_deviation
        
        # Return the error bar size (half the CI width) as percentage
        return (upper_bound - lower_bound) * 50  # Convert to percentage and take half
    
    def save_results(self, base_filepath: str) -> Tuple[str, str]:
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