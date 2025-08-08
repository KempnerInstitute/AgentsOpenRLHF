import gymnasium as gym
import numpy as np
import re
from typing import Any, Dict, List

from scripts.fl_datagen import make_prompt
from openrlhf.utils.agent import AgentInstanceBase
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
from scripts.fl_evaluator import FrozenLakeEvaluator


class FrozenLakeAgentInstance(AgentInstanceBase):
    """Execute one step of verification and return a random reward using torch.rand

    Args:
        map_sizes: list of n for nxn map sizes
        env: gym object 

    Returns:
        Dict[str, Any]: A dictionary containing:
            - rewards: np.array reward value for advantage calculation
            - scores: np.array reward value for dynamic filtering
            - next_observation: prompt for the next observation
            - done: bool indicating if the episode is complete
            - sampling_params: Parameters for vLLM sampling
            - extra_logs: Additional logging information    # this was giving me bugs wrt how it was being batched by experience_maker so i disabled for now
    """
    def __init__(self, interactive=False, *args, **kwargs):
        self.map_sizes = [4,5,6,7,8]
        self.env = None
        self.interactive = interactive
        self.action_name_to_id =  {
            "LEFT": 0,
            "DOWN": 1,
            "RIGHT": 2,
            "UP": 3
        }
        

    def _parse_action(self, response: str):
        match = re.search(r"<answer>\s*(\w+)\s*</answer>", response, re.IGNORECASE)
        if match:
            action_name = match.group(1).strip().upper()
            return self.action_name_to_id.get(action_name)
        return None

    def _parse_full_trajectory(self, response: str) -> tuple:
        """Take full reasoning trajectory and return Tuple(end_state: Gym env, reward: bool)"""
        multistep_eval = FrozenLakeEvaluator()
        traj = multistep_eval._extract_reasoning(response)
        actions = multistep_eval._extract_action_sequence(traj)
        end = multistep_eval._simulate_path(env_to_list(self.env), actions)
        reward = end["reward"]
        end_state = end["state"]
        return end_state, reward 

    async def step(self, observation, action, label, **kwargs) -> Dict[str, Any]:
        # Get action from LLM response
        response = action
        if self.interactive:
            env_action = self._parse_action(response)

            if env_action is None:      # penalize invalid action
                return {
                    "rewards": np.array([0.]),
                    "next_observation": "Invalid action. Episode terminated.",
                    "done": True,
                    "scores": np.array([0.]),
                    "extra_logs": {"error": "Invalid action"}
                }
            
            # Perform action in Gymnasium env
            obs, reward, terminated, truncated, info = self.env.step(env_action)
            done = terminated or truncated

            # Format game state into prompt
            env_str = env_to_str(self.env)
            environment_feedback = make_prompt(env_str)
            
            return {
                "rewards": np.array([reward]),
                "scores": np.array([reward]),
                "next_observation": environment_feedback,
                "done": done, 
                "extra_logs": info
            }

        else:
            print("parsing full trajectory from model response (evaluating non-interactively)")
            end_state, goal = self._parse_full_trajectory(response)
            end_str = env_to_str(end_state)
            env_feedback = make_prompt(end_str)
            
            if goal is True:
                reward = 1.0
            else:
                reward = 0.0
                print(f"\nenv feedback for next observation \n {env_feedback}\n (over)")
            return {
                "rewards": np.array([reward]),
                "scores": np.array([reward]),
                "next_observation": observation + action + env_feedback,
                "done": True, 
                "extra_logs": None
            }


def _extract_grid_from_prompt(prompt: str) -> List[List[str]]:
    """Extract grid from prompt text"""
    try:
        # Find the grid section
        if "Grid:" in prompt:
            grid_section = prompt.split("Grid:")[1]
        else:
            raise ValueError("No Grid: section found in prompt")

        if "What action" in grid_section:
            grid_section = grid_section.split("What action")[0]

        grid_section = grid_section.strip()
        lines = [line.strip() for line in grid_section.split("\n") if line.strip()]

        # Parse map
        grid = []
        for line in lines:
            if " " in line:  # Grid rows should have spaces between cells
                row = line.split()
                if row and all(
                    cell in ["S", "F", "H", "G", "@"] for cell in row
                ):  # Valid symbols check
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

def _create_gym_env_from_grid(grid: List[List[str]]):
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
        "FrozenLake-v1", desc=desc, is_slippery=False
    )

def env_to_str(env):
    grid_bytes = env.unwrapped.desc
    grid = []
    for row_bytes in grid_bytes:
        row_str = " ".join([char.decode('utf-8') for char in row_bytes])
        grid.append(row_str)
    grid_str = "\n".join([row for row in grid])
    return grid_str

def env_to_list(env):
    grid_bytes = env.unwrapped.desc
    grid = []
    for row_bytes in grid_bytes:
        # row_str = " ".join([char.decode('utf-8') for char in row_bytes])
        row = [char.decode('utf-8') for char in row_bytes]
        grid.append(row)
    return grid 

_agent_instance = FrozenLakeAgentInstance()

async def step(observation, action, label, **kwargs):
    # If new map make new env 
    if _agent_instance.env is None:
        grid_str = _extract_grid_from_prompt(observation)
        _agent_instance.env = _create_gym_env_from_grid(grid_str)
        init_state = _agent_instance.env.reset()

    return await _agent_instance.step(observation, action, label, **kwargs)