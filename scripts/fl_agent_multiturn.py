import gymnasium as gym
import json
import numpy as np
import re
import time
from typing import Any, Dict, List

from openrlhf.utils.agent import AgentInstanceBase
from scripts.fl_evaluator import FrozenLakeEvaluator

def make_prompt(str_representation):
    return f"""
Grid:
{str_representation}

What action should you take next? Decide the next action:
Always output: <answer> [your answer] </answer> with no extra text.
Strictly follow this format. <|im_end|>
<|im_start|>assistant
<think>
"""

class FrozenLakeAgentInstance(AgentInstanceBase):
    """Execute one step of verification and return a random reward using torch.rand

    Args:
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
    def __init__(self, *args, **kwargs):
        self.env = None
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

    async def step(self, observation, action, label, **kwargs) -> Dict[str, Any]:
        # Get action from LLM response
        response = action

        env_action = self._parse_action(response)

        if env_action is None:
            return {
                "rewards": np.array([0.]),
                "next_observation": observation + action + "Invalid action. Episode terminated.",
                "done": True,
                "scores": np.array([0.])
            }
        
        # Perform action in Gymnasium env
        obs, reward, terminated, truncated, info = self.env.step(env_action)
        done = terminated or truncated

        # Format game state into prompt
        env_str = env_to_str(self.env)
        grid_list = str_to_grid_list(env_str)
        x0, y0, size = get_curr_pos(env_str)
        assert grid_list[x0][y0] == 'S'
        x,y = obs//size, obs % size
        
        if done:
            return {
            "rewards": np.array([reward]),
            "scores": np.array([reward]),
            "next_observation": observation + action,
            "done": done
            }

        # if moved and not done:
        if grid_list[x][y] == 'F':
            grid_list[x0][y0] = 'F'
            grid_list[x][y] = 'S'
        
        # make new prompt
        next_prompt = make_prompt(grid_list_to_str(grid_list))
        
        return {
            "rewards": np.array([reward]),
            "scores": np.array([reward]),
            "next_observation": observation + action + next_prompt,
            "done": done, 
            "extra_logs": info
        }

def grid_list_to_str(grid_list):
    return '\n'.join(' '.join(row) for row in grid_list)

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
                    cell in ["S", "F", "H", "G"] for cell in row
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
            if cell in ["S", "F", "H", "G"]:  # Valid gym symbols
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

def get_curr_pos(string):
    curr_grid_flat = string.split()
    init_obs = curr_grid_flat.index('S')
    size = int(len(curr_grid_flat)**0.5)
    x = init_obs // size
    y = init_obs % size 
    return x, y, size     

def str_to_grid_list(string):
    return [row.split() for row in string.split('\n')] 

_agent_instance = FrozenLakeAgentInstance()

async def step(observation, action, label, **kwargs):
    # If new map make new env 
    if _agent_instance.env is None:
        grid_str = _extract_grid_from_prompt(observation)
        _agent_instance.env = _create_gym_env_from_grid(grid_str)
        _agent_instance.env.reset()

    return await _agent_instance.step(observation, action, label, **kwargs)