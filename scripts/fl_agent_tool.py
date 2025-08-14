import gymnasium as gym
import json
import numpy as np
import re
from typing import Any, Dict, List

from scripts.fl_datagen import make_prompt
from openrlhf.utils.agent import AgentInstanceBase
from scripts.fl_evaluator import FrozenLakeEvaluator
from scripts.simulator import Simulator

def make_prompt_sim(end_str, init_str):
    # TODO edit this 
    return f"""
after simulating, the grid becomes:
{end_str}
currently the grid is still 
{init_str}
Commit or simulate?
"""

class FrozenLakeAgentInstanceTool():
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
        self.env_str = None
        self.action_name_to_id =  {
            "LEFT": 0,
            "DOWN": 1,
            "RIGHT": 2,
            "UP": 3
        }

    def _parse_action(self, response: str):
        """Return single committed action"""
        match = re.search(r"<answer>\s*(\w+)\s*</answer>", response, re.IGNORECASE)
        if match:
            action_name = match.group(1).strip().upper()
            return self.action_name_to_id.get(action_name)
        return None
    
    def _parse_tool(self, response: str) -> bool:
        """Return True if using simulator else false"""
        if "<simulate>" in response:
            return True
        return False

    def _parse_full_trajectory(self, response: str) -> tuple:
        """Take full reasoning trajectory and return Tuple(end_state: Gym env, reward: bool)"""
        multistep_eval = FrozenLakeEvaluator()
        traj = multistep_eval._extract_reasoning(response)
        actions = multistep_eval._extract_action_sequence(traj)
        end = multistep_eval._simulate_path(env_to_list(self.env), actions)
        reward = end["reward"]
        end_state = end["state"]
        return end_state, reward 
    
    def _parse_sim_actions(self, response: str) -> list[str]:
        """Return list of actions to simulate"""
        match = re.search(r"<simulate>\s*(\w+)\s*</simulate>", response, re.IGNORECASE)
        actions = []
        if match:
            action_name = match.group(1).strip().upper()
            actions.append(self.action_name_to_id.get(action_name))
        return actions

    async def step(self, observation, response, label, **kwargs) -> Dict[str, Any]:
        tool_use = self._parse_tool(response)
        
        if tool_use:
            actions = self._parse_sim_actions(response)
            simulator = Simulator(self.env_str, "frozenlake", actions, return_intermed_states=False)
            end_state, initial_state = simulator.simulate()
            end_str, reward, info = end_state[-1]
            next_prompt = make_prompt_sim(end_str, initial_state) # TODO: change this to be end string of simulation and initial env string to "return" to actual state 
            return {
                "rewards": np.array([reward]),
                "next_observation": observation + response + next_prompt,
                "done": False,
                "scores": np.array([reward])
            }
            
        else: 
            env_action = self._parse_action(response)
            
            if env_action is None:
                return {
                    "rewards": np.array([0.]),
                    "next_observation": observation + response + "Invalid action. Episode terminated.",
                    "done": True,
                    "scores": np.array([0.]),
                }
            
            obs, reward, terminated, truncated, info = self.env.step(env_action)
            done = terminated or truncated
            
            env_str = env_to_str(self.env)
            next_prompt = make_prompt(env_str) # TODO: change make_prompt function to the simulate or commit thing and figure out how to formulate 
            return {
                "rewards": np.array([reward]),
                "scores": np.array([reward]),
                "next_observation": observation+response+next_prompt,
                "done": done, 
                "extra_logs": info
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

_agent_instance = FrozenLakeAgentInstanceTool()

async def step(observation, action, label, **kwargs):
    if _agent_instance.env is None:
        grid_str = _extract_grid_from_prompt(observation)
        _agent_instance.env_str = grid_str
        _agent_instance.env = _create_gym_env_from_grid(grid_str)
        _agent_instance.env.reset()
    return await _agent_instance.step(observation, action, label, **kwargs)