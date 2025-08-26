import gymnasium as gym
import json
import numpy as np
import re
from typing import Any, Dict, List

from scripts.simulator import Simulator

def make_prompt(str_representation):
    return f"""
Grid:
{str_representation}

What action should you take next? Decide to simulate or commit actions. 
If simulating, output <simulate> [your answers] </simulate>
If committing, output <answer> [your answer] </answer>
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
        self.action_id_to_name = {
            0: "Left",
            1: "Down",
            2: "Right",
            3: "Up"
        }
        self.n_sim = 0

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
    
    def _parse_sim_actions(self, response: str) -> list[int]:
        """Return list of actions to simulate"""
        match = re.search(r"<simulate>\s*(.*?)\s*</simulate>", response, re.IGNORECASE)
        actions = []
        if match:
            actions = match.group(1).split()
            actions = [self.action_name_to_id.get(act.upper()) for act in actions]
        return actions
    
    def make_prompt_sim(self, end_str, init_str, actions, actions_simulated, reward):
        actions_sim_str = [self.action_id_to_name.get(action_id) for action_id in actions_simulated]
        actions_str = [self.action_id_to_name.get(action_id) for action_id in actions]
        if actions == actions_simulated:
            return f"""
        X reached hole (H) | * reached goal (G)
        After simulating {" ".join(actions_str)}, a reward of {reward} was obtained and the grid becomes:
        {end_str}
        The true state of the grid is still:
        {init_str}
        What action should you take next? Decide to simulate or commit actions. 
        If simulating, output <simulate> [your answers] </simulate>
        If committing, output <answer> [your answer] </answer>
        """
        else:
            return f"""
        After simulating {" ".join(actions_sim_str)}, a reward of {reward} was obtained and the grid becomes:
        {end_str}
        
        The true state of the grid is still:
        {init_str}
        What action should you take next? Decide to simulate or commit actions. 
        If simulating, output <simulate> [your answers] </simulate>
        If committing, output <answer> [your answer] </answer>
        """

    async def step(self, observation, response, label, **kwargs) -> Dict[str, Any]:
        tool_use = self._parse_tool(response)
        
        if tool_use:
            self.n_sim+=1
            actions = self._parse_sim_actions(response)
            simulator = Simulator(env=self.env_str, env_type="frozenlake", actions=actions, strict=False, return_intermed_states=False)
            end_state, initial_state = simulator.simulate() 
            end_str, reward, actions_simulated = end_state
            
            next_prompt = self.make_prompt_sim(end_str, initial_state, actions, actions_simulated, reward) 
            
            if self.n_sim == 1000:
                return {
                "rewards": np.array([0.]),
                "next_observation": observation + response + "Number of allotted simulations reached. Episode terminated.",
                "done": False,
                "scores": np.array([0.]),
                "extra_logs": self.n_sim
            }
            
            return {
                "rewards": np.array([0.]),
                "next_observation": observation + response + next_prompt,
                "done": False,
                "scores": np.array([0.]),
                "extra_logs": self.n_sim
            }
            
        else: 
            env_action = self._parse_action(response)
            
            if env_action is None:
                return {
                    "rewards": np.array([0.]),
                    "next_observation": observation + response + "Invalid action. Episode terminated.",
                    "done": True,
                    "scores": np.array([0.]),
                    "extra_logs": self.n_sim
                }
            
            obs, reward, terminated, truncated, info = self.env.step(env_action)
            done = terminated or truncated
            if done:
                return {
                    "rewards": np.array([reward]),
                    "scores": np.array([reward]),
                    "next_observation": observation+response,
                    "done": done, 
                    "extra_logs": self.n_sim
                }
            
            env_flat_list = self.env_str.split()
            env_list = env_to_list(self.env)
            init_obs = env_flat_list.index('S')
            size = len(env_list)
            init_x, init_y = init_obs // size, init_obs % size
            env_list[init_x][init_y] = 'F'
            x,y = obs//size, obs%size 
            env_list[x][y] = 'S'
            env_str = '\n'.join(' '.join(row) for row in env_list)
            next_prompt = make_prompt(env_str) 
            
            return {
                "rewards": np.array([reward]),
                "scores": np.array([reward]),
                "next_observation": observation+response+next_prompt,
                "done": done, 
                "extra_logs": self.n_sim
            }


def _extract_grid_from_prompt(prompt: str) -> List[List[str]]:
    """Extract grid from prompt text"""
    try:
        # Find grid section
        if "the grid becomes:" in prompt:
            grid_section = prompt.split("the grid becomes:")[1]
        
            if "The true state" in prompt:
                grid_section = grid_section.split("The true state")[0]

        elif "Grid:" in prompt:
            grid_section = prompt.split("Grid:")[1]
            if "What action" in grid_section:
                grid_section = grid_section.split("What action")[0]
        else:
            raise ValueError("No Grid: section found in prompt")


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
        row = [char.decode('utf-8') for char in row_bytes]
        grid.append(row)
    return grid 

_agent_instance = FrozenLakeAgentInstanceTool()

async def step(observation, action, label, **kwargs):
    if _agent_instance.env is None:
        grid_list = _extract_grid_from_prompt(observation)
        _agent_instance.env_str = '\n'.join([' '.join(row) for row in grid_list])
        _agent_instance.env = _create_gym_env_from_grid(grid_list)
        _agent_instance.env.reset()
        _agent_instance.n_sim = 0
    return await _agent_instance.step(observation, action, label, **kwargs)