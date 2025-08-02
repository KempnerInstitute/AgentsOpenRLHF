import gymnasium as gym
import torch
import random
import re
from typing import Any, Dict, List

from scripts.fl_datagen import make_prompt
from openrlhf.utils.agent import AgentInstanceBase
from gymnasium.envs.toy_text.frozen_lake import generate_random_map


class FrozenLakeAgentInstance(AgentInstanceBase):
    """
    Handles state of  episode (reset, step) and parse LLM output
    """
    def __init__(self, prompt, *args, **kwargs):
        self.map_sizes = [4,5,6,7,8]
        self.env = gym.make('FrozenLake-v1', desc=generate_random_map(size=map_size), is_slippery=False, render_mode="ansi")

        self.action_name_to_id =  {
            "LEFT": 0,
            "DOWN": 1,
            "RIGHT": 2,
            "UP": 3
        }
        
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

    def _parse_action(self, response: str):
        match = re.search(r"<answer>\s*(\w+)\s*</answer>", response, re.IGNORECASE)
        if match:
            action_name = match.group(1).strip().upper()
            return self.action_name_to_id.get(action_name)
        return None
    
    async def reset(self, states: dict, **kwargs) -> Dict[str, Any]:
        """
        Resets the environment to its initial state and generates the first prompt.
        
        Args:
            states (dict): The initial states from the executor.

        Returns:
            Dict[str, Any]: A dictionary containing the initial observation and a done flag.
        """
        # Randomly select a map size for each episode
        map_size = random.choice(self.map_sizes)
        
        # Create a new environment instance with the chosen size
        if self.env:
            self.env.close()
        self.env = gym.make('FrozenLake-v1', desc=generate_random_map(size=map_size), is_slippery=False, render_mode="ansi")
        
        # Reset the Gymnasium environment
        obs, info = self.env.reset(seed=random.randint(0, 10000))
        
        # Format the initial observation for the LLM
        observation = make_prompt(self.env.render())
        
        # We return the initial observation in the `observation` key
        # states["observation"] = observation
        return {"observation": observation}

    async def step(self, observation, action, label, **kwargs) -> Dict[str, Any]:
        response = action

        env_action = self._parse_action(response)
        if env_action is None:
            return {
                "rewards": torch.tensor(-10.),
                "environment_feedback": "Invalid action. Episode terminated.",
                "done": True,
                "scores": -10.,
                "extra_logs": {"error": "Invalid action"}
            }
        
        obs, reward, terminated, truncated, info = self.env.step(env_action)
        done = terminated or truncated

        env_str = self.env.render()
        environment_feedback = make_prompt(env_str)

        return {
            "rewards": torch.tensor(reward),
            "scores": reward,
            "environment_feedback": environment_feedback,
            "done": done, 
            "extra_logs": info
        }


_agent_instance = FrozenLakeAgentInstance()

async def reset(states: dict, **kwargs):
    return await _agent_instance.reset(states, **kwargs)

async def step(observation, action, label, **kwargs):
    return await _agent_instance.step(observation, action, label, **kwargs)