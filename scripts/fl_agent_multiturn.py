import gymnasium as gym
import numpy as np
from typing import Any, Dict, List

from openrlhf.utils.agent import AgentInstanceBase

import scripts.fl as fl

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
            - extra_logs: Additional logging information  
    """
    def __init__(self, *args, **kwargs):
        self.env = None

    async def step(self, observation, action, label, **kwargs) -> Dict[str, Any]:
        response = action

        env_action = fl.parse_action(response)

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
        
        if done:
            return {
            "rewards": np.array([reward]),
            "scores": np.array([reward]),
            "next_observation": observation + action,
            "done": done
            }

        # Format game state into prompt
        env_str = fl.env_to_str(self.env)
        grid_list = fl.str_to_grid_list(env_str)
        x0, y0, size = get_curr_pos(env_str)
        assert grid_list[x0][y0] == 'S'
        x,y = divmod(obs, size)

        # if moved and not done:
        if grid_list[x][y] == 'F':
            grid_list[x0][y0] = 'F'
            grid_list[x][y] = 'S'
        
        # make new prompt
        next_prompt = fl.make_prompt_no_tool(fl.grid_list_to_str(grid_list))
        
        return {
            "rewards": np.array([reward]),
            "scores": np.array([reward]),
            "next_observation": observation + action + next_prompt,
            "done": done, 
            "extra_logs": info
        }

def get_curr_pos(string):
    curr_grid_flat = string.split()
    init_obs = curr_grid_flat.index('S')
    size = int(len(curr_grid_flat)**0.5)
    x = init_obs // size
    y = init_obs % size 
    return x, y, size     

_agent_instance = FrozenLakeAgentInstance()

async def step(observation, action, label, **kwargs):
    # If new map make new env 
    if _agent_instance.env is None:
        grid_str = fl.extract_grid_from_prompt(observation)
        _agent_instance.env = fl.create_gym_env_from_grid(grid_str)
        _agent_instance.env.reset()

    return await _agent_instance.step(observation, action, label, **kwargs)