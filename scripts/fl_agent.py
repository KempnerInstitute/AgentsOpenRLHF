import gymnasium as gym
import numpy as np
from typing import Any, Dict

from openrlhf.utils.agent import AgentInstanceBase
from scripts.fl_evaluator import FrozenLakeEvaluator
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
            - extra_logs: Additional logging information    # this was giving me bugs wrt how it was being batched by experience_maker so i disabled for now
    """
    def __init__(self, *args, **kwargs):
        self.env = None
        
    def parse_full_trajectory(self, response: str) -> tuple:
        """Take full reasoning trajectory and return Tuple(end_state: Gym env, reward: bool)"""
        multistep_eval = FrozenLakeEvaluator()
        traj = multistep_eval._extract_reasoning(response)
        actions = multistep_eval._extract_action_sequence(traj)
        end = multistep_eval._simulate_path(fl.env_to_list(self.env), actions)
        reward = end["reward"]
        end_state = end["state"]
        return end_state, reward 

    async def step(self, observation, action, label, **kwargs) -> Dict[str, Any]:
        # Get action from LLM response
        response = action

        end_state, goal = self.parse_full_trajectory(response)
        
        if goal is True:
            reward = 1.0
            
        else:
            reward = 0.0
            
        return {
            "rewards": np.array([reward]),
            "scores": np.array([reward]),
            "next_observation": observation + action,
            "done": True, 
            "extra_logs": None
        }

_agent_instance = FrozenLakeAgentInstance()

async def step(observation, action, label, **kwargs):
    # If new map make new env 
    if _agent_instance.env is None:
        grid_str = fl.extract_grid_from_prompt(observation)
        _agent_instance.env = fl.create_gym_env_from_grid(grid_str)
        _agent_instance.env.reset()

    return await _agent_instance.step(observation, action, label, **kwargs)