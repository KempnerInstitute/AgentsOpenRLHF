import gymnasium as gym
import numpy as np
from typing import Any, Dict, List

import scripts.sokoban.sokoban_utils as skb
from openrlhf.utils.agent import AgentInstanceBase

class SokobanAgentInstance(AgentInstanceBase):
    def __init__(self, *args, **kwargs):
        self.env = None

    async def step(self, observation, action, label, **kwargs) -> Dict[str, Any]:
        if action == "":
            return {
               "rewards": np.array([0.]),
                "next_observation": observation + action + "Invalid action. Episode terminated.",
                "done": True,
                "scores": np.array([0.]) 
            } 
        
        env_action = skb.parse_action(action)
        
        if env_action is None:
            return {
                "rewards": np.array([0.]),
                "next_observation": observation + action + "Invalid action. Episode terminated.",
                "done": True,
                "scores": np.array([0.])
            }
        
        obs, reward, done, info = self.env.step(env_action)

        if done:
            reward = 1 if info["all_boxes_on_target"] else 0
            return {
                "rewards": np.array([reward]),
                "scores": np.array([reward]),
                "next_observation": observation + action,
                "done": done
            }
            
        next_state_str = self.env.render_text()
        next_prompt = skb.make_prompt_no_tool(next_state_str)
        
        return {
            "rewards": np.array([reward]),
            "scores": np.array([reward]),
            "next_observation": observation + action + next_prompt,
            "done": done, 
            "extra_logs": info
        }
    
_agent_instance = SokobanAgentInstance()

async def step(observation, action, label, **kwargs):
    if _agent_instance.env is None:
        # parse state
        grid_str = skb.extract_grid_from_prompt(observation)
        # init gym environment
        _agent_instance.env = skb.create_env_from_map(grid_str, 75)
        _agent_instance.env.reset()
        
    return await _agent_instance.step(observation, action, label, **kwargs)

