import gymnasium as gym
import numpy as np
from typing import Any, Dict

from scripts.simulator import FrozenLakeSimulator
import scripts.fl as fl

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
            - extra_logs: Additional logging information  
    """
    def __init__(self, *args, **kwargs):
        self.env = None
        self.env_str = None
        self.n_sim = 0

    async def step(self, observation, response, label, **kwargs) -> Dict[str, Any]:
        tool_use = fl.parse_tool(response)
        
        if tool_use:
            # simulating
            self.n_sim+=1
            actions = fl.parse_sim_actions(response)
            grid_list = fl.extract_grid_from_prompt(observation)
            grid_str = fl.grid_list_to_str(grid_list)
            simulator = FrozenLakeSimulator(init_str=grid_str, actions=actions, strict=False)
            end_state, initial_state = simulator.simulate() 
            fl.assert_valid(end_str)
            end_str, reward, actions_simulated = end_state

            next_prompt = fl.make_prompt_sim(end_str, initial_state, actions, actions_simulated, reward) 
            
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
            # committing
            env_action = fl.parse_action(response)
            
            if env_action is None:
                return {
                    "rewards": np.array([0.]),
                    "next_observation": observation + response + "Invalid action" + observation,
                    "done": False,      # NOTE: allow to continue? 
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
            
            env_list = fl.str_to_grid_list(self.env_str)
            
            size = len(env_list)
            
            # get prev position
            prev = [(i,j) for i,row in enumerate(env_list) for j,t in enumerate(row) if t == 'S']
            if prev:
                px, py = prev[0]
            else:
                px, py = divmod(self.env.unwrapped.s, size)     # fallback ? 
            
            nx,ny = divmod(obs, size)
            
            # clear prev tile if different
            if (px,py) != (nx,ny):
                if env_list[px][py] == 'S':
                    env_list[px][py] = 'F'
            
            env_list[nx][ny] = 'S'
            
            self.env_str = fl.grid_list_to_str(env_list)
            fl.assert_valid(self.env_str)
            next_prompt = fl.make_prompt(self.env_str) 
            return {
                "rewards": np.array([reward]),
                "scores": np.array([reward]),
                "next_observation": observation+response+next_prompt,
                "done": done, 
                "extra_logs": self.n_sim
            }

_agent_instance = FrozenLakeAgentInstanceTool()

async def step(observation, action, label, **kwargs):
    if _agent_instance.env is None:
        grid_list = fl.extract_grid_from_prompt(observation)
        _agent_instance.env_str = fl.grid_list_to_str(grid_list)
        _agent_instance.env = fl.create_gym_env_from_grid(grid_list)
        _agent_instance.env.reset()
        _agent_instance.n_sim = 0
    return await _agent_instance.step(observation, action, label, **kwargs)