import gymnasium as gym 
import scripts.fl as fl 
    
class FrozenLakeSimulator:
    def __init__(self, init_str, actions, strict):
        self.init_str = init_str    # initial state stored as str 
        self.actions = actions
        self.strict = strict 
        
        self.size = len(fl.str_to_grid_list(init_str))
        
        # store poisitions as tuple (x,y)
        self.init_pos = self.get_s_pos(init_str)
        self.curr_pos = self.get_s_pos(init_str)
        
        # gym action id to (x,y) delta
        self.action_delta = {
            0: (0,-1),  # left
            1: (1,0),   # down
            2: (0,1),   # right
            3: (-1,0)   # up
        }
        
        # gym env object
        self.env = None
        
    def assert_valid(self, grid):
        """assert there is 1 player tile and 1 goal tile in grid"""
        if isinstance(grid, str):
            flat = grid.split()
            
        elif isinstance(grid, list):
            flat = [tile for row in grid for tile in row]
        else:
            raise TypeError(f"got {type(grid).__name__}. need str or list of lists")
            
        player_count = sum(t in ('S','X','*') for t in flat)
        goal_count   = sum(t in ('G','*') for t in flat)
        assert player_count == 1, f"invalid player markers: {grid} \n initial str: {self.init_str} \n actions: {self.actions}"
        assert goal_count   == 1, f"invalid goal markers: {grid} \n initial str: {self.init_str} \n actions: {self.actions}"
        
    def assert_valid_for_env(self, grid):
        if isinstance(grid, str):
            flat = grid.split()
            
        elif isinstance(grid, list):
            flat = [tile for row in grid for tile in row]
        else:
            raise TypeError(f"got {type(grid).__name__}. need str or list of lists")
            
        player_count = sum(t == 'S' for t in flat)
        goal_count   = sum(t == 'G' for t in flat)
        assert player_count == 1, f"invalid player markers: {grid} \n initial str: {self.init_str} \n actions: {self.actions}"
        assert goal_count   == 1, f"invalid goal markers: {grid} \n initial str: {self.init_str} \n actions: {self.actions}"
        
    def obs_to_coords(self, obs):
        return divmod(obs, self.size)
    
    def get_s_pos(self, grid):
        if isinstance(grid, str):
            curr_grid_flat = grid.split()
        elif isinstance(grid, list):
            curr_grid_flat = [tile for row in grid for tile in row]
        init_obs = curr_grid_flat.index('S')
        return self.obs_to_coords(init_obs)
    
    def _create_env_from_str(self, state: str):
        self.assert_valid_for_env(state)
        self.init_pos = self.get_s_pos(state)
        desc = [list(line.replace(" ","")) for line in state.split("\n")]
        self.env = gym.make("FrozenLake-v1", desc=desc, is_slippery=False)
        self.env.reset()
    
    def render_end_pos(self):
        grid = fl.str_to_grid_list(self.init_str)
        
        # replace old S from self.init pos as F 
        rS,cS = self.init_pos
        rE,cE = self.curr_pos

        # find goal in immutable terrain
        flat = self.init_str.split()
        g_idx = flat.index('G')
        n = self.size
        rG, cG = divmod(g_idx, n)

        if (rE,cE) != (rS,cS) and grid[rS][cS] == 'S':
            grid[rS][cS] = 'F'

        tile = grid[rE][cE]
        if tile == 'G':
            grid[rE][cE] = '*'
        elif tile == 'H':
            grid[rE][cE] = 'X'; grid[rG][cG] = 'G'
        else:
            grid[rE][cE] = 'S'; grid[rG][cG] = 'G'

        end_str = fl.grid_list_to_str(grid)
        self.assert_valid(end_str)
        return end_str
    
    def clamp(self, pos):
        return max(0, min(pos, self.size - 1))
    
    def manual_step_from_goal(self, action):
        """return obs, reward, done"""
        # assert current position is on goal 
        x0, y0 = self.curr_pos
        assert self.curr_state[x0][y0] == 'G'
        
        # get movement
        dx, dy = self.action_delta.get(action)
        
        # step and clamp to make sure in bounds
        x,y = self.clamp(x0 + dx), self.clamp(y0 + dy)
        
        # check if terminal
        end_tile = self.curr_state[x][y]
        next_obs = self.size*x + y 
        
        if end_tile == 'H':
            return next_obs, 0, True
        elif end_tile == 'G':
            return next_obs, 1, True
        
        return next_obs, 0, False
        
    def simulate(self):
        """return (end state: str, reward: int, actions_simulated: list), initial_state_str"""
        # end simulation if no actions to simulate
        if self.actions == []:
            return (self.init_str, 0, []), self.init_str 
        
        self._create_env_from_str(self.init_str)        # includes env.reset()
        
        act_i = 0
        while act_i < len(self.actions):
            # handle invalid action
            if self.actions[act_i] is None:
                return (self.render_end_pos(), 0, self.actions[:act_i]), self.init_str
            
            # step
            obs, reward, terminated, truncated, info = self.env.step(self.actions[act_i])
            done = terminated or truncated 
            
            # update current state 
            self.curr_pos = self.obs_to_coords(obs)
            
            # handle terminal states
            if done:
                if self.strict == True and reward > 0 and act_i < len(self.actions)-1:
                    # while loop to handle edge case where G is on edge and model keeps returning to it 
                    while reward > 0 and act_i < len(self.actions) - 1:
                        # manually step as if goal was never reached 
                        act_i+=1
                    
                        # perform next action 
                        next_obs, reward, done = self.manual_step_from_goal(self.actions[act_i])
                        
                        # update current position
                        self.curr_pos = self.obs_to_coords(next_obs)
                        
                        # check if terminal state 
                        if done and reward == 0:
                            return (self.render_end_pos(), reward, self.actions[:act_i+1]), self.init_str
                        
                    # case where ran out of actions and on goal 
                    if reward > 0 and act_i == len(self.actions)-1:
                        return (self.render_end_pos(), 1, self.actions[:act_i+1]), self.init_str
                    
                    # case where stepped off goal and done is False
                    x, y = self.curr_pos 
                    
                    # re-set S in env 
                    self.env.unwrapped.s = x*self.size + y
                else:
                    # end simulation
                    return (self.render_end_pos(), reward, self.actions[:act_i+1]), self.init_str
                
            act_i+=1
        
        return (self.render_end_pos(), 0, self.actions), self.init_str
  