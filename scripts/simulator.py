import multiprocessing as mp
import gymnasium as gym 
from concurrent.futures import ProcessPoolExecutor

class Simulator:
    def __init__(self, env: str, env_type: str, actions, return_intermed_states=False):
        # Initialize with env 
        self.env = None
        self.env_str = env  
        self.env_type = env_type   # "sokoban" or "frozenlake"
        self.actions = actions
        self.return_intermed_states = return_intermed_states
    
    def simulate(self):
        # save true env state 
        saved = self.env_str
        
        # exit if no actions to simulate
        if not self.actions:
            return [(self.env, 0, None)], saved
        
        results = []

        # simulate
        if not self.env:
            self.env = self._restore_state(self.env_str)
            
        self.env.reset()
        
        for act in self.actions:
            obs, reward, done, info = self.env.step(act)
            text_state = self._extract_state(self.env)
            results.append((text_state, reward, info)) 
            if done:                                        # TODO: implement if reached goal but unfinished simulation
                break                                       # TODO: what to return if on goal?

        # restore true env state
        self._restore_state(saved)  

        if self.return_intermed_states:
            return results, saved
        else:
            return results[-1], saved
    
    def simulate_parallel(self, action_seqs, num_workers=None):
        num_workers = num_workers or min(len(action_seqs), mp.cpu_count())
        curr_state = self._extract_state()

        tasks = [
            (self.env, curr_state, action_seq, self.return_intermed_states) for action_seq in action_seqs
        ]

        with ProcessPoolExecutor(max_workers=num_workers) as executor: 
            results = list(executor.map(_simulate_worker, tasks))
        
        return results
    
    def commit(self, action):
        return self.env.step(action)  
    
    def _extract_state(self) -> str:
        if "frozenlake" == self.env_type:
            def env_to_str(env):
                grid_bytes = env.unwrapped.desc
                grid = []
                for row_bytes in grid_bytes:
                    row_str = " ".join([char.decode('utf-8') for char in row_bytes])
                    grid.append(row_str)
                grid_str = "\n".join([row for row in grid])
                return grid_str

            return env_to_str(self.env)
        
        elif "sokoban" == self.env_type:
            return self.env.render_text()

    def _restore_state(self, state: str):
        if "frozenlake" == self.env_type:
            desc = [line.replace(" ","") for line in state.strip().split("\n")]
            self.env = gym.make("FrozenLake-v1", desc=desc, is_slippery=False)
            
        elif "sokoban" == self.env_type:
            from scripts.sokoban_evaluator import SokobanEvaluator
            evaluator = SokobanEvaluator()
            self.env = evaluator._create_env_from_map(state)


def _simulate_worker(task):
    env, initial_state, action_seq, return_intermed_states = task
    
    # Initialize environment
    simulator = Simulator(env, return_intermed_states)
    
    # Restore env state
    simulator._restore_state(initial_state)
    
    # Run simulation
    return simulator.simulate(action_seq)