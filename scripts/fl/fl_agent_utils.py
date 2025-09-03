import re
from typing import List, Sequence
from scripts.fl_evaluator import FrozenLakeEvaluator
import gymnasium as gym 

action_name_to_id =  {
        "LEFT": 0,
        "DOWN": 1,
        "RIGHT": 2,
        "UP": 3
    }

action_id_to_name = {
        0: "Left",
        1: "Down",
        2: "Right",
        3: "Up"
    }

def assert_valid(grid):
    """assert there is 1 player tile and 1 goal tile in grid"""
    if isinstance(grid, str):
        flat = grid.split()
        
    elif isinstance(grid, list):
        flat = [tile for row in grid for tile in row]
    else:
        raise TypeError(f"got {type(grid).__name__}. need str or list of lists")
        
    player_count = sum(t in ('S','X','*') for t in flat)
    goal_count   = sum(t in ('G','*') for t in flat)
    assert player_count == 1, f"invalid player markers: {grid}"
    assert goal_count   == 1, f"invalid goal markers: {grid}"
    
def make_prompt(str_representation):
    return f"""
Grid:
{str_representation}

What action should you take next? Decide to simulate or commit actions. 
If simulating, output <simulate> [your answers] </simulate>
If committing, output <answer> [your answer] </answer>
"""

def create_gym_env_from_grid(grid: List[List[str]]):
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

def extract_grid_from_prompt(
    prompt: str,
    valid_tokens: Sequence[str] = ("S", "F", "H", "G"),
) -> List[List[str]]:
    """Extract latest grid and return list of lists"""
    p_low = prompt.lower()
    # Prefer the latest of these markers, case-insensitive
    markers = ("the true state of the grid is still:", "grid is still:", "grid:")
    last_pos = -1
    last_marker = None
    for mk in markers:
        pos = p_low.rfind(mk)
        if pos > last_pos:
            last_pos, last_marker = pos, mk
    if last_marker is None:
        raise ValueError("No grid marker found in prompt.")

    # Slice from the last marker to the next "What action" (if present)
    section = prompt[last_pos + len(last_marker):]
    wa_idx = section.lower().find("what action")
    if wa_idx != -1:
        section = section[:wa_idx]

    # Stream lines, keep only contiguous rows entirely made of valid tokens
    grid: List[List[str]] = []
    started = False
    valid = set(valid_tokens)
    for raw in section.splitlines():
        line = raw.strip()
        if not line:
            if started:
                break
            continue
        toks = [t.upper() for t in line.split()]
        if toks and all(t in valid for t in toks):
            grid.append(toks)
            started = True
        else:
            if started:
                break  # stop at first non-grid line after starting

    if not grid:
        raise ValueError("No grid rows found after marker.")
    
    return grid

def parse_action(self, response: str):
    """Return single committed action"""
    match = list(re.finditer(r"<answer>\s*(\w+)\s*</answer>", response, re.IGNORECASE))
    if match:
        match = match[-1]
        action_name = match.group(1).strip().upper()
        return self.action_name_to_id.get(action_name)
    return None


def parse_tool(response: str) -> bool:
    """Return True if using simulator else false"""
    # if "<simulate>" in response:
    #     return True
    last_word = response.split()[-1] if response and response.split() else ""
    if last_word.endswith("</simulate>"):
        return True
    return False

def parse_full_trajectory(self, response: str) -> tuple:
        """Take full reasoning trajectory and return Tuple(end_state: Gym env, reward: bool)"""
        multistep_eval = FrozenLakeEvaluator()
        traj = multistep_eval._extract_reasoning(response)
        actions = multistep_eval._extract_action_sequence(traj)
        end = multistep_eval._simulate_path(env_to_list(self.env), actions)
        reward = end["reward"]
        end_state = end["state"]
        return end_state, reward 

def parse_sim_actions(response: str) -> list[int]:
    """Return action names from the last <simulate>...</simulate> block (no ID mapping)."""
    matches = list(re.finditer(
        r"<simulate>\s*(.*?)\s*</simulate>",
        response,
        re.IGNORECASE | re.DOTALL,
    ))
    if not matches:
        return []
    content = matches[-1].group(1)
    actions = [action_name_to_id.get(tok.upper()) for tok in content.split()]
    # Filter out unknowns
    return [a for a in actions if a is not None]

def make_prompt_sim(end_str, init_str, actions, actions_simulated, reward):
    actions_sim_str = [action_id_to_name.get(action_id) for action_id in actions_simulated]
    actions_str = [action_id_to_name.get(action_id) for action_id in actions]
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
    
def str_to_grid_list(string):
    return [row.split() for row in string.split('\n')]

def grid_list_to_str(grid_list):
    return '\n'.join(' '.join(row) for row in grid_list)