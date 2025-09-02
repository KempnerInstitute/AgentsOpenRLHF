# fl_eval_multiturn_minimal.py  (drop-in replacement for your current eval script)

import argparse
import os
import json
import re
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

ACTION_NAME_TO_ID = {"LEFT": 0, "DOWN": 1, "RIGHT": 2, "UP": 3}

def make_prompt(str_representation):
    return f"""
Grid:
{str_representation}

What action should you take next? Decide the next action:
Always output: <answer> [your answer] </answer> with no extra text.
Strictly follow this format. <|im_end|>
<|im_start|>assistant
<think>
"""

def grid_list_to_str(grid_list):
    return '\n'.join(' '.join(row) for row in grid_list)

def str_to_grid_list(string):
    return [row.split() for row in string.split('\n') if row.strip()]

def env_to_str(env):
    grid_bytes = env.unwrapped.desc
    grid = []
    for row_bytes in grid_bytes:
        row_str = " ".join([char.decode('utf-8') for char in row_bytes])
        grid.append(row_str)
    return "\n".join(grid)

def get_curr_pos(string):
    curr_grid_flat = string.split()
    init_obs = curr_grid_flat.index('S')
    size = int(len(curr_grid_flat)**0.5)
    x = init_obs // size
    y = init_obs % size
    return x, y, size

def _extract_grid_from_prompt(prompt: str):
    if "Grid:" not in prompt:
        raise ValueError("No Grid: section found in prompt")
    grid_section = prompt.split("Grid:", 1)[1]
    if "What action" in grid_section:
        grid_section = grid_section.split("What action", 1)[0]
    grid_section = grid_section.strip()
    lines = [line.strip() for line in grid_section.split("\n") if line.strip()]

    grid = []
    for line in lines:
        if " " in line:
            row = line.split()
            if row and all(cell in ["S", "F", "H", "G", "@"] for cell in row):
                grid.append(row)

    if not grid:
        raise ValueError("Invalid grid (empty)")
    n = len(grid[0])
    if any(len(r) != n for r in grid) or len(grid) != n:
        raise ValueError("Grid must be square with equal row lengths")
    return grid

def _create_gym_env_from_grid(grid):
    import gymnasium as gym
    desc = []
    for row in grid:
        row_string = ""
        for cell in row:
            if cell == "@":
                row_string += "S"
            elif cell in ["S", "F", "H", "G"]:
                row_string += cell
            else:
                raise ValueError(f"Unknown grid symbol: {cell}")
        desc.append(row_string)
    return gym.make("FrozenLake-v1", desc=desc, is_slippery=False)

def _parse_action(response: str):
    # keep your original tolerant level (exact words only)
    match = re.search(r"<answer>\s*(\w+)\s*</answer>", response, re.IGNORECASE)
    if match:
        action_name = match.group(1).strip().upper()
        return ACTION_NAME_TO_ID.get(action_name)
    return None

def _update_grid_after_move(prev_grid_str: str, obs_idx: int) -> str:
    size = int(len(prev_grid_str.split()) ** 0.5)
    x0, y0, _ = get_curr_pos(prev_grid_str)
    x, y = obs_idx // size, obs_idx % size
    grid_list = str_to_grid_list(prev_grid_str)
    if grid_list[x][y] == 'F':
        grid_list[x0][y0] = 'F'
        grid_list[x][y] = 'S'
    return grid_list_to_str(grid_list)

def create_model_filename(model_path, model_type, output_path):
    if "/" in model_path:
        model_name = model_path.split("/")[-1]
    else:
        model_name = model_path
    model_name = model_name.lower().replace("-", "_").replace(" ", "_")
    filename = f"{model_name}_base" if model_type == "base" else f"{model_name}_finetuned"
    return os.path.join(output_path, filename)

# -------------------------
# One-shot (your original, unchanged)
# -------------------------

def generate_responses(model, tokenizer, test_file, output_file, model_name):
    print(f"Generating responses with {model_name}...")
    tokenizer.padding_side = 'left'

    responses = []
    with open(test_file, 'r') as f:
        test_data = [json.loads(line) for line in f if line.strip()]
    print(f"Processing {len(test_data)} test cases...")

    batch_size = 100  # keep your default
    batches = [test_data[i:i+batch_size] for i in range(0, len(test_data), batch_size)]

    for batch in tqdm(batches, desc="Processing batches"):
        prompts = [item['prompt'] for item in batch]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=3000,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                use_cache=True
            )

        input_lengths = [len(inputs['input_ids'][j]) for j in range(len(batch))]

        for j, output in enumerate(outputs):
            response_tokens = output[input_lengths[j]:]
            response = tokenizer.decode(response_tokens, skip_special_tokens=True).strip()
            responses.append({'prompt': prompts[j], 'response': response})

    with open(output_file, 'w') as f:
        for item in responses:
            f.write(json.dumps(item) + '\n')

    print(f"Saved {len(responses)} responses to {output_file}")

# -------------------------
# Multi-turn (new but minimal)
# -------------------------

def generate_multiturn_responses(model, tokenizer, test_file, output_file, model_name,
                                 max_steps=200, max_new_tokens=1000, batch_size=100):
    print(f"Generating MULTI-TURN responses with {model_name}...")
    tokenizer.padding_side = 'left'
    tokenizer.truncation_side = 'left'
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model.eval()

    with open(test_file, 'r') as f:
        test_data = [json.loads(line) for line in f if line.strip()]
    print(f"Processing {len(test_data)} test cases...")

    episodes_out = []
    batches = [test_data[i:i+batch_size] for i in range(0, len(test_data), batch_size)]

    for batch in tqdm(batches, desc="Batches"):
        # prepare envs from prompt grids
        init_prompts = [item['prompt'] for item in batch]
        envs = []
        last_grid_str = []
        map_sizes = []
        
        for p in init_prompts:
            grid = _extract_grid_from_prompt(p)
            env = _create_gym_env_from_grid(grid)
            n = len(grid)
            map_sizes.append(n)
            env.reset()
            envs.append(env)
            last_grid_str.append(env_to_str(env))  # printable state

        # per-episode state
        done = [False] * len(init_prompts)
        steps = [0] * len(init_prompts)
        success = [False] * len(init_prompts)
        turns = [[] for _ in range(len(init_prompts))]

        # current prompts (start with dataset prompt as-is)
        curr_prompts = list(init_prompts)

        for _ in range(max_steps):
            alive = [i for i, d in enumerate(done) if not d]
            if not alive:
                break

            # tokenize/generate only for alive episodes
            inputs = tokenizer([curr_prompts[i] for i in alive],
                               return_tensors="pt", padding=True, truncation=True).to(model.device)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,            # greedy for eval
                    num_beams=1,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    use_cache=True
                )

            input_lens = [inputs['input_ids'][j].shape[0] for j in range(len(alive))]

            # decode and step envs
            for k, i in enumerate(alive):
                resp_tok = outputs[k, input_lens[k]:]
                resp_text = tokenizer.decode(resp_tok, skip_special_tokens=True).strip()
                turns[i].append({"prompt": curr_prompts[i], "response": resp_text})

                act = _parse_action(resp_text)
                if act is None:
                    done[i] = True
                    success[i] = False
                    continue

                obs, reward, terminated, truncated, info = envs[i].step(act)
                steps[i] += 1
                episode_done = bool(terminated or truncated)

                if episode_done:
                    done[i] = True
                    success[i] = bool(reward == 1.0)
                    continue

                # update printable grid and next prompt
                last_grid_str[i] = _update_grid_after_move(last_grid_str[i], int(obs))
                curr_prompts[i] = make_prompt(last_grid_str[i])

        # collect outputs for this batch
        for i in range(len(init_prompts)):
            episodes_out.append({
                "prompt": init_prompts[i],
                "turns": turns[i],
                "steps": steps[i],
                "success": success[i],
                "map_size": map_sizes[i]
            })

    # write JSONL
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    with open(output_file, 'w') as f:
        for ep in episodes_out:
            f.write(json.dumps(ep) + '\n')

    by_size = {}
    from collections import defaultdict
    acc = defaultdict(lambda: {"num_episodes": 0, "success": [], "steps": []})
    for ep in episodes_out:
        s = int(ep["map_size"])
        acc[s]["num_episodes"] += 1
        acc[s]["success"].append(int(ep["success"]))
        acc[s]["steps"].append(int(ep["steps"]))

    for s, v in acc.items():
        by_size[int(s)] = {
            "num_episodes": v["num_episodes"],
            "success_rate": float(np.mean(v["success"])) if v["success"] else 0.0,
            "avg_steps": float(np.mean(v["steps"])) if v["steps"] else 0.0,
        }

    return by_size

# -------------------------
# Your existing evaluate() kept for one-shot; new multiturn variant
# -------------------------

def evaluate_model(model_path, model_name, test_file, evaluator_class, output_path, model_type="base"):
    print(f"\n{'='*50}")
    print(f"Evaluating: {model_name}")
    print(f"Model path: {model_path}")
    print(f"{'='*50}")

    print("Loading model and tokenizer...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path, device_map="balanced", use_flash_attention_2=True, torch_dtype=torch.bfloat16
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    os.makedirs(output_path, exist_ok=True)
    base_filename = create_model_filename(model_path, model_type, output_path)
    output_file = f"{base_filename}_responses.jsonl"

    generate_responses(model, tokenizer, test_file, output_file, model_name)

    print(f"\nEvaluating frozen lake performance...")
    evaluator = evaluator_class()
    results = evaluator.evaluate_reasoning_paths(output_file)
    evaluator.print_detailed_results(results)

    results_file = f"{base_filename}_results.json"
    evaluator.save_results(results, results_file)
    return results

def evaluate_model_multiturn(model_path, model_name, test_file, output_path, model_type="finetuned"):
    print(f"\n{'='*50}")
    print(f"Evaluating (MULTI-TURN): {model_name}")
    print(f"Model path: {model_path}")
    print(f"{'='*50}")

    print("Loading model and tokenizer...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path, device_map="balanced", use_flash_attention_2=True, torch_dtype=torch.bfloat16
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    os.makedirs(output_path, exist_ok=True)
    base_filename = create_model_filename(model_path, model_type, output_path)
    output_file = f"{base_filename}_responses_multiturn.jsonl"

    summary = generate_multiturn_responses(
        model, tokenizer, test_file, output_file, model_name,
        max_steps=100, max_new_tokens=1000, batch_size=100
    )

    # also save summary json
    summary_file = f"{base_filename}_results_multiturn.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nDetailed results saved to:\n  - {summary_file}")
    print(f"Response files saved to:\n  - {output_file}")
    return summary

# -------------------------
# CLI
# -------------------------

def main():
    parser = argparse.ArgumentParser(description="Compare base vs fine-tuned model on frozen lake")
    parser.add_argument("--test-file", required=True, help="JSONL file with test prompts")
    parser.add_argument("--base-model", default="Qwen/Qwen3-14B", help="Base model path")
    parser.add_argument("--finetuned-model", default="./openrlhf_artifacts/sft_qwen14", help="Fine-tuned model path")
    parser.add_argument("--output-path", required=True, help="Output directory path")
    args = parser.parse_args()

    print("\nEVALUATING FINE-TUNED MODEL (MULTI-TURN)")
    _ = evaluate_model_multiturn(
        args.finetuned_model, "Fine-tuned Qwen",
        args.test_file, args.output_path, model_type="finetuned"
    )


    # filenames for reference (matches your prints)
    # base_filename = create_model_filename(args.base_model, "base", args.output_path)
    ft_filename = create_model_filename(args.finetuned_model, "finetuned", args.output_path)

    print(f"\nDetailed results saved to:\n  - {ft_filename}_results_multiturn.json")
    print(f"Response files saved to:\n  - {ft_filename}_responses_multiturn.jsonl")


if __name__ == "__main__":
    main()
