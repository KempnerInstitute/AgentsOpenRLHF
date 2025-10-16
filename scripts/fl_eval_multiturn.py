import argparse
import asyncio
import os
import json
import numpy as np
import torch
from collections import defaultdict
from scripts.fl_agent_tool import FrozenLakeAgentInstanceTool
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import scripts.fl as fl 

async def _agent_batch_step(agents, curr_prompts, actions, alive):
    tasks = [agents[i].step(curr_prompts[i], actions[k], label=None) for k, i in enumerate(alive)]
    return await asyncio.gather(*tasks)

def get_curr_pos(string):
    curr_grid_flat = string.split()
    init_obs = curr_grid_flat.index('S')
    size = int(len(curr_grid_flat)**0.5)
    x,y = divmod(init_obs, size)
    return x, y, size

def create_model_filename(model_path, model_type, output_path):
    if "/" in model_path:
        model_name = model_path.split("/")[-1]
    else:
        model_name = model_path
    model_name = model_name.lower().replace("-", "_").replace(" ", "_")
    filename = f"{model_name}_base" if model_type == "base" else f"{model_name}_finetuned"
    return os.path.join(output_path, filename)

def generate_multiturn_responses(model, tokenizer, test_file, output_file, model_name,
                                 max_steps=200, max_new_tokens=1000, batch_size=100):
    print(f"Generating MULTI-TURN responses with {model_name}...")
    tokenizer.padding_side = 'left'
    tokenizer.truncation_side = 'left'
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model.eval()

    # load test data
    with open(test_file, 'r') as f:
        test_data = [json.loads(line) for line in f if line.strip()]
    print(f"Processing {len(test_data)} test cases...")

    episodes_out = []
    batches = [test_data[i:i+batch_size] for i in range(0, len(test_data), batch_size)]

    # batching
    for batch in tqdm(batches, desc="Batches"):
        # prepare envs from prompt grids
        init_prompts = [item['prompt'] for item in batch]
        map_sizes = []
        agents = []
        
        for p in init_prompts:
            grid = fl.extract_grid_from_prompt(p)
            map_sizes.append(len(grid))
            agent = FrozenLakeAgentInstanceTool()
            agent.env_str = fl.grid_list_to_str(grid)
            agent.env = fl.create_gym_env_from_grid(grid)
            agent.env.reset()
            agents.append(agent)


        # per-episode state
        done = [False] * len(init_prompts)
        steps = [0] * len(init_prompts)
        success = [False] * len(init_prompts)
        turns = [[] for _ in range(len(init_prompts))]

        # current prompts (start with dataset prompt as-is)
        curr_prompts = list(init_prompts)
        histories = list(init_prompts)

        for _ in range(max_steps):
            alive = [i for i, d in enumerate(done) if not d]
            if not alive:
                break

            # tokenize/generate only for alive episodes
            inputs = tokenizer([curr_prompts[i] for i in alive],
                               return_tensors="pt", padding=True, truncation=True)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,            # greedy for eval
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    use_cache=True
                )
            
            # # left padding: same input length for all rows
            # pad_len = inputs['input_ids'].shape[1]

            # # decode model responses
            # resp_texts = [
            #     tokenizer.decode(outputs[k][pad_len:], skip_special_tokens=True).strip()
            #     for k in range(len(alive))
            # ]
            
            input_lens = inputs["attention_mask"].sum(dim=1).tolist()
            resp_texts = [
                tokenizer.decode(outputs[k][int(input_lens[k]):], skip_special_tokens=True).strip()
                for k in range(len(alive))
            ]
            
            # async step
            results = asyncio.run(_agent_batch_step(agents, curr_prompts, resp_texts, alive))

            # update episodes
            for k, i in enumerate(alive):
                resp_text = resp_texts[k]
                res = results[k]
                
                # log turn 
                turns[i].append({"prompt": curr_prompts[i], "response": resp_text})
                histories[i] += resp_text
                steps[i] += 1
                
                if bool(res.get("done", False)):
                    # success if reward/score == 1.0
                    score_arr = res.get("scores", res.get("rewards", np.array([0.0])))
                    rew = float(score_arr[0]) if isinstance(score_arr, (list, np.ndarray)) else float(score_arr)
                    success[i] = (rew >= 1.0)
                    done[i] = True
                    continue

                # agent returns observation + action + next_prompt; keep only the tail (next_prompt)
                next_obs = res.get("next_observation", "")
                prefix = curr_prompts[i] + resp_text
                next_prompt_tail = next_obs[len(prefix):] if next_obs.startswith(prefix) else next_obs

                histories[i] += next_prompt_tail
                curr_prompts[i] = histories[i]     

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
            "solved": float(np.sum(v["success"])) if v["success"] else 0.0
        }

    return by_size


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


def main():
    parser = argparse.ArgumentParser(description="Compare base vs fine-tuned model on frozen lake")
    parser.add_argument("--test-file", required=True, help="JSONL file with test prompts")
    parser.add_argument("--finetuned-model", required=True, help="Fine-tuned model path")
    parser.add_argument("--output-path", required=True, help="Output directory path")
    args = parser.parse_args()

    print("\nEVALUATING FINE-TUNED MODEL (MULTI-TURN)")
    
    evaluate_model_multiturn(
        args.finetuned_model, "Fine-tuned Qwen",
        args.test_file, args.output_path, model_type="finetuned"
    )

    ft_filename = create_model_filename(args.finetuned_model, "finetuned", args.output_path)

    print(f"\nDetailed results saved to:\n  - {ft_filename}_results_multiturn.json")
    print(f"Response files saved to:\n  - {ft_filename}_responses_multiturn.jsonl")


if __name__ == "__main__":
    main()