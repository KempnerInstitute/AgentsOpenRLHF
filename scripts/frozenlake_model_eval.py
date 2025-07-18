#!/usr/bin/env python3
"""
Evaluate both base Qwen and fine-tuned Qwen on frozen lake problems
"""

import json
from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
from pathlib import Path
from scripts.fl_evaluator import FrozenLakeEvaluator

# def generate_responses(model, tokenizer, test_file, output_file, model_name):
#     """Generate model responses for all test prompts"""
#     print(f"Generating responses with {model_name}...")
    
#     responses = []
    
#     with open(test_file, 'r') as f:
#         test_data = [json.loads(line) for line in f if line.strip()]
    
#     print(f"Processing {len(test_data)} test cases...")
    
#     for i, item in enumerate(tqdm(test_data, desc="Processing")):
#         # if i % 10 == 0:
#         #     print(f"Progress: {i}/{len(test_data)}")
            
#         prompt = item['prompt']
        
#         # Tokenize and generate
#         inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
#         with torch.no_grad():
#             outputs = model.generate(
#                 **inputs, 
#                 max_length=512, 
#                 temperature=0.1,
#                 do_sample=True,
#                 pad_token_id=tokenizer.eos_token_id,
#                 repetition_penalty=1.1,
#                 use_cache=True
#             )
        
#         # Decode response (remove the prompt part)
#         full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
#         response = full_response[len(prompt):].strip()
        
#         # Save in format your evaluator expects
#         responses.append({
#             'prompt': prompt,
#             'response': response
#         })
    
#     # Save responses
#     with open(output_file, 'w') as f:
#         for item in responses:
#             f.write(json.dumps(item) + '\n')
    
#     print(f"Saved {len(responses)} responses to {output_file}")

def generate_responses(model, tokenizer, test_file, output_file, model_name):
    """Generate model responses for all test prompts"""
    print(f"Generating responses with {model_name}...")
    
    responses = []
    
    with open(test_file, 'r') as f:
        test_data = [json.loads(line) for line in f if line.strip()]
    
    print(f"Processing {len(test_data)} test cases...")
    
    batch_size = 8  # Start with 8, adjust based on memory
    
    # Create batches and wrap with tqdm
    batches = [test_data[i:i+batch_size] for i in range(0, len(test_data), batch_size)]
    
    for batch in tqdm(batches, desc="Processing batches"):
        prompts = [item['prompt'] for item in batch]
        
        # Tokenize batch
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=300,  
                temperature=0.1,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.1,
                use_cache=True
            )
        
        # Process batch results
        for j, output in enumerate(outputs):
            full_response = tokenizer.decode(output, skip_special_tokens=True)
            response = full_response[len(prompts[j]):].strip()
            responses.append({
                'prompt': prompts[j],
                'response': response
            })
    
    # Save responses
    with open(output_file, 'w') as f:
        for item in responses:
            f.write(json.dumps(item) + '\n')
    
    print(f"Saved {len(responses)} responses to {output_file}")

def evaluate_model(model_path, model_name, test_file, evaluator_class):
    """Load model and evaluate on frozen lake"""
    print(f"\n{'='*50}")
    print(f"Evaluating: {model_name}")
    print(f"Model path: {model_path}")
    print(f"{'='*50}")
    
    # Load model
    print("Loading model and tokenizer...")
    model = AutoModelForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Using device: {device}")
    
    # Generate responses
    output_file = f"{model_name.lower().replace(' ', '_')}_responses.jsonl"
    generate_responses(model, tokenizer, test_file, output_file, model_name)
    
    # Evaluate with your script
    print(f"\nEvaluating frozen lake performance...")
    evaluator = evaluator_class()
    results = evaluator.evaluate_reasoning_paths(output_file)
    
    # Print results
    evaluator.print_detailed_results(results)
    
    # Save detailed results
    results_file = f"{model_name.lower().replace(' ', '_')}_results.json"
    evaluator.save_results(results, results_file)
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Compare base vs fine-tuned model on frozen lake")
    parser.add_argument("--test_file", default="/n/holylfs06/LABS/kempner_undergrads/Lab/ellenma/openrlhf-proj/AgentsOpenRLHF/data/frozen_lake/test_deduplicated.jsonl", help="JSONL file with test prompts")
    parser.add_argument("--base_model", default="Qwen/Qwen3-8B", help="Base model path")
    parser.add_argument("--finetuned_model", default="./openrlhf_artifacts/sft_qwen8", help="Fine-tuned model path")
    
    args = parser.parse_args()
    
    # Import your evaluator
    try:
        from scripts.fl_evaluator import FrozenLakeEvaluator
    except ImportError:
        print("ERROR: Could not import FrozenLakeEvaluator")
        print("Make sure your evaluation script is in the same directory or Python path")
        print("Update the import line at the top of this script")
        return
    
    # Evaluate base model
    print("EVALUATING BASE MODEL")
    base_results = evaluate_model(
        args.base_model, 
        "Base Qwen", 
        args.test_file, 
        FrozenLakeEvaluator
    )
    
    # Evaluate fine-tuned model
    print("\nEVALUATING FINE-TUNED MODEL")
    ft_results = evaluate_model(
        args.finetuned_model, 
        "Fine-tuned Qwen", 
        args.test_file, 
        FrozenLakeEvaluator
    )
    
    # Compare results
    print(f"\n{'='*60}")
    print("COMPARISON RESULTS")
    print(f"{'='*60}")
    print(f"Base model success rate:       {base_results['success_rate']:.2%}")
    print(f"Fine-tuned model success rate: {ft_results['success_rate']:.2%}")
    
    improvement = ft_results['success_rate'] - base_results['success_rate']
    print(f"Improvement:                   {improvement:+.2%}")
    
    print(f"\nDetailed results saved to:")
    print(f"  - base_qwen8_results.json")
    print(f"  - fine_tuned_qwen8_results.json")

if __name__ == "__main__":
    main()