import argparse
import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

def create_model_filename(model_path, model_type, output_path):
    """Create descriptive filename based on model path and type"""
    # Extract model name from path
    if "/" in model_path:
        # For paths like "Qwen/Qwen3-8B" or "./openrlhf_artifacts/sft_qwen8"
        model_name = model_path.split("/")[-1]
    else:
        model_name = model_path
    
    # Clean up the model name and make it filename-safe
    model_name = model_name.lower().replace("-", "_").replace(" ", "_")
    
    # Add model type (base or finetuned)
    if model_type == "base":
        filename = f"{model_name}_base"
    else:
        filename = f"{model_name}_finetuned"
    
    return os.path.join(output_path, filename)

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

def evaluate_model(model_path, model_name, test_file, evaluator_class, output_path, model_type="base"):
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
    
    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    
    # Generate responses with descriptive filename
    base_filename = create_model_filename(model_path, model_type, output_path)
    output_file = f"{base_filename}_hilr_responses.jsonl"
    
    generate_responses(model, tokenizer, test_file, output_file, model_name)
    
    # Evaluate with your script
    print(f"\nEvaluating frozen lake performance...")
    evaluator = evaluator_class()
    results = evaluator.evaluate_reasoning_paths(output_file)
    
    # Print results
    evaluator.print_detailed_results(results)
    
    # Save detailed results
    results_file = f"{base_filename}_results.json"
    evaluator.save_results(results, results_file)
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Compare base vs fine-tuned model on frozen lake")
    parser.add_argument("--test-file", default="/n/holylfs06/LABS/kempner_undergrads/Lab/ellenma/openrlhf-proj/AgentsOpenRLHF/data/frozen_lake/test_deduplicated.jsonl", help="JSONL file with test prompts")
    parser.add_argument("--base-model", default="Qwen/Qwen3-14B", help="Base model path")
    parser.add_argument("--finetuned-model", default="./openrlhf_artifacts/sft_qwen14", help="Fine-tuned model path")
    parser.add_argument("--output-path", default="/n/holylfs06/LABS/kempner_undergrads/Lab/ellenma/openrlhf-proj/AgentsOpenRLHF/eval_output/", help="Output directory path")
    
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
        FrozenLakeEvaluator,
        args.output_path,
        model_type="base"
    )
    
    # Evaluate fine-tuned model
    print("\nEVALUATING FINE-TUNED MODEL")
    ft_results = evaluate_model(
        args.finetuned_model, 
        "Fine-tuned Qwen", 
        args.test_file, 
        FrozenLakeEvaluator,
        args.output_path,
        model_type="finetuned"
    )
    
    # Compare results
    print(f"\n{'='*60}")
    print("COMPARISON RESULTS")
    print(f"{'='*60}")
    print(f"Base model success rate:       {base_results['success_rate']:.2%}")
    print(f"Fine-tuned model success rate: {ft_results['success_rate']:.2%}")
    
    improvement = ft_results['success_rate'] - base_results['success_rate']
    print(f"Improvement:                   {improvement:+.2%}")
    
    # Show actual filenames that were created
    base_filename = create_model_filename(args.base_model, "base", args.output_path)
    ft_filename = create_model_filename(args.finetuned_model, "finetuned", args.output_path)
    
    print(f"\nDetailed results saved to:")
    print(f"  - {base_filename}_results.json")
    print(f"  - {ft_filename}_results.json")
    print(f"\nResponse files saved to:")
    print(f"  - {base_filename}_responses.jsonl")
    print(f"  - {ft_filename}_responses.jsonl")

if __name__ == "__main__":
    main()