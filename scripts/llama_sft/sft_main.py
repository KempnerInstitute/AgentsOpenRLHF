"""
main.py

Main evaluation script for SFT Llama 8B on FrozenLake navigation tasks.

Usage:
    python sft_main.py --model_path /path/to/sft_model --dataset /path/to/test.jsonl
"""

import argparse
import logging
import os
from datetime import datetime
from pathlib import Path
from sft_llm_policy import SFTLLMPolicy
from sft_evaluator import SFTEvaluator

def main():
    parser = argparse.ArgumentParser(description="Evaluate SFT Llama 8B on FrozenLake")
    
    # Model arguments
    parser.add_argument("--model_path", required=True, 
                       help="Path to fine-tuned Llama model directory")
    parser.add_argument("--dataset", required=True,
                       help="Path to JSONL dataset file")
    parser.add_argument("--temperature", type=float, default=0.1,
                       help="Sampling temperature (default: 0.1 for deterministic)")
    parser.add_argument("--max_tokens", type=int, default=200,
                       help="Maximum tokens to generate")
    
    # Evaluation arguments  
    parser.add_argument("--results_dir", default="./sft_results",
                       help="Directory to save results")
    parser.add_argument("--no_quantization", action="store_true",
                       help="Disable 4-bit quantization")
    
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    # Validate inputs
    if not Path(args.model_path).exists():
        logger.error(f"Model path does not exist: {args.model_path}")
        return
    
    if not Path(args.dataset).exists():
        logger.error(f"Dataset file does not exist: {args.dataset}")
        return
    
    # Create results directory
    os.makedirs(args.results_dir, exist_ok=True)
    
    # Create base filename for results (without extension)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = Path(args.model_path).name
    dataset_name = Path(args.dataset).stem
    results_base = f"{args.results_dir}/sft_eval_{model_name}_{dataset_name}_{timestamp}"
    
    try:
        # Initialize SFT model policy
        logger.info(f"Loading SFT model from: {args.model_path}")
        policy = SFTLLMPolicy(
            model_path=args.model_path,
            temperature=args.temperature,
            use_quantization=not args.no_quantization,
            max_tokens=args.max_tokens
        )
        
        # Initialize evaluator
        evaluator = SFTEvaluator()
        
        # Run evaluation
        logger.info(f"Starting evaluation on dataset: {args.dataset}")
        results = evaluator.run_evaluation(policy, args.dataset)
        
        # Save results (creates both summary and detailed files)
        summary_file, detailed_file = evaluator.save_results(results_base)
        
        # Print summary
        summary = results["summary"]
        print("\n" + "="*50)
        print("EVALUATION SUMMARY")
        print("="*50)
        print(f"Model: {model_name}")
        print(f"Dataset: {dataset_name}")
        print(f"Temperature: {args.temperature}")
        print("-"*30)
        print(f"Overall Accuracy: {summary['overall_accuracy']:.2%}")
        print(f"Total Problems: {summary['total_problems']}")
        print(f"Correct Predictions: {summary['total_correct']}")
        print("-"*30)
        print("Accuracy by Grid Size:")
        
        for size in sorted(results["by_grid_size"].keys()):
            stats = results["by_grid_size"][size]
            print(f"  {size}x{size}: {stats['accuracy']:.2%} "
                  f"({stats['correct']}/{stats['total']}) "
                  f"[Parse: {stats['parsing_success_rate']:.1%}]")
        
        print("-"*30)
        print(f"Summary results: {summary_file}")
        print(f"Detailed results: {detailed_file}")
        print("="*50)
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise
    
    finally:
        # Clean up model resources
        try:
            policy.cleanup()
        except:
            pass

if __name__ == "__main__":
    main()