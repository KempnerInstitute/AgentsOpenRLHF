"""
Main script for comparing Base Llama 8B vs SFT Llama 8B on FrozenLake navigation tasks.

This script runs both models on the same dataset and provides side-by-side comparison.

Usage:
    python comparison_main.py --base_model meta-llama/Llama-3.1-8B-Instruct --sft_model_path /path/to/sft_model --dataset /path/to/test.jsonl
"""

import argparse
import logging
import os
from datetime import datetime
from pathlib import Path
from sft_llm_policy import SFTLLMPolicy
from base_llm_policy import BaseLLMPolicy
from sft_evaluator import ComparisonEvaluator

def main():
    parser = argparse.ArgumentParser(description="Compare Base Llama 8B vs SFT Llama 8B on FrozenLake")
    
    # Model arguments
    parser.add_argument("--base_model", default="meta-llama/Llama-3.1-8B-Instruct",
                       help="Base Llama model name")
    parser.add_argument("--sft_model_path", required=True, 
                       help="Path to fine-tuned Llama model directory")
    parser.add_argument("--dataset", required=True,
                       help="Path to JSONL dataset file")
    parser.add_argument("--temperature", type=float, default=0.1,
                       help="Sampling temperature (default: 0.1 for deterministic)")
    parser.add_argument("--max_tokens", type=int, default=200,
                       help="Maximum tokens to generate")
    
    # Evaluation arguments  
    parser.add_argument("--results_dir", default="./comparison_results",
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
    if not Path(args.sft_model_path).exists():
        logger.error(f"SFT model path does not exist: {args.sft_model_path}")
        return
    
    if not Path(args.dataset).exists():
        logger.error(f"Dataset file does not exist: {args.dataset}")
        return
    
    # Create results directory
    os.makedirs(args.results_dir, exist_ok=True)
    
    # Create base filename for results (without extension)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sft_model_name = Path(args.sft_model_path).name
    dataset_name = Path(args.dataset).stem
    results_base = f"{args.results_dir}/comparison_{sft_model_name}_{dataset_name}_{timestamp}"
    
    base_policy = None
    sft_policy = None
    
    try:
        # Initialize base model policy
        logger.info(f"Loading base model: {args.base_model}")
        base_policy = BaseLLMPolicy(
            model_name=args.base_model,
            temperature=args.temperature,
            use_quantization=not args.no_quantization,
            max_tokens=args.max_tokens
        )
        
        # Initialize SFT model policy
        logger.info(f"Loading SFT model from: {args.sft_model_path}")
        sft_policy = SFTLLMPolicy(
            model_path=args.sft_model_path,
            temperature=args.temperature,
            use_quantization=not args.no_quantization,
            max_tokens=args.max_tokens
        )
        
        # Initialize evaluator
        evaluator = ComparisonEvaluator()
        
        # Run evaluation
        logger.info(f"Starting comparison evaluation on dataset: {args.dataset}")
        results = evaluator.run_evaluation(base_policy, sft_policy, args.dataset)
        
        # Save results (creates both summary and detailed files)
        summary_file, detailed_file = evaluator.save_results(results_base)
        
        # Print summary
        base_summary = results["summary"]["base"]
        sft_summary = results["summary"]["sft"]
        improvement = results["summary"]["improvement"]
        
        print("\n" + "="*60)
        print("BASE vs SFT MODEL COMPARISON")
        print("="*60)
        print(f"Base Model: {args.base_model}")
        print(f"SFT Model: {sft_model_name}")
        print(f"Dataset: {dataset_name}")
        print(f"Temperature: {args.temperature}")
        print("-"*40)
        print(f"Base Model Accuracy: {base_summary['overall_accuracy']:.2%}")
        print(f"SFT Model Accuracy:  {sft_summary['overall_accuracy']:.2%}")
        print(f"Improvement:         {improvement:+.2%}")
        print(f"Total Problems:      {base_summary['total_problems']}")
        print("-"*40)
        print("Accuracy by Grid Size:")
        
        for size in sorted(results["by_grid_size"].keys()):
            base_stats = results["by_grid_size"][size]["base"]
            sft_stats = results["by_grid_size"][size]["sft"]
            size_improvement = sft_stats['accuracy'] - base_stats['accuracy']
            print(f"  {size}x{size}: Base {base_stats['accuracy']:.2%} "
                  f"({base_stats['correct']}/{base_stats['total']}) | "
                  f"SFT {sft_stats['accuracy']:.2%} "
                  f"({sft_stats['correct']}/{sft_stats['total']}) | "
                  f"Δ {size_improvement:+.2%}")
        
        print("-"*40)
        print(f"Summary results: {summary_file}")
        print(f"Detailed results: {detailed_file}")
        print("="*60)
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise
    
    finally:
        # Clean up model resources
        try:
            if base_policy:
                base_policy.cleanup()
            if sft_policy:
                sft_policy.cleanup()
        except:
            pass

if __name__ == "__main__":
    main()