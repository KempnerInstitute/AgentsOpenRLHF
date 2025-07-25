#!/bin/bash
#SBATCH --job-name="eval"
#SBATCH --account=kempner_undergrads
#SBATCH --time=5:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --partition=kempner_h100
#SBATCH --output /n/holylfs06/LABS/kempner_undergrads/Lab/myrahmoun/AgentsOpenRLHF/run_logs/%A.log
#SBATCH --error /n/holylfs06/LABS/kempner_undergrads/Lab/myrahmoun/AgentsOpenRLHF/run_logs//error_%j.out
#SBATCH --export=ALL

source ~/.bashrc
source /n/holylfs06/LABS/kempner_undergrads/Lab/myrahmoun/AgentsOpenRLHF/.venv/bin/activate
cd /n/holylfs06/LABS/kempner_undergrads/Lab/myrahmoun/AgentsOpenRLHF


export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True


python3 -m scripts.llama_frozenlake_model_eval \
    --test-file /n/holylfs06/LABS/kempner_undergrads/Lab/myrahmoun/AgentsOpenRLHF/data/frozen_lake/test_merged_600.jsonl \
    --base-model meta-llama/Llama-3.1-8B-Instruct \
    --finetuned-model /n/holylfs06/LABS/kempner_undergrads/Lab/myrahmoun/AgentsOpenRLHF/openrlhf_artifacts/sft_llama8 \
    --output-path /n/holylfs06/LABS/kempner_undergrads/Lab/myrahmoun/AgentsOpenRLHF/eval_output