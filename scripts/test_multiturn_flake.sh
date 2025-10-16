#!/bin/bash
#SBATCH --job-name="eval_8"
#SBATCH --account=kempner_sham_lab
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128GB
#SBATCH --partition=kempner
#SBATCH --output /n/holylfs06/LABS/kempner_undergrads/Lab/ellenma/openrlhf-proj/AgentsOpenRLHF/eval_logs/frozenlake_rl/8_s%A.log
#SBATCH --error /n/holylfs06/LABS/kempner_undergrads/Lab/ellenma/openrlhf-proj/AgentsOpenRLHF/eval_logs/frozenlake_rl/8_serror_%j.out
#SBATCH --export=ALL
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ellenma@g.harvard.edu

source ~/.bashrc
source /n/holylfs06/LABS/kempner_undergrads/Lab/ellenma/openrlhf-proj/AgentsOpenRLHF/.venv/bin/activate
cd /n/holylfs06/LABS/kempner_undergrads/Lab/ellenma/openrlhf-proj/AgentsOpenRLHF 

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/n/netscratch/kempner_undergrads/Lab/ellenma/moved/hf_cache



python3 -m scripts.fl_eval_multiturn \
    --test-file /n/holylfs06/LABS/kempner_undergrads/Lab/ellenma/openrlhf-proj/AgentsOpenRLHF/data/frozen_lake/test_merged_600_example.jsonl \
    --finetuned-model "/n/holylfs06/LABS/kempner_undergrads/Lab/ellenma/openrlhf-proj/AgentsOpenRLHF/openrlhf_artifacts/frozenlake_rl_tool/rl_ftqwen8_ppo_example1" \
    --output-path /n/holylfs06/LABS/kempner_undergrads/Lab/ellenma/openrlhf-proj/AgentsOpenRLHF/eval_output/frozenlake_rl_multiturn_tool/ppo_example_1_ex

# python3 -m scripts.fl_eval_multiturn \
#     --test-file /n/holylfs06/LABS/kempner_undergrads/Lab/ellenma/openrlhf-proj/AgentsOpenRLHF/data/frozen_lake/rl_train_sample100.jsonl \
#     --finetuned-model /n/holylfs06/LABS/kempner_undergrads/Lab/ellenma/openrlhf-proj/AgentsOpenRLHF/openrlhf_artifacts/frozenlake_rl_multiturn/rl_qwen8\
#     --output-path /n/holylfs06/LABS/kempner_undergrads/Lab/ellenma/openrlhf-proj/AgentsOpenRLHF/eval_output/frozenlake_rl_multiturn_no_tool/multiturn_test