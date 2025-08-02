#!/bin/bash
#SBATCH --job-name="rl_06flake"
#SBATCH --account=kempner_undergrads
#SBATCH --output /n/holylfs06/LABS/kempner_undergrads/Lab/ellenma/openrlhf-proj/AgentsOpenRLHF/rl_run_logs/06_%A.log
#SBATCH --error /n/holylfs06/LABS/kempner_undergrads/Lab/ellenma/openrlhf-proj/AgentsOpenRLHF/rl_run_logs/06_error_%j.out
#SBATCH --export=ALL
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --mem=128GB
#SBATCH --time=15:00:00
#SBATCH --partition=kempner

module purge

module load cuda/12.4.1-fasrc01
module load gcc/12.2.0-fasrc01
module load cmake/3.31.6-fasrc01
module load cudnn
module load python/3.12.8-fasrc01
source /n/holylfs06/LABS/kempner_undergrads/Lab/ellenma/openrlhf-proj/AgentsOpenRLHF/.venv/bin/activate

# CUDA environment variables
export CUDA_HOME=/n/sw/helmod-rocky8/apps/Core/cuda/12.4.1-fasrc01/cuda
export CUDA_ROOT=$CUDA_HOME
export PATH=$CUDA_HOME/bin:$PATH

# Library paths for CUDA
export LD_LIBRARY_PATH=$CUDA_HOME/lib64/stubs:$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export LIBRARY_PATH=$CUDA_HOME/lib64/stubs:$CUDA_HOME/lib64:$LIBRARY_PATH

# Combined PYTHONPATH (put system Python first, then your project)
export PYTHONPATH="/n/holylfs06/LABS/kempner_undergrads/Lab/ellenma/openrlhf-proj/AgentsOpenRLHF:$PYTHONPATH"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

# debug custom-reduce-all
# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=ALL
export NCCL_SOCKET_IFNAME=ib0



if [[ -z $SLURM_GPUS_ON_NODE ]]; then
    RAY_NUM_GPUS=0
else
    RAY_NUM_GPUS=$SLURM_GPUS_ON_NODE
fi

# choose available port on the head node
head_port=`comm -23 <(seq 15000 20000 | sort) <(ss -Htan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1`
nodes=`scontrol show hostnames $SLURM_JOB_NODELIST`
nodes_array=( $nodes )
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)
echo "Head node: $head_node"
echo "Head node ip: $head_node_ip"
echo "Head port: $head_port"
export RAY_HEAD_ADDR="$head_node_ip:$head_port"
echo "Head address: $RAY_HEAD_ADDR"

echo "Starting Ray head on $head_node"
srun -N 1 -n 1 -w "$head_node" ray start --head --node-ip-address="$head_node_ip" --temp-dir /tmp/$USER/$SLURM_JOB_ID/ray \
    --port=$head_port --num-cpus $SLURM_CPUS_PER_TASK --num-gpus $RAY_NUM_GPUS --min-worker-port 20001 --max-worker-port 30000 --block &

# wait for head node to start
sleep 20

# start ray on the rest of the nodes
worker_num=$((SLURM_NNODES - 1))
for (( i = 1; i <= worker_num; i++ )); do
    node=${nodes_array[$i]}
    echo "Starting Ray worker on $node"
    srun -N 1 -n 1 -w "$node" ray start --address="$RAY_HEAD_ADDR" \
        --num-cpus $SLURM_CPUS_PER_TASK --num-gpus $RAY_NUM_GPUS --min-worker-port 20001 --max-worker-port 30000 --block &
    sleep 10
done

export RAY_ADDRESS="$RAY_HEAD_ADDR"

#  # --remote_rm_url http://$head_node:5000/get_reward \

#/n/holylabs/LABS/sham_lab/Users/mkwun/inference-reasoning/openrlhf_wd/sft_llama3_8b \
#/n/holylabs/LABS/sham_lab/Users/mkwun/inference-reasoning/models--meta-llama--Meta-Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659
# /n/netscratch/sham_lab/Everyone/mkwun/sft_llama


srun --overlap -N 1 -n 1 -w "$head_node" ray job submit --address="http://127.0.0.1:8265" \
  --runtime-env-json='{"working_dir": "/n/netscratch/kempner_undergrads/Lab/ellenma/rl_flake_working_dir", "env_vars": {"HUGGINGFACE_HUB_TOKEN": "'$HUGGINGFACE_HUB_TOKEN'"}}' \
  -- python3 -m openrlhf.cli.train_ppo_ray \
  --ref_num_nodes 1 \
  --ref_num_gpus_per_node 1 \
  --colocate_actor_ref \
  --actor_num_nodes 1 \
  --actor_num_gpus_per_node 1 \
  --pretrain /n/holylfs06/LABS/kempner_undergrads/Lab/ellenma/openrlhf-proj/AgentsOpenRLHF/openrlhf_artifacts/sft_qwen06 \
  --save_path /n/netscratch/kempner_undergrads/Lab/ellenma/openrlhf_artifacts/frozenlake_rl/ \
  --vllm_num_engines 1 \
  --vllm_tensor_parallel_size 1 \
  --micro_train_batch_size 2 \
  --train_batch_size 32 \
  --micro_rollout_batch_size 4 \
  --rollout_batch_size 16 \
  --max_samples 50000 \
  --max_epochs 3 \
  --num_episodes 10 \
  --prompt_max_len 1536 \
  --generate_max_len 512 \
  --n_samples_per_prompt 4 \
  --zero_stage 2 \
  --bf16 \
  --actor_learning_rate 5e-7 \
  --advantage_estimator group_norm \
  --eps_clip 0.0 \
  --normalize_reward \
  --packing_samples \
  --adam_offload \
  --flash_attn \
  --gradient_checkpointing \
  --use_wandb True \
  --wandb_project frozenlake_rl \
  --wandb_run_name sft_06B \
  --agent_func_path /n/holylfs06/LABS/kempner_undergrads/Lab/ellenma/openrlhf-proj/AgentsOpenRLHF/scripts/gym_agent.py \
  --prompt_data /n/holylfs06/LABS/kempner_undergrads/Lab/ellenma/openrlhf-proj/AgentsOpenRLHF/data/frozen_lake/rl_dataset.jsonl \
  --input_key prompt \
  --async_train \

