#!/bin/bash
#SBATCH --job-name="sft_qw8"
#SBATCH --account=kempner_undergrads
#SBATCH --output /n/holylfs06/LABS/kempner_undergrads/Lab/ellenma/openrlhf-proj/AgentsOpenRLHF/run_logs/qwen_%A.log
#SBATCH --error /n/holylfs06/LABS/kempner_undergrads/Lab/ellenma/openrlhf-proj/AgentsOpenRLHF/run_logs/qwen_error_%j.out  # Error file
#SBATCH --export=ALL
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=64                     
#SBATCH --mem=192GB                               
#SBATCH --time=05:00:00                 
#SBATCH --partition=kempner_h100

module load cuda/12.4.1-fasrc01
export LD_LIBRARY_PATH=/n/sw/helmod-rocky8/apps/Core/cuda/11.8.0-fasrc01/lib64:$LD_LIBRARY_PATH

module load cudnn
module load gcc/12.2.0-fasrc01

source ~/.bashrc
source /n/holylfs06/LABS/kempner_undergrads/Lab/ellenma/openrlhf-proj/AgentsOpenRLHF/.venv/bin/activate

export NCCL_SOCKET_FAMILY=AF_INET
export NCCL_SOCKET_IFNAME=ib0
export NCCL_DEBUG=INFO
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TRITON_CACHE_DIR="/tmp/ellenma/triton_cache_8b"

# Distributed Training Configuration
export GPUS_PER_NODE=4
export NNODES=$SLURM_NNODES
export NUM_PROCESSES=$(expr $NNODES \* $GPUS_PER_NODE)

export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT_CANDIDATES=(6000 6001 6002 6003)

# Choose an available port on the head node
for MPC in ${MASTER_PORT_CANDIDATES[@]}; do
    NUM_LISTENING_PROCESSES=$(lsof -Pi :${MPC} -sTCP:LISTEN | wc -l)
    if test $NUM_LISTENING_PROCESSES -eq 0; then
        MASTER_PORT=${MPC}
        export MASTER_PORT=${MPC}
        echo "Setting master port to ${MASTER_PORT}."
        break
    fi
done

if [ -z ${MASTER_PORT+x} ]; then
    echo "Could not find an available master port. Exiting."
    exit
fi

# Launch SFT Training with DeepSpeed
srun deepspeed --module openrlhf.cli.train_sft \
   --max_len 768 \
   --dataset data/frozen_lake/frozen_lake_train.jsonl \
   --eval_dataset data/frozen_lake/frozen_lake_val.jsonl \
   --input_key prompt \
   --output_key response \
   --pretrain Qwen/Qwen3-8B \
   --save_path /n/holylfs06/LABS/kempner_undergrads/Lab/ellenma/openrlhf-proj/AgentsOpenRLHF/openrlhf_artifacts/sft_qwen8 \
   --apply_chat_template \
   --input_template "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{}\n<|im_end|>\n<|im_start|>assistant\n" \
   --train_batch_size 8 \
   --micro_train_batch_size 1 \
   --max_samples 1000 \
   --save_steps 50 \
   --logging_steps 1 \
   --eval_steps 50 \
   --zero_stage 2 \
   --max_epochs 3 \
   --bf16 \
   --flash_attn \
   --learning_rate 5e-5 \
   --gradient_checkpointing \
   --use_wandb true 

   #    --packing_samples \
# try learning rates of 5e-5 and 5e-4