#!/bin/bash
#SBATCH --job-name="sft_llama8"
#SBATCH --account=kempner_undergrads
#SBATCH --partition=kempner_h100    
#SBATCH --output /n/holylfs06/LABS/kempner_undergrads/Lab/myrahmoun/AgentsOpenRLHF/run_logs/llama_%A.log
#SBATCH --error /n/holylfs06/LABS/kempner_undergrads/Lab/myrahmoun/AgentsOpenRLHF/run_logs/llama_%j.out  # Error file
#SBATCH --export=ALL
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=64                     
#SBATCH --mem=192GB                               
#SBATCH --time=05:00:00 

module load cuda/12.4.1-fasrc01
export LD_LIBRARY_PATH=/n/sw/helmod-rocky8/apps/Core/cuda/11.8.0-fasrc01/lib64:$LD_LIBRARY_PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

module load cudnn
module load gcc/12.2.0-fasrc01

source ~/.bashrc
source /n/holylfs06/LABS/kempner_undergrads/Lab/myrahmoun/AgentsOpenRLHF/.venv/bin/activate

# ===========================================
# AUTO-CLEANUP SECTION - Free up disk space  
# ===========================================
echo "Starting cleanup to free disk space..."
echo "Before cleanup:"
quota /n/holylfs06/LABS/kempner_undergrads/Lab/myrahmoun/

# Remove old training checkpoints (137GB!)
echo "Removing old checkpoints..."
rm -rf /n/holylfs06/LABS/kempner_undergrads/Lab/myrahmoun/AgentsOpenRLHF/scripts/llama_sft/ckpt/
rm -rf /n/holylfs06/LABS/kempner_undergrads/Lab/myrahmoun/AgentsOpenRLHF/scripts/llama_sft/wandb/

# Clean up cache and temporary files
echo "Cleaning caches and temporary files..."
rm -rf /n/home05/myrahmoun/.cache/torch_extensions/
rm -rf /tmp/myrahmoun/triton_cache_8b/
rm -rf /tmp/triton-*

# Clean up Python cache files
find /n/holylfs06/LABS/kempner_undergrads/Lab/myrahmoun/AgentsOpenRLHF/ -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
find /n/holylfs06/LABS/kempner_undergrads/Lab/myrahmoun/AgentsOpenRLHF/ -name "*.pyc" -delete 2>/dev/null || true

echo "After cleanup:"
quota /n/holylfs06/LABS/kempner_undergrads/Lab/myrahmoun/
echo "Cleanup complete! Proceeding with training..."

export NCCL_SOCKET_FAMILY=AF_INET
export NCCL_SOCKET_IFNAME=ib0
export NCCL_DEBUG=INFO
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TRITON_CACHE_DIR="/tmp/myrahmoun/triton_cache_8b"

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

# Add these BEFORE the srun command
export NCCL_SOCKET_FAMILY=AF_INET
export NCCL_SOCKET_IFNAME=ib0
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=2

# Force IPv4 for PyTorch distributed
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500
export WORLD_SIZE=$SLURM_NTASKS
export RANK=$SLURM_PROCID
export LOCAL_RANK=$SLURM_LOCALID

# Force PyTorch to use IPv4
export TORCH_DISTRIBUTED_DETAIL=DEBUG
export PYTHONUNBUFFERED=1

echo "Master address: $MASTER_ADDR"
echo "Master port: $MASTER_PORT"
echo "World size: $WORLD_SIZE"
echo "Rank: $RANK"
echo "Local rank: $LOCAL_RANK"

# Launch SFT Training with DeepSpeed
srun deepspeed --module openrlhf.cli.train_sft \
    --adam_offload \
    --max_len 512 \
    --dataset /n/holylfs06/LABS/kempner_undergrads/Lab/ellenma/openrlhf-proj/AgentsOpenRLHF/data/frozen_lake/frozen_lake_train.jsonl \
    --eval_dataset /n/holylfs06/LABS/kempner_undergrads/Lab/ellenma/openrlhf-proj/AgentsOpenRLHF/data/frozen_lake/frozen_lake_val.jsonl \
    --input_key prompt \
    --output_key response \
    --pretrain meta-llama/Llama-3.1-8B-Instruct \
    --save_path /n/holylfs06/LABS/kempner_undergrads/Lab/myrahmoun/AgentsOpenRLHF/openrlhf_artifacts/sft_llama8 \
    --apply_chat_template \
    --input_template "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{}\n<|im_end|>\n<|im_start|>assistant\n" \
    --train_batch_size 4 \
    --micro_train_batch_size 1 \
    --max_samples 1000 \
    --save_steps 250 \
    --logging_steps 1 \
    --eval_steps 50 \
    --disable_ds_ckpt \
    --save_hf_ckpt \
    --zero_stage 3 \
    --max_epochs 3 \
    --bf16 \
    --zpg 8 \
    --flash_attn \
    --learning_rate 5e-5 \
    --gradient_checkpointing \
    --use_wandb true