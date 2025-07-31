#!/bin/bash
#SBATCH --job-name="sft_qw14"
#SBATCH --account=kempner_sham_lab
#SBATCH --output /n/holylfs06/LABS/sham_lab/Users/mkwun/AgentsOpenRLHF/run_logs/qwen14_%A.log
#SBATCH --error /n/holylfs06/LABS/sham_lab/Users/mkwun/AgentsOpenRLHF/run_logs/qwen14_error_%j.out  # Error file
#SBATCH --export=ALL
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1  
#SBATCH --gpus-per-node=4    
#SBATCH --cpus-per-task=24    
#SBATCH --mem=384GB           
#SBATCH --time=7:00:00 
#SBATCH --partition kempner_h100

module load cuda/12.4.1-fasrc01 cudnn gcc/12.2.0-fasrc01

source ~/.bashrc
conda activate openrlhf

# export NCCL_SOCKET_FAMILY=AF_INET
# export NCCL_SOCKET_IFNAME=ib0
# export NCCL_DEBUG=INFO
# export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
# export TRITON_CACHE_DIR="/tmp/ellenma/triton_cache_14b"
# export HF_HOME=/n/holylfs06/LABS/kempner_undergrads/Lab/ellenma/hf_cache


# Distributed Training Configuration
export GPUS_PER_NODE=4
export NNODES=$SLURM_NNODES
export NUM_PROCESSES=$(expr $NNODES \* $GPUS_PER_NODE)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True


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

# export HF_DATASETS_CACHE=/n/holylfs06/LABS/kempner_undergrads/Lab/ellenma/cache/hf_datasets
# export TMPDIR=/n/holylfs06/LABS/kempner_undergrads/Lab/ellenma/tmp
# export HOME_CACHE=/n/holylfs06/LABS/kempner_undergrads/Lab/ellenma/home_cache
# export TORCH_HOME=/n/holylfs06/LABS/kempner_undergrads/Lab/ellenma/torch_cache
# export XDG_CACHE_HOME=/n/holylfs06/LABS/kempner_undergrads/Lab/ellenma/xdg_cache

# deepspeed hanging prevention?
# export TORCH_EXTENSIONS_DIR=/tmp/ellenma/torch_extensions
# export CUDA_LAUNCH_BLOCKING=1
# export NCCL_P2P_DISABLE=1
# export CUDA_VISIBLE_DEVICES=0,1,2,3

deepspeed --module openrlhf.cli.train_sft \
   --max_len 512 \
   --dataset /n/holylfs06/LABS/sham_lab/Users/mkwun/AgentsOpenRLHF/data/frozen_lake/frozen_lake_train.jsonl \
   --eval_dataset /n/holylfs06/LABS/sham_lab/Users/mkwun/AgentsOpenRLHF/data/frozen_lake/frozen_lake_val.jsonl \
   --input_key prompt \
   --output_key response \
   --pretrain Qwen/Qwen3-14B \
   --save_path /n/holylfs06/LABS/sham_lab/Users/mkwun/AgentsOpenRLHF/openrlhf_artifacts/sft_qwen14 \
   --train_batch_size 8 \
   --micro_train_batch_size 1 \
   --max_samples 1000 \
   --save_steps 50 \
   --logging_steps 1 \
   --eval_steps 50 \
   --zero_stage 3 \
   --max_epochs 3 \
   --bf16 \
   --flash_attn \
   --learning_rate 5e-5 \
   --gradient_checkpointing \
   --use_wandb true \
   --adam_offload

   
   #    --packing_samples \
# try learning rates of 5e-5 and 5e-4

#    --apply_chat_template \
#    --input_template "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{}\n<|im_end|>\n<|im_start|>assistant\n" \
# --resume_path /n/holylfs06/LABS/kempner_undergrads/Lab/ellenma/openrlhf-proj/AgentsOpenRLHF/openrlhf_artifacts/sft_qwen14/
