#!/bin/bash
#SBATCH --job-name=eval_sft_frozen_lake
#SBATCH --account=kempner_undergrads
#SBATCH --partition=kempner_h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --time=02:00:00
#SBATCH --output=/n/holylfs06/LABS/kempner_undergrads/Lab/myrahmoun/AgentsOpenRLHF/run_logs/sft_eval_%A.out
#SBATCH --error=/n/holylfs06/LABS/kempner_undergrads/Lab/myrahmoun/AgentsOpenRLHF/run_logs/sft_eval_%A.err
#SBATCH --export=ALL

# === Load modules and activate virtual environment ===
module load cuda/12.4.1-fasrc01
module load cudnn
module load gcc/12.2.0-fasrc01

source ~/.bashrc
source /n/holylfs06/LABS/kempner_undergrads/Lab/myrahmoun/AgentsOpenRLHF/.venv/bin/activate

# === Define paths ===
WORK_DIR="/n/holylfs06/LABS/kempner_undergrads/Lab/myrahmoun/AgentsOpenRLHF"
TEST_FILE="$WORK_DIR/data/frozen_lake/test_debug.jsonl"
SFT_MODEL="$WORK_DIR/openrlhf_artifacts/sft_llama8"
RESULTS_DIR="$WORK_DIR/sft_evaluation_results"
EVAL_SCRIPT_DIR="$WORK_DIR/scripts/llama_sft"

# Create necessary directories
mkdir -p "$RESULTS_DIR"
mkdir -p "$WORK_DIR/run_logs"

# === Move to project root for proper imports ===
cd "$WORK_DIR"

# === Print job info ===
echo "================================================================"
echo "SFT Llama 8B FrozenLake Evaluation Job"
echo "================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Start time: $(date)"
echo "Work directory: $WORK_DIR"
echo "SFT model: $SFT_MODEL"
echo "Test dataset: $TEST_FILE"
echo "Results directory: $RESULTS_DIR"
echo "================================================================"

# === Verify files exist ===
if [[ ! -f "$TEST_FILE" ]]; then
    echo "ERROR: Test file not found: $TEST_FILE"
    exit 1
fi

if [[ ! -d "$SFT_MODEL" ]]; then
    echo "ERROR: SFT model directory not found: $SFT_MODEL"
    exit 1
fi

# === Run SFT model evaluation ===
echo "Starting SFT Llama 8B evaluation..."
echo "Command: python $EVAL_SCRIPT_DIR/sft_main.py --model_path $SFT_MODEL --dataset $TEST_FILE --results_dir $RESULTS_DIR"

python "$EVAL_SCRIPT_DIR/sft_main.py" \
    --model_path "$SFT_MODEL" \
    --dataset "$TEST_FILE" \
    --results_dir "$RESULTS_DIR" \
    --temperature 0.1 \
    --max_tokens 150

EVAL_EXIT_CODE=$?

# === Check evaluation results ===
if [[ $EVAL_EXIT_CODE -eq 0 ]]; then
    echo "================================================================"
    echo "SFT evaluation completed successfully!"
    echo "================================================================"
    
    # List generated result files
    echo "Generated result files:"
    ls -la "$RESULTS_DIR"/sft_eval_*.json | tail -5
    
    # Show quick summary if results exist
    LATEST_RESULT=$(ls -t "$RESULTS_DIR"/sft_eval_*.json 2>/dev/null | head -1)
    if [[ -f "$LATEST_RESULT" ]]; then
        echo ""
        echo "Quick summary from latest results:"
        echo "Result file: $LATEST_RESULT"
        
        # Extract summary stats using jq if available, otherwise use grep
        if command -v jq &> /dev/null; then
            echo "Overall accuracy: $(jq -r '.summary.overall_accuracy' "$LATEST_RESULT" | awk '{printf "%.2f%%", $1*100}')"
            echo "Total problems: $(jq -r '.summary.total_problems' "$LATEST_RESULT")"
            echo "Grid sizes tested: $(jq -r '.summary.grid_sizes_tested | join(", ")' "$LATEST_RESULT")"
        else
            echo "Install jq for detailed summary parsing"
        fi
    fi
    
else
    echo "================================================================"
    echo "ERROR: SFT evaluation failed with exit code: $EVAL_EXIT_CODE"
    echo "================================================================"
    echo "Check the error log for details:"
    echo "tail /n/holylfs06/LABS/kempner_undergrads/Lab/myrahmoun/AgentsOpenRLHF/run_logs/sft_eval_$SLURM_JOB_ID.err"
fi

# === Cleanup and final info ===
echo ""
echo "Job completed at: $(date)"
echo "Total runtime: $SECONDS seconds"
echo "Results directory: $RESULTS_DIR"
echo "Log files:"
echo "  Output: /n/holylfs06/LABS/kempner_undergrads/Lab/myrahmoun/AgentsOpenRLHF/run_logs/sft_eval_$SLURM_JOB_ID.out"
echo "  Error:  /n/holylfs06/LABS/kempner_undergrads/Lab/myrahmoun/AgentsOpenRLHF/run_logs/sft_eval_$SLURM_JOB_ID.err"
echo "================================================================"

exit $EVAL_EXIT_CODE