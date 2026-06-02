#!/bin/bash
#SBATCH --job-name=sp1500_dfl
#SBATCH --account=bfsl-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/sp1500_%j.out
#SBATCH --error=logs/sp1500_%j.err

module load anaconda3
conda activate sparse-tracking

cd $SLURM_SUBMIT_DIR/../..
mkdir -p logs

echo "=== S&P 1500 Experiment ==="
echo "Node: $(hostname), GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Start: $(date)"

python scripts/run_sp1500_experiment.py

echo "End: $(date)"
