#!/bin/bash
#SBATCH --job-name=weather_dfl
#SBATCH --account=bfsl-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/weather_%j.out
#SBATCH --error=logs/weather_%j.err

module load anaconda3
conda activate sparse-tracking

cd $SLURM_SUBMIT_DIR/../..
mkdir -p logs

echo "=== Weather Station Experiment ==="
echo "Node: $(hostname), GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Start: $(date)"

python scripts/run_weather_stations.py

echo "End: $(date)"
