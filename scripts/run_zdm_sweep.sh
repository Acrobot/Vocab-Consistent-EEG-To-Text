#!/bin/bash
# Run the WandB Sweep Agent for Subject ZDM (10 iterations)
set -e

PYTHON_EXE="python"
WANDB_EXE="wandb"

echo "Starting WandB Sweep Agent for ZDM (Max 10 runs)"
$WANDB_EXE agent YOUR_WANDB_ENTITY/YOUR_WANDB_PROJECT/YOUR_SWEEP_ID --count 10

echo "Sweep completed."
