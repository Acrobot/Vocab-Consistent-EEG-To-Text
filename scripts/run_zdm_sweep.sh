#!/bin/bash
# Run the WandB Sweep Agent for Subject ZDM (10 iterations)
set -e

PYTHON_EXE="${PYTHON_EXE:-python}"
WANDB_EXE="${WANDB_EXE:-wandb}"

echo "Starting WandB Sweep Agent for ZDM (Max 10 runs)"
"$WANDB_EXE" agent "${WANDB_SWEEP:?Set WANDB_SWEEP to entity/project/sweep_id}" --count "${N_TRIALS:-8}"

echo "Sweep completed."
