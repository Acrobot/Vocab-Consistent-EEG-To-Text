#!/usr/bin/env python
"""W&B Sweep training wrapper for BrainTranslatorBaseline with contrastive loss.

Called directly by the sweep agent. Does NOT call wandb.init() itself —
instead passes sweep config as CLI args to train_decoding.py which handles
its own wandb initialization.
"""
import os
import subprocess
import sys

def main():
    # Parse sweep args (wandb agent passes them as key=value positional args)
    config = {}
    for arg in sys.argv[1:]:
        if '=' in arg:
            key, value = arg.split('=', 1)
            config[key] = value

    contrastive_lambda = config.get('contrastive_lambda', '0.1')
    dropout_rate = config.get('dropout_rate', '0.386')
    lr2 = config.get('learning_rate_step2', '0.000314')

    # Use a unique save path per sweep run
    run_id = os.environ.get('WANDB_RUN_ID', 'manual')
    save_path = f"./checkpoints/decoding/sweep_baseline/{run_id}"
    os.makedirs(f"{save_path}/best", exist_ok=True)
    os.makedirs(f"{save_path}/last", exist_ok=True)

    python_exe = "python"

    cmd = [
        python_exe, "-u", "train_decoding.py",
        "--model_name", "BrainTranslatorBaseline",
        "--task_name", "task1_task2_taskNRv2",
        "--one_step",  # Important: using one_step for the baseline!
        "--pretrained",
        "--not_load_step1_checkpoint",
        "--num_epoch_step1", "20",
        "--num_epoch_step2", "30",
        "--train_input", "EEG",
        "-lr1", "0.00002",
        "-lr2", lr2,
        "--dropout_rate", dropout_rate,
        "--contrastive_lambda", contrastive_lambda,
        "-b", "8",
        "-s", save_path,
        "--subjects", "ALL",
    ]

    print(f"Sweep trial:")
    print(f"  contrastive_lambda={contrastive_lambda}")
    print(f"  dropout_rate={dropout_rate}")
    print(f"  learning_rate_step2={lr2}")

    result = subprocess.run(cmd, cwd=".")
    sys.exit(result.returncode)

if __name__ == "__main__":
    main()
