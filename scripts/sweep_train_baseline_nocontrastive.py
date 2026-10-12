#!/usr/bin/env python
"""W&B Sweep training wrapper for the vocab-consistent baseline.

Sweeps dropout + learning rate for BrainTranslatorBaseline on the
vocabulary-consistent split, WITHOUT a contrastive penalty. This provides
a sweep directly on the architecture-and-split combination used in the paper,
replacing the transferred-from-ug5lbk5f hyperparameters.

After each trial completes, the `last/` checkpoint is removed to save disk
(~1.7 GB per trial); only `best/` is retained for evaluation.
"""
import os
import shutil
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    config = {}
    for arg in sys.argv[1:]:
        if '=' in arg:
            key, value = arg.split('=', 1)
            config[key] = value

    dropout_rate = config.get('dropout_rate', '0.386')
    lr2 = config.get('learning_rate_step2', '0.000314')

    run_id = os.environ.get('WANDB_RUN_ID', 'manual')
    save_path_rel = f"./checkpoints/decoding/sweep_baseline_nocontrastive/{run_id}"
    save_path_abs = os.path.join(REPO_ROOT, save_path_rel[2:])
    os.makedirs(os.path.join(save_path_abs, "best"), exist_ok=True)
    os.makedirs(os.path.join(save_path_abs, "last"), exist_ok=True)

    python_exe = sys.executable

    cmd = [
        python_exe, "-u", "train_decoding.py",
        "--model_name", "BrainTranslatorBaseline",
        "--task_name", "task1_task2_taskNRv2",
        "--one_step",
        "--pretrained",
        "--not_load_step1_checkpoint",
        "--num_epoch_step1", "20",
        "--num_epoch_step2", "30",
        "--train_input", "EEG",
        "-lr1", "0.00002",
        "-lr2", lr2,
        "--dropout_rate", dropout_rate,
        # No --contrastive_lambda flag; defaults to 0.0 (no contrastive penalty)
        "-b", "8",
        "-s", save_path_rel,
        "--subjects", "ALL",
    ]

    print("Sweep trial (vocab-consistent baseline, no contrastive):")
    print(f"  dropout_rate={dropout_rate}")
    print(f"  learning_rate_step2={lr2}")

    result = subprocess.run(cmd, cwd=REPO_ROOT)

    # Post-training cleanup: delete `last/` checkpoint to save ~1.7 GB/trial.
    # Only proceeds if training succeeded AND the path is well-formed.
    last_dir = os.path.join(save_path_abs, "last")
    if result.returncode == 0 and os.path.isdir(last_dir):
        if "sweep_baseline_nocontrastive" in last_dir and last_dir.rstrip("/").endswith("last"):
            print(f"[cleanup] Removing {last_dir} to free ~1.7 GB")
            shutil.rmtree(last_dir)
        else:
            print(f"[cleanup] WARNING: skipping deletion of {last_dir} (path safety check failed)")

    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
