"""Evaluate a paper checkpoint with its recovered configuration and split."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["contrastive", "vocab", "literature", "zdm"], required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input", choices=["EEG", "noise"], default="EEG")
    parser.add_argument("--dry-run", action="store_true", help="Prepare config and print the command without inference")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    config = json.loads((root / "configs" / f"{args.model}.json").read_text())
    split = root / "splits" / f"{args.model}.json"
    if split.exists():
        config["split_indices"] = json.loads(split.read_text())
    config["use_sequential_split"] = args.model == "literature"
    output = root / "results" / "paper_eval" / args.model / args.input
    output.mkdir(parents=True, exist_ok=True)
    config_file = output / "evaluation_config.json"
    config_file.write_text(json.dumps(config, indent=2) + "\n")
    cmd = [sys.executable, "eval_decoding.py", "--checkpoint_path", str(args.checkpoint.resolve()),
           "--config_path", str(config_file), "--test_input", args.input, "--train_input", "EEG",
           "--results_path", str(output / "predictions.txt"), "--score_results_path", str(output / "scores.txt")]
    if args.dry_run:
        print(" ".join(cmd))
        return
    if not args.checkpoint.is_file():
        parser.error("Checkpoint file does not exist")
    subprocess.run(cmd, cwd=root, check=True)

if __name__ == "__main__":
    main()
