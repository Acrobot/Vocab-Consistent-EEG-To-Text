#!/bin/bash
# Evaluate Single Subject (ZDM) Model
set -e

SUBJ="ZDM"
CHECKPOINT="./checkpoints/decoding/sweep_zdm/slwo549q/best/task1_task2_taskNRv2_finetune_BrainTranslatorBaseline_skipstep1_b8_20_30_2e-05_5.061295065821606e-05_unique_sent_EEG.pt"
CONFIG="config/decoding/task1_task2_taskNRv2_finetune_BrainTranslatorBaseline_skipstep1_b8_20_30_2e-05_5.061295065821606e-05_unique_sent_EEG.json"
RESULTS_DIR="results/single_subject_ZDM"
SCORE_DIR="score_results/single_subject_ZDM"
LOG_DIR="logs/eval/single_subject_ZDM"

echo "Evaluating Single Subject ZDM"
./scripts/eval_generic.sh "$CHECKPOINT" "$CONFIG" "$RESULTS_DIR" "$SCORE_DIR" "$LOG_DIR" "EACH" "$SUBJ"
echo "Done."
