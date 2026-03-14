#!/bin/bash
# Train Single Subject (ZDM) Model - V5 Best (Baseline+OneStep+Sweep)
# Hyperparameters from Sweep ihig0y4v
#   - Dropout Rate: 0.4022835940989354
#   - Learning Rate Step 2: 0.0004851820064384097
#   - Batch Size: 8

# Configuration
SUBJ="ZDM"
SAVE_PATH="./checkpoints/decoding/single_subject_ZDM"
LOG_DIR="./logs/single_subject_ZDM"
PYTHON_EXE="python"

mkdir -p "$LOG_DIR"
mkdir -p "$SAVE_PATH"

echo "Starting Single Subject Training - V5 Best Configuration"
echo "Subject: $SUBJ"
echo "Params: Model=BrainTranslatorBaseline, LR2=0.000485, Dropout=0.402, BatchSize=8"
echo "================================================================"

CUDA_VISIBLE_DEVICES=0 $PYTHON_EXE -u train_decoding.py \
    --model_name BrainTranslatorBaseline \
    --task_name task1_task2_taskNRv2 \
    --one_step \
    --pretrained \
    --not_load_step1_checkpoint \
    --num_epoch_step1 20 \
    --num_epoch_step2 30 \
    --train_input EEG \
    --learning_rate_step1 2e-05 \
    --learning_rate_step2 0.0004851820064384097 \
    --dropout_rate 0.4022835940989354 \
    -b 8 \
    -s "$SAVE_PATH" \
    --subjects "$SUBJ" > "${LOG_DIR}/${SUBJ}_train.log" 2>&1

if [ $? -eq 0 ]; then
    echo "SUCCESS: $SUBJ"
else
    echo "FAILURE: $SUBJ. Check ${LOG_DIR}/${SUBJ}_train.log"
fi
    
echo "================================================================"
echo "Single Subject Training Complete."
