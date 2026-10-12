#!/bin/bash
# Generic Evaluation Script
# Usage: ./eval_generic.sh <checkpoint_file> <config_file> <results_dir> <score_dir> <log_dir> <subjects_mode> [subject_list...]
#
# subjects_mode: "ALL" for cohort evaluation, "EACH" to loop through subjects
# If "EACH", provide subject list as remaining arguments

set -e

PYTHON_EXE="${PYTHON_EXE:-python}"

# Parse arguments
CHECKPOINT_FILE="$1"
CONFIG_FILE="$2"
RESULTS_DIR="$3"
SCORE_DIR="$4"
LOG_DIR="$5"
SUBJECTS_MODE="$6"
shift 6

# Default subject list if not provided
if [ $# -eq 0 ]; then
    SUBJECTS=("YAC" "YAG" "YAK" "YDG" "YDR" "YFR" "YFS" "YHS" "YIS" "YLS" "YMD" "YMS" "YRH" "YRK" "YRP" "YSD" "YSL" "YTL" "ZAB" "ZDM" "ZDN" "ZGW" "ZJM" "ZJN" "ZJS" "ZKB" "ZKH" "ZKW" "ZMG" "ZPH")
else
    SUBJECTS=("$@")
fi

mkdir -p "$RESULTS_DIR"
mkdir -p "$SCORE_DIR"
mkdir -p "$LOG_DIR"

echo "================================================================"
echo "Checkpoint: $CHECKPOINT_FILE"
echo "Config: $CONFIG_FILE"
echo "Subjects Mode: $SUBJECTS_MODE"
echo "================================================================"

# Create a temporary config with the subject field modified
create_temp_config() {
    local SUBJ_VALUE="$1"
    local TEMP_CONFIG="./config/temp_eval_${SUBJ_VALUE}_$$.json"
    
    # Copy and replace subjects field
    sed "s/\"subjects\": \"[^\"]*\"/\"subjects\": \"$SUBJ_VALUE\"/" "$CONFIG_FILE" > "$TEMP_CONFIG"
    echo "$TEMP_CONFIG"
}

run_eval() {
    local TEMP_CONFIG="$1"
    local TEST_INPUT="$2"
    local LOG_PREFIX="$3"
    
    PYTHONUNBUFFERED=1 $PYTHON_EXE eval_decoding.py \
        --checkpoint_path "$CHECKPOINT_FILE" \
        --config_path "$TEMP_CONFIG" \
        --test_input "$TEST_INPUT" \
        --train_input "EEG" \
        -cuda "cuda:0" > "$LOG_DIR/${LOG_PREFIX}_${TEST_INPUT}.log" 2>&1
    
    return $?
}

move_results() {
    local RESULT_SUBDIR="$1"
    local SCORE_SUBDIR="$2"
    
    mkdir -p "$RESULTS_DIR/$RESULT_SUBDIR"
    mkdir -p "$SCORE_DIR/$SCORE_SUBDIR"
    mv results/*.txt "$RESULTS_DIR/$RESULT_SUBDIR/" 2>/dev/null || true
    mv score_results/*.txt "$SCORE_DIR/$SCORE_SUBDIR/" 2>/dev/null || true
}

if [ "$SUBJECTS_MODE" == "ALL" ]; then
    # Cohort evaluation - single run on all subjects
    TEMP_CONFIG=$(create_temp_config "ALL")
    for TEST_INPUT in "EEG" "noise"; do
        echo "Evaluating on $TEST_INPUT (ALL subjects)..."
        if run_eval "$TEMP_CONFIG" "$TEST_INPUT" "cohort"; then
            echo "SUCCESS: $TEST_INPUT"
            move_results "$TEST_INPUT" "$TEST_INPUT"
        else
            echo "FAILED: $TEST_INPUT. See $LOG_DIR/cohort_${TEST_INPUT}.log"
        fi
    done
    rm -f "$TEMP_CONFIG"
else
    # Per-subject evaluation
    for SUBJ in "${SUBJECTS[@]}"; do
        echo "Evaluating Subject: $SUBJ"
        TEMP_CONFIG=$(create_temp_config "$SUBJ")
        for TEST_INPUT in "EEG" "noise"; do
            if run_eval "$TEMP_CONFIG" "$TEST_INPUT" "$SUBJ"; then
                echo "SUCCESS: $SUBJ ($TEST_INPUT)"
                move_results "$SUBJ/$TEST_INPUT" "$SUBJ/$TEST_INPUT"
            else
                echo "FAILED: $SUBJ ($TEST_INPUT). See $LOG_DIR/${SUBJ}_${TEST_INPUT}.log"
            fi
        done
        rm -f "$TEMP_CONFIG"
    done
fi

echo "================================================================"
echo "Evaluation Complete."
