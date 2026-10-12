#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
echo "This scirpt construct .pickle files from .mat files from ZuCo dataset."
echo "Note: This process can take time, please be patient..."

python3 ./util/construct_dataset_mat_to_pickle_v1.py -t task1-SR
python3 ./util/construct_dataset_mat_to_pickle_v1.py -t task2-NR
python3 ./util/construct_dataset_mat_to_pickle_v1.py -t task3-TSR
python3 ./util/construct_dataset_mat_to_pickle_v2.py
