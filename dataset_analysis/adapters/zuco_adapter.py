"""
ZuCo adapter — wraps the existing ZuCo pickle loading so ZuCo can participate
in the cross-dataset comparison run by analyse_multi_dataset.py.

No changes to data.py are required.

Expected directory layout (same as the main codebase):
  <data_dir>/
    task1-SR/pickle/task1-SR-dataset.pickle
    task2-NR/pickle/task2-NR-dataset.pickle
    task3-TSR/pickle/task3-TSR-dataset.pickle
    task2-NR-2.0/pickle/task2-NR-2.0-dataset.pickle
"""

from __future__ import annotations

import os
import pickle
from typing import Dict, List, Optional, Any


TASK_FILES = {
    'task1': 'task1-SR/pickle/task1-SR-dataset.pickle',
    'task2': 'task2-NR/pickle/task2-NR-dataset.pickle',
    'task3': 'task3-TSR/pickle/task3-TSR-dataset.pickle',
    'taskNRv2': 'task2-NR-2.0/pickle/task2-NR-2.0-dataset.pickle',
}

CITATION = "Hollenstein et al. (2018, 2020) — https://osf.io/q3zws/ & https://osf.io/2urht/"


def load(
    data_dir: str,
    tasks: Optional[List[str]] = None,
) -> List[Dict[str, List[Any]]]:
    """
    Load ZuCo dataset dicts from pickle files.

    Args:
        data_dir: Path to the ZuCo root directory (containing task1-SR/, task2-NR/ etc.)
        tasks: List of task keys to load (default: ['task1', 'task2', 'taskNRv2']).
               Options: 'task1', 'task2', 'task3', 'taskNRv2'

    Returns:
        List of dicts { subject_id: [ sent_obj, ... ] }  (the native ZuCo format)
    """
    if tasks is None:
        tasks = ['task1', 'task2', 'taskNRv2']

    dataset_dicts = []
    for task_key in tasks:
        if task_key not in TASK_FILES:
            raise ValueError(f"Unknown ZuCo task: {task_key!r}. "
                             f"Choose from: {list(TASK_FILES.keys())}")
        path = os.path.join(data_dir, TASK_FILES[task_key])
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"ZuCo pickle not found: {path}\n"
                f"Download from https://osf.io/q3zws/ (v1) or https://osf.io/2urht/ (v2)"
            )
        print(f'[ZuCo] Loading {task_key} from {path}')
        with open(path, 'rb') as fh:
            dataset_dicts.append(pickle.load(fh))

    return dataset_dicts


def dataset_name(tasks: Optional[List[str]] = None) -> str:
    if tasks is None:
        tasks = ['task1', 'task2', 'taskNRv2']
    task_str = '+'.join(tasks)
    return f'ZuCo ({task_str})'


DOWNLOAD_INSTRUCTIONS = """
ZuCo Dataset Download Instructions
====================================
ZuCo 1.0 (task1-SR, task2-NR, task3-TSR):  https://osf.io/q3zws/
ZuCo 2.0 (task2-NR-2.0):                   https://osf.io/2urht/

Download the pickle files and place them in:
  ./dataset/ZuCo/task1-SR/pickle/task1-SR-dataset.pickle
  ./dataset/ZuCo/task2-NR/pickle/task2-NR-dataset.pickle
  ./dataset/ZuCo/task2-NR-2.0/pickle/task2-NR-2.0-dataset.pickle

Then run:
  python datasets/analyse_multi_dataset.py --datasets zuco \\
      --zuco_dir ./dataset/ZuCo
"""
