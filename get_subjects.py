import sys
import os
import pickle
# Add current dir to path
sys.path.append(os.getcwd())

# Import to ensure classes are available if needed, though usually standard dicts don't need it.
# But sometimes pickle dependencies matter.
try:
    from data import ZuCo_dataset
except ImportError:
    print("Could not import ZuCo_dataset, trying without it.")

paths = [
    './dataset/ZuCo/task1-SR/pickle/task1-SR-dataset.pickle',
    './dataset/ZuCo/task2-NR/pickle/task2-NR-dataset.pickle',
    './dataset/ZuCo/task2-NR-2.0/pickle/task2-NR-2.0-dataset.pickle'
]

subjects = set()
for p in paths:
    if os.path.exists(p):
        try:
            with open(p, 'rb') as f:
                data = pickle.load(f)
                subjects.update(data.keys())
        except Exception as e:
            print(f"Error loading {p}: {e}")
    else:
        print(f"Warning: {p} not found")

print("SUBJECTS:", sorted(list(subjects)))
