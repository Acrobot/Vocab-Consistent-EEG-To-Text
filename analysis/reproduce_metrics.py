from pathlib import Path
import json
import confidence_intervals as ci
root = Path(__file__).resolve().parent.parent
expected = {r["condition"]: r for r in json.loads((root / "metrics/recomputed_metrics.json").read_text())}
for key, arrays in ci._load_arrays().items():
    for metric, value in zip(ci.METRICS, ci._score_vector(arrays)):
        assert abs(value - expected[key]["metrics"][metric]) < 1e-9, (key, metric)
print("Verified metric point estimates for all 16 conditions.")
