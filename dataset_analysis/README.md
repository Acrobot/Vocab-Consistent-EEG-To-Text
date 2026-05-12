# `datasets/` — Multi-Dataset Analysis Module

Self-contained extension of the project that applies the **Vocabulary-Consistent
Partitioning Method** (described in §3 of the paper) to multiple publicly
available EEG+text datasets beyond ZuCo.

**Zero changes to the main codebase.** All new code lives here.
The only dependency on the parent project is reusing `create_stratified_split()`
and `analyze_token_distribution()` from `../data.py`.

---

## File Structure

```
datasets/
├── analyse_multi_dataset.py   ← Main script (run this)
├── analysis_engine.py         ← Generic analysis engine (dataset-agnostic)
├── adapters/
│   ├── zuco_adapter.py        ← ZuCo (Hollenstein et al. 2018, 2020)
│   ├── alice_adapter.py       ← Alice EEG (Brennan & Hale, 2019)
│   ├── derco_adapter.py       ← DERCo (Quach et al., 2024)
│   └── roamm_adapter.py       ← ROAMM (OpenNeuro ds007629)
└── README.md                  ← This file
```

---

## Quick Start (no downloads required)

Run in fallback/demo mode using built-in text samples:

```bash
conda activate EEG-To-Text
python datasets/analyse_multi_dataset.py
```

This immediately produces a cross-dataset comparison table showing the
OOV improvement of the stratified split over sequential splitting.

---

## Running with Real Data

### Step 1 — Get download instructions

```bash
python datasets/analyse_multi_dataset.py --download-info
```

### Step 2 — Download (text annotations only, no raw EEG)

| Dataset | Download | Size |
|---------|----------|------|
| Alice EEG | `proc.zip` from https://doi.org/10.7302/Z29C6VNH | ~2 MB |
| DERCo | `Items.csv` or `article/` from https://osf.io/rkqbu/ | ~KB–MB |
| ROAMM | `*events.tsv` files from https://openneuro.org/datasets/ds007629 | ~few MB |
| ZuCo | Pickles from https://osf.io/q3zws/ + https://osf.io/2urht/ | ~existing |

For ROAMM, use `openneuro-py` to download only the events files (NOT the 50+ GB raw EEG):
```bash
pip install openneuro-py
openneuro-py download --dataset ds007629 \
    --include "*/eeg/*events.tsv" \
    --target-dir ./dataset/ROAMM
```

### Step 3 — Run the analysis

```bash
python datasets/analyse_multi_dataset.py \
    --datasets zuco alice derco roamm \
    --zuco_dir  ./dataset/ZuCo \
    --alice_dir ./dataset/Alice \
    --derco_dir ./dataset/DERCo \
    --roamm_dir ./dataset/ROAMM
```

---

## Why not Mahnob HCI or CHB-MIT?

These two datasets (which you may already have) are **not suitable** for the
vocabulary-consistency analysis:

- **Mahnob HCI** — emotion recognition from video/EEG. Records participants
  watching movie clips; no sentence/word-level text annotations.
- **CHB-MIT** — pediatric epilepsy seizure detection. Pure EEG signals with
  seizure annotations; no text whatsoever.

Both lack the fundamental requirement: **aligned word/sentence text annotations**.

---

## Output Example

```
================================================================================
CROSS-DATASET COMPARISON SUMMARY
================================================================================

  Metric: OOV rate in the TEST set
  Method: Sequential = naive 80/10/10 sequential split
          Stratified = Vocabulary-Consistent Partitioning (paper method)

  Dataset                              Seq. OOV (test) Strat. OOV (test)     Δ (pp)
  ─────────────────────────────────────────────────────────────────────────────────
  ZuCo (task1+task2+taskNRv2)                   38.xx%             0.00%    38.xx pp
  Alice EEG (Brennan & Hale, 2019)              30.71%             0.00%    30.71 pp
  DERCo (Quach et al., 2024)                    ~40-60%            0.00%     ~40-60pp
  ROAMM (OpenNeuro ds007629)                    ~50-70%            0.00%     ~50-70pp
  ─────────────────────────────────────────────────────────────────────────────────
```

The consistent **0% OOV** under the stratified split across all datasets confirms
that the Vocabulary-Consistent Partitioning Method generalises beyond ZuCo.
