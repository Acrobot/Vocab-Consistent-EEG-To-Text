"""
ROAMM adapter (Reading Observed At Mindless Moments, 2024).

Loads sentence text from the ROAMM BIDS dataset on OpenNeuro (ds007629).
No raw EEG data is needed for the token-distribution analysis.

Dataset: ROAMM — Reading Observed At Mindless Moments
Data:    https://openneuro.org/datasets/ds007629
Format:  BIDS (EEG + eye-tracking, simultaneous naturalistic reading)
         44 participants, 50+ hours, sentence-level stimuli

──────────────────────────────────────────────────────────────────────────────
DOWNLOAD INSTRUCTIONS (stimulus text only — no EEG needed)
──────────────────────────────────────────────────────────────────────────────
The BIDS events.tsv files contain the stimulus text. You only need:

Option A — Use the OpenNeuro CLI (recommended, downloads just metadata):
  pip install openneuro-py
  openneuro-py download --dataset ds007629 --include "*events.tsv" \\
      --target-dir ./dataset/ROAMM

  This downloads only the events files (~MB), NOT the raw EEG (~50+ GB).

Option B — Manual download from the web:
  1. Go to https://openneuro.org/datasets/ds007629
  2. Navigate to any subject's eeg/ folder
  3. Download the *_events.tsv files (they contain sentence text)
  Place at: <data_dir>/sub-*/eeg/*_events.tsv

Option C — Use the dataset README / stimuli folder if present:
  Some BIDS datasets include a stimuli/ folder with text files.

Run with: --roamm_dir <data_dir>

Disk space: Only events.tsv files (~few MB total for all subjects).
            Raw EEG: ~50+ GB — do NOT download.
──────────────────────────────────────────────────────────────────────────────

BIDS events.tsv column reference (ROAMM):
  onset        — event onset in seconds
  duration     — event duration
  trial_type   — event type (e.g., 'sentence', 'word', 'fixation')
  stim_value   — the actual stimulus text (sentence or word)
  OR
  sentence     — sentence text
  word         — word text

We extract sentences from the trial_type == 'sentence' rows,
or reconstruct them from word-level events if needed.
"""

from __future__ import annotations

import os
import re
import csv
import glob
from typing import Dict, List, Any, Optional


CITATION = "ROAMM: Reading Observed At Mindless Moments. OpenNeuro ds007629."




def _parse_events_tsv(tsv_path: str) -> List[str]:
    """
    Extract sentence text from a BIDS events.tsv file.
    Tries multiple column naming conventions.
    """
    sentences = []
    try:
        with open(tsv_path, 'r', encoding='utf-8-sig', errors='replace') as f:
            reader = csv.DictReader(f, delimiter='\t')
            rows = list(reader)
    except Exception:
        return []

    if not rows:
        return []

    # Determine which column has sentence text
    # Common patterns in BIDS EEG reading datasets:
    sentence_col = None
    word_col = None
    type_col = None

    sample_row = rows[0]
    for col in sample_row:
        col_lower = col.lower()
        if col_lower in ('stim_value', 'stimulus', 'stim', 'sentence', 'content', 'text'):
            sentence_col = col
        if col_lower in ('word', 'word_text', 'wordtext'):
            word_col = col
        if col_lower in ('trial_type', 'event_type', 'type', 'condition'):
            type_col = col

    # Strategy 1: direct sentence column
    if sentence_col:
        for row in rows:
            val = row.get(sentence_col, '').strip()
            # Filter for sentence-level events if trial_type is present
            if type_col:
                event_type = row.get(type_col, '').lower()
                if 'word' in event_type and 'sentence' not in event_type:
                    continue  # skip word-level events
            if len(val) > 10 and not val.startswith('n/a'):
                sentences.append(val)

    # Strategy 2: reconstruct sentences from word events
    if not sentences and word_col:
        current_sentence_words = []
        for row in rows:
            word = row.get(word_col, '').strip()
            if not word or word.lower() in ('n/a', 'na', ''):
                if current_sentence_words:
                    s = ' '.join(current_sentence_words)
                    if len(s) > 10:
                        sentences.append(s)
                    current_sentence_words = []
                continue
            current_sentence_words.append(word)
            # Check for sentence boundary
            if word.endswith(('.', '?', '!')):
                if current_sentence_words:
                    s = ' '.join(current_sentence_words)
                    if len(s) > 10:
                        sentences.append(s)
                    current_sentence_words = []
        if current_sentence_words:
            s = ' '.join(current_sentence_words)
            if len(s) > 10:
                sentences.append(s)

    return sentences


def _find_events_files(data_dir: str) -> List[str]:
    """Find all events.tsv files in a BIDS directory tree."""
    patterns = [
        os.path.join(data_dir, '**', '*events.tsv'),
        os.path.join(data_dir, '*events.tsv'),
    ]
    files = []
    for pattern in patterns:
        files.extend(glob.glob(pattern, recursive=True))
    return sorted(set(files))


def _load_stimuli_txt(data_dir: str) -> List[str]:
    """Look for plain text stimulus files."""
    sentences = set()
    
    # Check for derivatives/stimuli/wiki_stories (ROAMM structure)
    stimuli_dirs = [
        os.path.join(data_dir, 'derivatives', 'stimuli', 'wiki_stories'),
        os.path.join(data_dir, 'stimuli')
    ]
    
    for stimuli_dir in stimuli_dirs:
        if os.path.isdir(stimuli_dir):
            for fname in sorted(os.listdir(stimuli_dir)):
                if fname.endswith('.csv'):
                    path = os.path.join(stimuli_dir, fname)
                    try:
                        with open(path, 'r', encoding='utf-8') as f:
                            reader = csv.DictReader(f)
                            for row in reader:
                                if 'sentence' in row:
                                    val = row['sentence'].strip()
                                    if len(val) > 10:
                                        sentences.add(val)
                    except Exception:
                        pass
                        
    return sorted(list(sentences))


def load(
    data_dir: str,
    max_subjects: Optional[int] = None,
) -> List[Dict[str, List[Any]]]:
    """
    Load ROAMM dataset from BIDS directory, returning a ZuCo-compatible dict list.

    Args:
        data_dir: Path to BIDS root directory (containing sub-*/ folders)
        max_subjects: Maximum number of subjects to load (None = all).
                      Each subject's events.tsv is parsed for stimulus text.

    Returns:
        [ { 'sub-01': [{'content': sentence}, ...], ... } ]
    """
    dataset_dict: Dict[str, List[Any]] = {}

    # ── Find and parse events.tsv files ──────────────────────────────────────
    events_files = _find_events_files(data_dir)

    if events_files:
        print(f'[ROAMM] Found {len(events_files)} events.tsv file(s).')

        # Group by subject
        subject_files: Dict[str, List[str]] = {}
        for fpath in events_files:
            # Extract subject from path: sub-XX
            parts = fpath.replace('\\', '/').split('/')
            subj = next((p for p in parts if p.startswith('sub-')), None)
            if subj:
                subject_files.setdefault(subj, []).append(fpath)
            else:
                subject_files.setdefault('sub-00', []).append(fpath)

        subjects = sorted(subject_files.keys())
        if max_subjects is not None:
            subjects = subjects[:max_subjects]

        for subj in subjects:
            sentences = []
            for fpath in subject_files[subj]:
                sentences.extend(_parse_events_tsv(fpath))
            # Deduplicate per subject (same sentence may appear in multiple runs)
            seen = set()
            unique = []
            for s in sentences:
                if s not in seen:
                    seen.add(s)
                    unique.append(s)
            if unique:
                dataset_dict[subj] = [{'content': s} for s in unique]
                print(f'[ROAMM]   {subj}: {len(unique)} sentences')

    # ── Try stimuli/ folder ───────────────────────────────────────────────────
    if not dataset_dict:
        stim_sentences = _load_stimuli_txt(data_dir)
        if stim_sentences:
            print(f'[ROAMM] Loaded {len(stim_sentences)} sentences from stimuli/ folder.')
            # Create synthetic subject structure
            n_subjects = max_subjects or 44
            subjects = [f'sub-{i:02d}' for i in range(1, n_subjects + 1)]
            sent_objects = [{'content': s} for s in stim_sentences]
            dataset_dict = {subj: list(sent_objects) for subj in subjects}

    # ── Fallback ───────────────────────────────────────────────────────────────
    if not dataset_dict:
        print(f'[ROAMM] Error: No BIDS data found in {data_dir}.')
        return []

    print(f'[ROAMM] Total subjects loaded: {len(dataset_dict)}')
    return [dataset_dict]


def dataset_name() -> str:
    return 'ROAMM (OpenNeuro ds007629)'


DOWNLOAD_INSTRUCTIONS = """
ROAMM Dataset Download Instructions (events.tsv only — no EEG!)
=================================================================
Dataset: https://openneuro.org/datasets/ds007629
Format:  BIDS

Option A — openneuro-py CLI (recommended):
  pip install openneuro-py
  openneuro-py download --dataset ds007629 \\
      --include "*/eeg/*events.tsv" \\
      --target-dir ./dataset/ROAMM

  This downloads ONLY the events files (a few MB total),
  NOT the raw EEG data (50+ GB).

Option B — Manual:
  1. Go to https://openneuro.org/datasets/ds007629
  2. Browse to sub-01/eeg/
  3. Download all *_events.tsv files for the subjects you want

  Place at: <data_dir>/sub-01/eeg/sub-01_task-reading_events.tsv etc.
  Run with: --roamm_dir <data_dir>

Disk space: ~few MB for all events.tsv files.
"""
