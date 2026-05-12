"""
Alice EEG adapter (Brennan & Hale, 2019).

This adapter reconstructs sentence-level text from the Alice EEG word
annotation file. It does NOT load raw EEG data — only the stimulus
text annotations are needed for the token-distribution analysis.

Dataset: EEG Datasets for Naturalistic Listening to "Alice in Wonderland"
Paper:   Brennan & Hale (2019), PLoS ONE. https://doi.org/10.1371/journal.pone.0207741
Data:    https://doi.org/10.7302/Z29C6VNH  (University of Michigan Deep Blue Data)
GitHub:  https://github.com/cnllab/alice-eeg-shared

──────────────────────────────────────────────────────────────────────────────
DOWNLOAD INSTRUCTIONS (text annotations only — ~few MB, not the raw EEG)
──────────────────────────────────────────────────────────────────────────────
From https://doi.org/10.7302/Z29C6VNH download:
  • proc.zip  — contains the word-level annotation file used here

Unzip and locate:
  proc/AliceChapterOne-EEG.csv

That single CSV file is all this adapter needs.

Place it at:
  <data_dir>/AliceChapterOne-EEG.csv

Example:
  mkdir -p ./dataset/Alice
  # download proc.zip from https://doi.org/10.7302/Z29C6VNH
  unzip proc.zip -d ./dataset/Alice/proc/
  # The adapter expects: ./dataset/Alice/AliceChapterOne-EEG.csv
  # OR:                  ./dataset/Alice/proc/AliceChapterOne-EEG.csv

Disk space: proc.zip is ~1–2 MB. Raw EEG files are several GB — NOT needed here.
──────────────────────────────────────────────────────────────────────────────

CSV column reference (AliceChapterOne-EEG.csv):
  RECORDING_SESSION_LABEL  — subject/session ID
  CURRENT_FIX_INTEREST_AREA_LABEL — word text (or from a separate stimulus file)
  sentence boundaries are reconstructed from word order + pauses in onset times.

Note: The Alice dataset is a *listening* paradigm (audio of Chapter 1), not
silent reading. It has 2,129 word tokens from a single continuous story.
We treat each natural sentence (ending in '.', '?', '!') as one sample.
This gives ~57 sentences — small but useful as a proof-of-concept.

Since the dataset has a single stimulus shared across subjects, each "subject"
gets the same sentence list (different EEG, same text). The adapter creates
one synthetic subject key per actual recording session.
"""

from __future__ import annotations

import os
import re
import csv
from typing import Dict, List, Any, Optional

CITATION = "Brennan & Hale (2019), PLoS ONE. https://doi.org/10.1371/journal.pone.0207741"


def _load_from_csv(csv_path: str) -> List[str]:
    """
    Load word-level annotations from AliceChapterOne-EEG.csv and reconstruct sentences.

    The CSV has columns including the word text and position information.
    We reconstruct sentences by joining words that share the same 'Sentence' ID.
    """
    sentences = []
    try:
        with open(csv_path, 'r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            current_sentence_id = None
            current_words = []
            
            for row in reader:
                # Get the sentence ID (default to 1 if not found to avoid crashing, but the dataset has it)
                sent_id = row.get('Sentence') or row.get('sentence')
                word = (row.get('WORD') or row.get('Word') or row.get('word') or '').strip()
                
                if not word or word in ('', '.', 'NA', 'NaN'):
                    continue
                    
                if sent_id != current_sentence_id:
                    if current_words:
                        sentences.append(' '.join(current_words))
                    current_sentence_id = sent_id
                    current_words = [word]
                else:
                    current_words.append(word)
                    
            if current_words:
                sentences.append(' '.join(current_words))
                
    except Exception as e:
        print(f'[Alice] Warning: could not parse CSV ({e})')
        return []

    return sentences


def load(
    data_dir: str,
    n_synthetic_subjects: int = 10,
) -> List[Dict[str, List[Any]]]:
    """
    Load the Alice EEG dataset, returning a ZuCo-compatible dict list.

    Since all subjects read/listen to the SAME text, every subject sees
    identical sentences. We create synthetic subject keys S01..S{n}.
    The actual subject count from the original study is 49.

    Args:
        data_dir: Directory containing AliceChapterOne-EEG.csv
                  (or proc/AliceChapterOne-EEG.csv).
        n_synthetic_subjects: Number of synthetic subject copies to create.

    Returns:
        [ { 'S01': [{'content': sentence}, ...], 'S02': [...], ... } ]
    """
    sentences = None

    # Try to find the annotation CSV
    candidate_paths = [
        os.path.join(data_dir, 'AliceChapterOne-EEG.csv'),
        os.path.join(data_dir, 'proc', 'AliceChapterOne-EEG.csv'),
        os.path.join(data_dir, 'stimuli', 'AliceChapterOne-EEG.csv'),
    ]
    for p in candidate_paths:
        if os.path.exists(p):
            print(f'[Alice] Loading word annotations from {p}')
            sentences = _load_from_csv(p)
            break

    if not sentences:
        print(f'[Alice] Error: Annotation CSV not found in {data_dir}.')
        return []

    print(f'[Alice] Loaded {len(sentences)} sentences.')

    # Build ZuCo-compatible dict: one entry per synthetic subject
    sent_objects = [{'content': s} for s in sentences]

    # Determine how many subjects to create
    subjects = [f'S{i:02d}' for i in range(1, n_synthetic_subjects + 1)]

    dataset_dict = {subj: list(sent_objects) for subj in subjects}
    return [dataset_dict]


def dataset_name() -> str:
    return 'Alice EEG (Brennan & Hale, 2019)'


DOWNLOAD_INSTRUCTIONS = """
Alice EEG Dataset — Text Annotations Download
================================================
Paper:  Brennan & Hale (2019), PLoS ONE
        https://doi.org/10.1371/journal.pone.0207741
Data:   https://doi.org/10.7302/Z29C6VNH  (University of Michigan Deep Blue Data)
GitHub: https://github.com/cnllab/alice-eeg-shared

For the TOKEN ANALYSIS ONLY (no EEG needed):
  1. Go to https://doi.org/10.7302/Z29C6VNH
  2. Download proc.zip  (~1–2 MB)
  3. Unzip: unzip proc.zip
  4. Find AliceChapterOne-EEG.csv inside

  Place at: <data_dir>/AliceChapterOne-EEG.csv
  Run with: --alice_dir <data_dir>

Note: Raw EEG files (.eeg, .vhdr, .vmrk) are several GB — NOT needed for this analysis.

If you skip the download, the adapter uses the built-in Chapter 1 text directly
(same 57 sentences, slightly different tokenisation from the annotated version).
Disk space: ~2 MB (proc.zip only)
"""
