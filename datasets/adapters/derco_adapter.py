"""
DERCo adapter (Quach et al., Scientific Data 2024).

Loads sentence text from the DERCo stimulus pickle files (article_*.pkl).
No raw EEG data is needed for the token-distribution analysis.

Dataset: DERCo: A Dataset for Human Behaviour in Reading Comprehension Using EEG
Paper:   Quach et al. (2024), Scientific Data. https://doi.org/10.1038/s41597-024-03915-8
Data:    https://osf.io/rkqbu/
GitHub:  https://github.com/Tayerquach/DERCo

──────────────────────────────────────────────────────────────────────────────
DOWNLOAD INSTRUCTIONS
──────────────────────────────────────────────────────────────────────────────
Go to:  https://osf.io/rkqbu/
Download the `article/` folder contents (the stimulus pickle files):
  article_0.pkl  … article_N.pkl

Place them at:
  <data_dir>/article/article_0.pkl
  <data_dir>/article/article_1.pkl
  ...

Disk space: The article/ folder is a few MB at most (text only, no EEG).
Raw EEG files are much larger — NOT needed for this analysis.

Alternatively, download Items.csv from the same OSF project for
sentence-level text.

Run with: --derco_dir <data_dir>
──────────────────────────────────────────────────────────────────────────────

DERCo stimulus structure (from GitHub):
  article_*.pkl  — Each pickle contains a list/dict of sentence strings or
                   word-level dicts for one article/passage.
  Items.csv      — May contain sentence-level stimulus text.

Since the exact internal structure can vary between dataset versions,
this adapter tries multiple common layouts and falls back gracefully.
"""

from __future__ import annotations

import os
import pickle
import csv
import re
from typing import Dict, List, Any, Optional


CITATION = "Quach et al. (2024), Scientific Data. https://doi.org/10.1038/s41597-024-03915-8"




def _extract_sentences_from_pickle(obj: Any) -> List[str]:
    """
    Try to extract a flat list of sentence strings from a DERCo pickle object.
    Handles several possible layouts:
      - list of str
      - list of dict with 'sentence', 'content', or 'text' key
      - dict with 'sentences', 'content', or similar key
      - nested: list of dicts each with a 'words' list
    """
    sentences = []

    if isinstance(obj, list):
        for item in obj:
            if isinstance(item, str) and len(item.strip()) > 5:
                sentences.append(item.strip())
            elif isinstance(item, dict):
                for key in ('sentence', 'content', 'text', 'Sentence', 'Content'):
                    if key in item and isinstance(item[key], str):
                        s = item[key].strip()
                        if len(s) > 5:
                            sentences.append(s)
                        break
                else:
                    # Try to reconstruct from word list
                    if 'words' in item and isinstance(item['words'], list):
                        words = [
                            w if isinstance(w, str) else w.get('word', w.get('content', ''))
                            for w in item['words']
                        ]
                        text = ' '.join(str(w) for w in words if w).strip()
                        if len(text) > 5:
                            sentences.append(text)
    elif isinstance(obj, dict):
        for key in ('sentences', 'Sentences', 'items', 'content', 'text'):
            if key in obj:
                return _extract_sentences_from_pickle(obj[key])
        # Try values directly
        for v in obj.values():
            sentences.extend(_extract_sentences_from_pickle(v))

    return sentences


def _load_items_csv(csv_path: str) -> List[str]:
    """Load sentences from Items.csv if present."""
    sentences = []
    with open(csv_path, 'r', encoding='utf-8-sig', errors='replace') as f:
        reader = csv.DictReader(f)
        for row in reader:
            for key in ('Sentence', 'sentence', 'Text', 'text', 'Content', 'content', 'Item', 'item'):
                if key in row and isinstance(row[key], str) and len(row[key].strip()) > 5:
                    sentences.append(row[key].strip())
                    break
    return sentences


def load(
    data_dir: str,
    n_subjects: int = 32,
) -> List[Dict[str, List[Any]]]:
    """
    Load DERCo dataset, returning a ZuCo-compatible dict list.

    Args:
        data_dir: Directory containing article/ folder or Items.csv
                  (downloaded from https://osf.io/rkqbu/)
        n_subjects: Number of subject keys to create in the dict.
                    DERCo has ~32 subjects.

    Returns:
        [ { 'sub-01': [{'content': sentence}, ...], ... } ]
    """
    sentences = []

    # ── Try Items.csv ──────────────────────────────────────────────────────────
    items_csv = os.path.join(data_dir, 'Items.csv')
    if os.path.exists(items_csv):
        print(f'[DERCo] Loading from Items.csv: {items_csv}')
        sentences = _load_items_csv(items_csv)
        if sentences:
            print(f'[DERCo] Loaded {len(sentences)} sentences from Items.csv')

    # ── Try article/*.pkl ──────────────────────────────────────────────────────
    if not sentences:
        article_dir = os.path.join(data_dir, 'article')
        if not os.path.isdir(article_dir):
            article_dir = data_dir  # maybe placed directly

        pkl_files = sorted([
            f for f in os.listdir(article_dir)
            if f.endswith('.pkl')
        ]) if os.path.isdir(article_dir) else []

        if pkl_files:
            print(f'[DERCo] Found {len(pkl_files)} article pickle(s) in {article_dir}')
            for fname in pkl_files:
                path = os.path.join(article_dir, fname)
                try:
                    with open(path, 'rb') as fh:
                        obj = pickle.load(fh)
                    extracted = _extract_sentences_from_pickle(obj)
                    sentences.extend(extracted)
                    print(f'[DERCo]   {fname}: {len(extracted)} sentences')
                except Exception as e:
                    print(f'[DERCo]   Warning: could not load {fname}: {e}')

    if not sentences:
        print(f'[DERCo] Error: No data found in {data_dir}.')
        return []

    # Deduplicate while preserving order
    seen = set()
    unique_sentences = []
    for s in sentences:
        if s not in seen:
            seen.add(s)
            unique_sentences.append(s)
    sentences = unique_sentences

    print(f'[DERCo] Total unique sentences: {len(sentences)}')

    sent_objects = [{'content': s} for s in sentences]
    subjects = [f'sub-{i:02d}' for i in range(1, n_subjects + 1)]
    dataset_dict = {subj: list(sent_objects) for subj in subjects}
    return [dataset_dict]


def dataset_name() -> str:
    return 'DERCo (Quach et al., 2024)'


DOWNLOAD_INSTRUCTIONS = """
DERCo Dataset Download Instructions
=====================================
Paper:  Quach et al. (2024), Scientific Data
        https://doi.org/10.1038/s41597-024-03915-8
Data:   https://osf.io/rkqbu/
GitHub: https://github.com/Tayerquach/DERCo

For the TOKEN ANALYSIS ONLY:
  1. Go to https://osf.io/rkqbu/
  2. Download EITHER:
     (a) Items.csv   — sentence-level stimulus text (~KB)
     (b) article/    — folder with article_*.pkl files (~MB)

  Place at (option a): <data_dir>/Items.csv
  Place at (option b): <data_dir>/article/article_0.pkl  etc.

  Run with: --derco_dir <data_dir>

Disk space: Items.csv is a few KB; article/ folder is a few MB.
Raw EEG files are NOT needed for this analysis.
"""
