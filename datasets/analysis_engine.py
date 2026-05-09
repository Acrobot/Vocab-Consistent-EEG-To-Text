"""
Generic EEG dataset analysis engine.

Implements the token distribution analysis and stratified splitting independently
of any specific dataset format. Adapters convert their native format to the
ZuCo-compatible dict structure { subject: [ {content: str, ...}, ... ] }.
"""

from __future__ import annotations

import sys
import os
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Any

import numpy as np
from transformers import BartTokenizer

# We reuse the stratified splitting and analysis functions from the main data.py
# without modifying that file at all.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from data import create_stratified_split, analyze_token_distribution

# Default tokenizer (can be overridden per call)
DEFAULT_TOKENIZER_NAME = 'facebook/bart-large'
DEFAULT_MAX_LEN = 56


def load_tokenizer(model_name: str = DEFAULT_TOKENIZER_NAME) -> BartTokenizer:
    """Load a BART tokenizer (same as the main codebase uses)."""
    from transformers import BartTokenizer, PegasusTokenizer, T5Tokenizer
    if 'bart' in model_name.lower():
        return BartTokenizer.from_pretrained(model_name)
    elif 'pegasus' in model_name.lower():
        return PegasusTokenizer.from_pretrained(model_name)
    elif 't5' in model_name.lower():
        return T5Tokenizer.from_pretrained(model_name)
    else:
        # Default to BART
        return BartTokenizer.from_pretrained(DEFAULT_TOKENIZER_NAME)


def run_sequential_analysis(
    dataset_dicts: List[Dict[str, List[Any]]],
    tokenizer,
    subject_choice: str = 'ALL',
    max_len: int = DEFAULT_MAX_LEN,
) -> Dict:
    """
    Replicate the sequential (old 80/10/10) split analysis from analyse_dataset.py
    but without importing ZuCo_dataset — works on any dataset dict.

    Returns dict with keys: train_size, dev_size, test_size,
                             train_tokens, dev_tokens, test_tokens,
                             dev_unseen, test_unseen,
                             dev_unseen_pct, test_unseen_pct
    """
    import torch

    def get_subjects(dataset_dicts, subject_choice):
        if subject_choice == 'ALL':
            subjects = set()
            for d in dataset_dicts:
                subjects.update(d.keys())
            return sorted(subjects)
        return [subject_choice]

    subjects = get_subjects(dataset_dicts, subject_choice)

    all_samples = []  # list of (content, is_valid) grouped per dataset/subject
    # For sequential split we just collect all sentences in order per subject
    per_dataset_per_subject = []  # list of lists of content strings
    for d in dataset_dicts:
        for subj in subjects:
            if subj not in d:
                continue
            sentences = [
                s['content'] for s in d[subj]
                if s is not None and 'content' in s
            ]
            per_dataset_per_subject.append(sentences)

    def split_seq(sentences, train_ratio=0.8, dev_ratio=0.1):
        n = len(sentences)
        train_end = int(n * train_ratio)
        dev_end = train_end + int(n * dev_ratio)
        return sentences[:train_end], sentences[train_end:dev_end], sentences[dev_end:]

    train_sents, dev_sents, test_sents = [], [], []
    for sents in per_dataset_per_subject:
        tr, dv, te = split_seq(sents)
        train_sents.extend(tr)
        dev_sents.extend(dv)
        test_sents.extend(te)

    def tokenize_sents(sents, tokenizer, max_len):
        all_tokens = set()
        for s in sents:
            tok = tokenizer(s, padding='max_length', max_length=max_len,
                            truncation=True, return_tensors='pt',
                            return_attention_mask=True)
            ids = tok['input_ids'][0]
            mask = tok['attention_mask'][0]
            all_tokens.update(ids[mask == 1].tolist())
        return all_tokens

    train_vocab = tokenize_sents(train_sents, tokenizer, max_len)
    dev_vocab   = tokenize_sents(dev_sents,   tokenizer, max_len)
    test_vocab  = tokenize_sents(test_sents,  tokenizer, max_len)

    dev_unseen  = dev_vocab  - train_vocab
    test_unseen = test_vocab - train_vocab

    dev_unseen_pct  = 100 * len(dev_unseen)  / len(dev_vocab)  if dev_vocab  else 0.0
    test_unseen_pct = 100 * len(test_unseen) / len(test_vocab) if test_vocab else 0.0

    return {
        'train_size': len(train_sents),
        'dev_size':   len(dev_sents),
        'test_size':  len(test_sents),
        'train_tokens': len(train_vocab),
        'dev_tokens':   len(dev_vocab),
        'test_tokens':  len(test_vocab),
        'dev_unseen':       len(dev_unseen),
        'test_unseen':      len(test_unseen),
        'dev_unseen_pct':   dev_unseen_pct,
        'test_unseen_pct':  test_unseen_pct,
    }


def run_stratified_analysis(
    dataset_dicts: List[Dict[str, List[Any]]],
    tokenizer,
    subject_choice: str = 'ALL',
    max_len: int = DEFAULT_MAX_LEN,
    random_state: int = 42,
) -> Dict:
    """
    Run vocabulary-consistent stratified split analysis (the method from the paper).
    Wraps create_stratified_split + analyze_token_distribution from data.py.

    Returns dict with same keys as analyze_token_distribution() plus dataset sizes.
    """
    if subject_choice == 'ALL':
        subjects = set()
        for d in dataset_dicts:
            subjects.update(d.keys())
        subjects = sorted(subjects)
    else:
        subjects = [subject_choice]

    train_idx, dev_idx, test_idx = create_stratified_split(
        dataset_dicts, tokenizer, subjects,
        max_len=max_len, random_state=random_state
    )

    stats = analyze_token_distribution(
        dataset_dicts, tokenizer, subjects,
        train_idx, dev_idx, test_idx, max_len=max_len
    )

    # Count dataset sizes
    def count_samples(dataset_dicts, idx_dict, subjects):
        n = 0
        for dataset_idx, d in enumerate(dataset_dicts):
            for subj in subjects:
                if dataset_idx in idx_dict and subj in idx_dict[dataset_idx]:
                    n += len(idx_dict[dataset_idx][subj])
        return n

    stats['train_size'] = count_samples(dataset_dicts, train_idx, subjects)
    stats['dev_size']   = count_samples(dataset_dicts, dev_idx,   subjects)
    stats['test_size']  = count_samples(dataset_dicts, test_idx,  subjects)
    stats['split_indices'] = {
        'train': train_idx, 'dev': dev_idx, 'test': test_idx
    }
    return stats


def analyse_dataset(
    dataset_dicts: List[Dict[str, List[Any]]],
    dataset_name: str,
    tokenizer,
    subject_choice: str = 'ALL',
    max_len: int = DEFAULT_MAX_LEN,
    random_state: int = 42,
) -> Dict:
    """
    Full analysis: sequential split + stratified split + comparison.
    Prints human-readable output matching analyse_dataset.py's format.

    Returns dict with keys 'sequential' and 'stratified'.
    """
    SEP = '=' * 80
    sep = '-' * 80

    print(f'\n{SEP}')
    print(f'DATASET: {dataset_name}')
    print(SEP)

    # Count subjects and sentences
    total_subjects: set = set()
    total_sentences = 0
    for d in dataset_dicts:
        for subj, sents in d.items():
            total_subjects.add(subj)
            total_sentences += sum(1 for s in sents if s is not None)
    print(f'  Subjects:  {len(total_subjects)}')
    print(f'  Sentences: {total_sentences}')
    print(f'  Tokenizer: {tokenizer.name_or_path}')

    # ── 1. Sequential split ──────────────────────────────────────────────────
    print(f'\n{sep}')
    print('1. SEQUENTIAL SPLIT (Old Method — 80/10/10 by sentence order)')
    print(sep)

    seq = run_sequential_analysis(
        dataset_dicts, tokenizer, subject_choice, max_len
    )
    print(f'\n  Dataset sizes:')
    print(f'    Train: {seq["train_size"]} samples')
    print(f'    Dev:   {seq["dev_size"]} samples')
    print(f'    Test:  {seq["test_size"]} samples')
    print(f'\n  Unique tokens per split:')
    print(f'    Train: {seq["train_tokens"]} tokens')
    print(f'    Dev:   {seq["dev_tokens"]} tokens')
    print(f'    Test:  {seq["test_tokens"]} tokens')
    print(f'\n  Unseen tokens (not in training set):')
    print(f'    Dev:  {seq["dev_unseen"]} tokens ({seq["dev_unseen_pct"]:.2f}%)')
    print(f'    Test: {seq["test_unseen"]} tokens ({seq["test_unseen_pct"]:.2f}%)')

    # ── 2. Stratified split ──────────────────────────────────────────────────
    print(f'\n{sep}')
    print('2. STRATIFIED SPLIT (New Method — Vocabulary-Consistent Partitioning)')
    print(sep)

    strat = run_stratified_analysis(
        dataset_dicts, tokenizer, subject_choice, max_len, random_state
    )
    print(f'\n  Dataset sizes:')
    print(f'    Train: {strat["train_size"]} samples')
    print(f'    Dev:   {strat["dev_size"]} samples')
    print(f'    Test:  {strat["test_size"]} samples')

    # ── 3. Comparison ────────────────────────────────────────────────────────
    print(f'\n{SEP}')
    print(f'COMPARISON ({dataset_name}): Sequential vs Stratified')
    print(SEP)

    print(f'\n  Unseen tokens in DEV set:')
    print(f'    Sequential: {seq["dev_unseen"]}  ({seq["dev_unseen_pct"]:.2f}%)')
    print(f'    Stratified: {strat["dev_unseen"]}  ({strat["dev_unseen_pct"]:.2f}%)')
    delta_dev = seq["dev_unseen_pct"] - strat["dev_unseen_pct"]
    print(f'    Improvement: {delta_dev:.2f} percentage points')

    print(f'\n  Unseen tokens in TEST set:')
    print(f'    Sequential: {seq["test_unseen"]}  ({seq["test_unseen_pct"]:.2f}%)')
    print(f'    Stratified: {strat["test_unseen"]}  ({strat["test_unseen_pct"]:.2f}%)')
    delta_test = seq["test_unseen_pct"] - strat["test_unseen_pct"]
    print(f'    Improvement: {delta_test:.2f} percentage points')

    return {'sequential': seq, 'stratified': strat, 'name': dataset_name}
