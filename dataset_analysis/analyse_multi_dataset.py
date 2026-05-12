#!/usr/bin/env python3
"""
analyse_multi_dataset.py — Multi-dataset token distribution analysis.

Applies the Vocabulary-Consistent Partitioning Method (described in the paper)
to multiple publicly available EEG+text datasets and produces a cross-dataset
comparison table showing OOV improvements.

Self-contained: lives entirely in datasets/, zero changes to the main codebase.
Reuses create_stratified_split() and analyze_token_distribution() from data.py.

Usage
-----
# Run on all datasets in fallback/demo mode (no data download required):
  python datasets/analyse_multi_dataset.py

# Run on specific datasets with actual data:
  python datasets/analyse_multi_dataset.py \\
      --datasets zuco alice derco roamm \\
      --zuco_dir  ./dataset/ZuCo \\
      --alice_dir ./dataset/Alice \\
      --derco_dir ./dataset/DERCo \\
      --roamm_dir ./dataset/ROAMM

# Run only on ZuCo + Alice (to verify against existing results):
  python datasets/analyse_multi_dataset.py \\
      --datasets zuco alice \\
      --zuco_dir  ./dataset/ZuCo \\
      --alice_dir ./dataset/Alice

# Show download instructions for all datasets:
  python datasets/analyse_multi_dataset.py --download-info

Supported datasets
------------------
  zuco   — ZuCo (Hollenstein et al. 2018, 2020) — same as analyse_dataset.py
  alice  — Alice EEG (Brennan & Hale, 2019)     — listening paradigm
  derco  — DERCo (Quach et al. 2024)             — reading paradigm
  roamm  — ROAMM (OpenNeuro ds007629, 2024)      — reading + mind-wandering

Note on existing datasets (Mahnob HCI, CHB-MIT)
------------------------------------------------
Mahnob HCI (emotion recognition from video/EEG) and CHB-MIT (epilepsy seizure
detection) do NOT contain text or word-level annotations, so they cannot be
used for the vocabulary-consistency analysis. The three datasets above are the
most suitable publicly available alternatives to ZuCo.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Optional, Any

# ── Path setup ─────────────────────────────────────────────────────────────────
# Allow imports from both the datasets/ package and the project root (for data.py)
_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(_here)
for p in [_here, _root]:
    if p not in sys.path:
        sys.path.insert(0, p)

from analysis_engine import load_tokenizer, analyse_dataset
from adapters import alice_adapter, derco_adapter, roamm_adapter, zuco_adapter


# ── Dataset registry ───────────────────────────────────────────────────────────
SUPPORTED_DATASETS = ['zuco', 'alice', 'derco', 'roamm']


def print_download_info():
    """Print download instructions for all supported datasets."""
    sep = '─' * 72
    for adapter, name in [
        (zuco_adapter,  'ZuCo'),
        (alice_adapter, 'Alice EEG'),
        (derco_adapter, 'DERCo'),
        (roamm_adapter, 'ROAMM'),
    ]:
        print(f'\n{sep}')
        print(getattr(adapter, 'DOWNLOAD_INSTRUCTIONS', f'No instructions for {name}.'))
    print(f'{sep}\n')


def load_zuco(zuco_dir: str, tasks: Optional[List[str]] = None) -> tuple:
    """Load ZuCo and return (dataset_dicts, name)."""
    if zuco_dir is None:
        raise ValueError('--zuco_dir is required for the zuco dataset.')
    dicts = zuco_adapter.load(zuco_dir, tasks=tasks)
    return dicts, zuco_adapter.dataset_name(tasks)


def load_alice(alice_dir: str = '', n_subjects: int = 49) -> tuple:
    """Load Alice EEG and return (dataset_dicts, name)."""
    dicts = alice_adapter.load(alice_dir or '', n_synthetic_subjects=n_subjects)
    return dicts, alice_adapter.dataset_name()


def load_derco(derco_dir: str = '', n_subjects: int = 32) -> tuple:
    """Load DERCo and return (dataset_dicts, name)."""
    dicts = derco_adapter.load(derco_dir or '', n_subjects=n_subjects)
    return dicts, derco_adapter.dataset_name()


def load_roamm(roamm_dir: str = '', max_subjects: Optional[int] = None) -> tuple:
    """Load ROAMM and return (dataset_dicts, name)."""
    dicts = roamm_adapter.load(roamm_dir or '', max_subjects=max_subjects)
    return dicts, roamm_adapter.dataset_name()


def print_cross_dataset_table(results: List[Dict]):
    """Print a formatted cross-dataset comparison table."""
    SEP = '=' * 80
    print(f'\n{SEP}')
    print('CROSS-DATASET COMPARISON SUMMARY')
    print(SEP)
    print(f'\n  Metric: OOV (Out-of-Vocabulary) rate in the TEST set')
    print(f'  Method: Sequential = naive 80/10/10 sequential split')
    print(f'          Stratified = Vocabulary-Consistent Partitioning (paper method)')
    print()

    # Header
    col1 = 35  # dataset name
    col2 = 16  # sequential OOV
    col3 = 16  # stratified OOV
    col4 = 14  # improvement

    header = (f'{"Dataset":<{col1}} {"Seq. OOV (test)":>{col2}} '
              f'{"Strat. OOV (test)":>{col3}} {"Δ (pp)":>{col4}}')
    divider = '-' * len(header)
    print(f'  {header}')
    print(f'  {divider}')

    for r in results:
        name = r['name']
        seq_pct  = r['sequential']['test_unseen_pct']
        strat_pct = r['stratified']['test_unseen_pct']
        delta = seq_pct - strat_pct
        print(f'  {name:<{col1}} {seq_pct:>{col2 - 1}.2f}%  '
              f'{strat_pct:>{col3 - 1}.2f}%  {delta:>{col4 - 2}.2f} pp')

    print(f'  {divider}')
    print()
    print('  pp = percentage points improvement of stratified over sequential.')
    print('  A Strat. OOV of ~0% confirms the method generalises across datasets.')
    print(f'\n{SEP}\n')


def main():
    parser = argparse.ArgumentParser(
        description='Multi-dataset token distribution analysis for EEG-to-text.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        '--datasets', nargs='+', default=['alice', 'derco', 'roamm'],
        choices=SUPPORTED_DATASETS,
        metavar='DATASET',
        help=(f'Datasets to analyse. Options: {", ".join(SUPPORTED_DATASETS)}. '
              f'Default: alice derco roamm  '
              f'(zuco requires --zuco_dir pointing to existing ZuCo pickles)'),
    )
    parser.add_argument(
        '--zuco_dir', type=str, default=None,
        help='Path to ZuCo root directory (containing task1-SR/, task2-NR/ etc.)',
    )
    parser.add_argument(
        '--zuco_tasks', nargs='+', default=None,
        choices=['task1', 'task2', 'task3', 'taskNRv2'],
        help='ZuCo tasks to load (default: task1 task2 taskNRv2)',
    )
    parser.add_argument(
        '--alice_dir', type=str, default='',
        help='Path to Alice EEG directory (containing AliceChapterOne-EEG.csv). '
             'If not provided, uses built-in Chapter 1 text.',
    )
    parser.add_argument(
        '--alice_subjects', type=int, default=49,
        help='Number of synthetic subject copies for Alice (default: 49, actual study N).',
    )
    parser.add_argument(
        '--derco_dir', type=str, default='',
        help='Path to DERCo directory (containing Items.csv or article/ folder). '
             'If not provided, uses built-in fallback passages.',
    )
    parser.add_argument(
        '--derco_subjects', type=int, default=32,
        help='Number of subjects for DERCo (default: 32).',
    )
    parser.add_argument(
        '--roamm_dir', type=str, default='',
        help='Path to ROAMM BIDS directory (containing sub-*/ folders). '
             'If not provided, uses built-in fallback passages.',
    )
    parser.add_argument(
        '--roamm_max_subjects', type=int, default=None,
        help='Maximum subjects to load from ROAMM (default: all).',
    )
    parser.add_argument(
        '--tokenizer', type=str, default='facebook/bart-large',
        help='Tokenizer to use (default: facebook/bart-large).',
    )
    parser.add_argument(
        '--max_len', type=int, default=56,
        help='Maximum sequence length for tokenization (default: 56).',
    )
    parser.add_argument(
        '--random_state', type=int, default=42,
        help='Random seed for reproducibility (default: 42).',
    )
    parser.add_argument(
        '--download-info', action='store_true',
        help='Print download instructions for all supported datasets and exit.',
    )

    args = parser.parse_args()

    if args.download_info:
        print_download_info()
        return

    print('\n' + '=' * 80)
    print('MULTI-DATASET TOKEN DISTRIBUTION ANALYSIS')
    print('Vocabulary-Consistent Partitioning Method Validation')
    print('=' * 80)

    # ── Warn about unsuitable existing datasets ──────────────────────────────
    print('\n[INFO] Note: Mahnob HCI and CHB-MIT datasets do NOT contain text')
    print('       annotations and cannot be used for vocabulary analysis.')
    print('       Targeting: Alice EEG, DERCo, ROAMM (and optionally ZuCo).\n')

    # ── Load tokenizer ────────────────────────────────────────────────────────
    print(f'[INFO] Loading tokenizer: {args.tokenizer}')
    tokenizer = load_tokenizer(args.tokenizer)

    # ── Run analysis per dataset ──────────────────────────────────────────────
    all_results = []

    for ds_name in args.datasets:
        try:
            if ds_name == 'zuco':
                dataset_dicts, name = load_zuco(args.zuco_dir, args.zuco_tasks)
            elif ds_name == 'alice':
                dataset_dicts, name = load_alice(args.alice_dir, args.alice_subjects)
            elif ds_name == 'derco':
                dataset_dicts, name = load_derco(args.derco_dir, args.derco_subjects)
            elif ds_name == 'roamm':
                dataset_dicts, name = load_roamm(args.roamm_dir, args.roamm_max_subjects)
            else:
                print(f'[WARN] Unknown dataset: {ds_name}, skipping.')
                continue

            result = analyse_dataset(
                dataset_dicts=dataset_dicts,
                dataset_name=name,
                tokenizer=tokenizer,
                subject_choice='ALL',
                max_len=args.max_len,
                random_state=args.random_state,
            )
            all_results.append(result)

        except FileNotFoundError as e:
            print(f'\n[ERROR] {ds_name}: {e}')
            print(f'[INFO]  Run with --download-info to see download instructions.')
        except Exception as e:
            print(f'\n[ERROR] Failed to process {ds_name}: {e}')
            import traceback
            traceback.print_exc()

    # ── Cross-dataset summary table ───────────────────────────────────────────
    if len(all_results) > 1:
        print_cross_dataset_table(all_results)
    elif len(all_results) == 1:
        print('\n[INFO] Only one dataset completed. Run with multiple --datasets for comparison.')

    print('[INFO] Analysis complete.')


if __name__ == '__main__':
    main()
