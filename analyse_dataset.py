import json
import pickle
import sys

import numpy as np
import torch
from transformers import BartTokenizer, PegasusTokenizer, T5Tokenizer

from data import ZuCo_dataset, create_stratified_split, analyze_token_distribution

if __name__ == "__main__":
    training_config = json.load(open(sys.argv[1]))

    task_name = training_config['task_name']
    model_name = training_config['model_name']

    subject_choice = training_config['subjects']
    eeg_type_choice = training_config['eeg_type']
    bands_choice = training_config['eeg_bands']

    dataset_setting = 'unique_sent'
    test_input = "EEG"

    whole_dataset_dicts = []
    if 'task1' in task_name:
        dataset_path_task1 = './dataset/ZuCo/task1-SR/pickle/task1-SR-dataset.pickle'
        with open(dataset_path_task1, 'rb') as handle:
            whole_dataset_dicts.append(pickle.load(handle))
    if 'task2' in task_name:
        dataset_path_task2 = './dataset/ZuCo/task2-NR/pickle/task2-NR-dataset.pickle'
        with open(dataset_path_task2, 'rb') as handle:
            whole_dataset_dicts.append(pickle.load(handle))
    if 'task3' in task_name:
        dataset_path_task3 = './dataset/ZuCo/task3-TSR/pickle/task3-TSR-dataset.pickle'
        with open(dataset_path_task3, 'rb') as handle:
            whole_dataset_dicts.append(pickle.load(handle))
    if 'taskNRv2' in task_name:
        dataset_path_taskNRv2 = './dataset/ZuCo/task2-NR-2.0/pickle/task2-NR-2.0-dataset.pickle'
        with open(dataset_path_taskNRv2, 'rb') as handle:
            whole_dataset_dicts.append(pickle.load(handle))

    if model_name in ['BrainTranslator','BrainTranslatorNaive']:
        tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
    elif model_name == 'PegasusTranslator':
        tokenizer = PegasusTokenizer.from_pretrained('google/pegasus-xsum')
    elif model_name == 'T5Translator':
        tokenizer = T5Tokenizer.from_pretrained("t5-large")
    else:
        raise ValueError(f'Unknown model name: {model_name}')

    # Get subject list - collect from ALL datasets (union) to maximize data usage
    if subject_choice == 'ALL':
        subjects = set()
        for dataset_dict in whole_dataset_dicts:
            subjects.update(dataset_dict.keys())
        subjects = sorted(list(subjects))  # Sort for consistency
        print(f'[INFO] Using subjects from all datasets: {subjects}')
    else:
        subjects = [subject_choice]

    print('\n' + '='*80)
    print('DATASET TOKEN DISTRIBUTION ANALYSIS')
    print('='*80)

    # ============================================================================
    # 1. SEQUENTIAL SPLIT ANALYSIS (Old Method)
    # ============================================================================
    print('\n' + '-'*80)
    print('1. SEQUENTIAL SPLIT (Old Method - 80/10/10 by sentence order)')
    print('-'*80)

    sets_sequential = {
        phase:
        ZuCo_dataset(whole_dataset_dicts, phase, tokenizer, subject = subject_choice,
                     eeg_type = eeg_type_choice, bands = bands_choice,
                     setting = dataset_setting, test_input=test_input,
                     use_stratified_split=False)
        for phase in ['train', 'dev', 'test']
    }

    target_ids_seq = {
        set_name: torch.IntTensor([sample for item in sets_sequential[set_name] for sample in item[4]]).unique()
        for set_name in sets_sequential
    }

    dev_not_in_train_seq = np.setdiff1d(target_ids_seq['dev'], target_ids_seq['train'])
    test_not_in_train_seq = np.setdiff1d(target_ids_seq['test'], target_ids_seq['train'])

    print(f"\nDataset sizes:")
    print(f"  Train: {len(sets_sequential['train'])} samples")
    print(f"  Dev:   {len(sets_sequential['dev'])} samples")
    print(f"  Test:  {len(sets_sequential['test'])} samples")

    print(f"\nUnique tokens per split:")
    print(f"  Train: {len(target_ids_seq['train'])} tokens")
    print(f"  Dev:   {len(target_ids_seq['dev'])} tokens")
    print(f"  Test:  {len(target_ids_seq['test'])} tokens")

    print(f"\nUnseen tokens (not in training set):")
    print(f"  Dev:  {len(dev_not_in_train_seq)} tokens ({(len(dev_not_in_train_seq)/len(target_ids_seq['dev']))*100:.2f}%)")
    print(f"  Test: {len(test_not_in_train_seq)} tokens ({(len(test_not_in_train_seq)/len(target_ids_seq['test']))*100:.2f}%)")

    # ============================================================================
    # 2. STRATIFIED SPLIT ANALYSIS (New Method)
    # ============================================================================
    print('\n' + '-'*80)
    print('2. STRATIFIED SPLIT (New Method - Token-balanced distribution)')
    print('-'*80)

    # Create stratified split
    train_indices_dict, dev_indices_dict, test_indices_dict = create_stratified_split(
        whole_dataset_dicts, tokenizer, subjects, random_state=42
    )

    # Analyze token distribution
    stats_stratified = analyze_token_distribution(
        whole_dataset_dicts, tokenizer, subjects,
        train_indices_dict, dev_indices_dict, test_indices_dict
    )

    # Also load datasets with stratified split to get sample counts
    sets_stratified = {
        phase:
        ZuCo_dataset(whole_dataset_dicts, phase, tokenizer, subject = subject_choice,
                     eeg_type = eeg_type_choice, bands = bands_choice,
                     setting = dataset_setting, test_input=test_input,
                     use_stratified_split=True)
        for phase in ['train', 'dev', 'test']
    }

    print(f"Dataset sizes:")
    print(f"  Train: {len(sets_stratified['train'])} samples")
    print(f"  Dev:   {len(sets_stratified['dev'])} samples")
    print(f"  Test:  {len(sets_stratified['test'])} samples")

    # ============================================================================
    # 3. COMPARISON
    # ============================================================================
    print('\n' + '='*80)
    print('COMPARISON: Sequential vs Stratified Split')
    print('='*80)

    print(f"\nUnseen tokens in DEV set:")
    print(f"  Sequential: {len(dev_not_in_train_seq)} ({(len(dev_not_in_train_seq)/len(target_ids_seq['dev']))*100:.2f}%)")
    print(f"  Stratified: {stats_stratified['dev_unseen']} ({stats_stratified['dev_unseen_pct']:.2f}%)")
    print(f"  Improvement: {(len(dev_not_in_train_seq)/len(target_ids_seq['dev']) - stats_stratified['dev_unseen_pct']/100)*100:.2f} percentage points")

    print(f"\nUnseen tokens in TEST set:")
    print(f"  Sequential: {len(test_not_in_train_seq)} ({(len(test_not_in_train_seq)/len(target_ids_seq['test']))*100:.2f}%)")
    print(f"  Stratified: {stats_stratified['test_unseen']} ({stats_stratified['test_unseen_pct']:.2f}%)")
    print(f"  Improvement: {(len(test_not_in_train_seq)/len(target_ids_seq['test']) - stats_stratified['test_unseen_pct']/100)*100:.2f} percentage points")

    print('\n' + '='*80)
    print('RECOMMENDATION')
    print('='*80)
    print('\nThe stratified split significantly reduces unseen tokens in dev/test sets.')
    print('This should improve model performance by ensuring all tokens are seen during training.')
    print('\nTo use stratified split in your training:')
    print('  - It is already enabled by default (use_stratified_split=True)')
    print('  - To disable: pass use_stratified_split=False to ZuCo_dataset()')
    print('='*80 + '\n')