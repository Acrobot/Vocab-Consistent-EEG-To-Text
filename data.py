import os
import numpy as np
import torch
import pickle
from torch.utils.data import Dataset, DataLoader
import json
import matplotlib.pyplot as plt
from glob import glob
from transformers import BartTokenizer, BertTokenizer
from tqdm import tqdm
from fuzzy_match import match
from fuzzy_match import algorithims
from transformers import T5Tokenizer
from collections import defaultdict, Counter
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

# Constants
DEFAULT_MAX_LEN = 56  # Maximum sequence length for tokenization

# macro
#ZUCO_SENTIMENT_LABELS = json.load(open('./dataset/ZuCo/task1-SR/sentiment_labels/sentiment_labels.json'))
#SST_SENTIMENT_LABELS = json.load(open('./dataset/stanfordsentiment/ternary_dataset.json'))

def normalize_1d(input_tensor):
    # normalize a 1d tensor
    mean = torch.mean(input_tensor)
    std = torch.std(input_tensor)
    input_tensor = (input_tensor - mean)/std
    return input_tensor 

def get_input_sample(sent_obj, tokenizer, eeg_type = 'GD', bands = ['_t1','_t2','_a1','_a2','_b1','_b2','_g1','_g2'], max_len = DEFAULT_MAX_LEN, add_CLS_token = False, test_input="noise"):
    
    def get_word_embedding_eeg_tensor(word_obj, eeg_type, bands):
        frequency_features = []

        for band in bands:
            frequency_features.append(word_obj['word_level_EEG'][eeg_type][eeg_type+band])
        word_eeg_embedding = np.concatenate(frequency_features)

        if len(word_eeg_embedding) != 105*len(bands):
            # print(f'expect word eeg embedding dim to be {105*len(bands)}, but got {len(word_eeg_embedding)}, return None')
            return None
        
        # assert len(word_eeg_embedding) == 105*len(bands)
        return_tensor = torch.from_numpy(word_eeg_embedding)
        return normalize_1d(return_tensor)

    def get_sent_eeg(sent_obj, bands):
        sent_eeg_features = []

        for band in bands:
            key = 'mean'+band
            sent_eeg_features.append(sent_obj['sentence_level_EEG'][key])

        sent_eeg_embedding = np.concatenate(sent_eeg_features)
        assert len(sent_eeg_embedding) == 105*len(bands)
        return_tensor = torch.from_numpy(sent_eeg_embedding)
        return normalize_1d(return_tensor)

    if sent_obj is None:
        # print(f'  - skip bad sentence')   
        return None

    input_sample = {}
    # get target label
    target_string = sent_obj['content']
    target_tokenized = tokenizer(target_string, padding='max_length', max_length=max_len, truncation=True, return_tensors='pt', return_attention_mask = True)
    input_sample['target_ids'] = target_tokenized['input_ids'][0]
    
    # get sentence level EEG features
    sent_level_eeg_tensor = get_sent_eeg(sent_obj, bands)
    # try:
    #     sent_level_eeg_tensor = torch.from_numpy(sent_obj['sentence_level_EEG']) # This gives a dictionary
    # except:
    #     return None
    
    if torch.isnan(sent_level_eeg_tensor).any():
        # print('[NaN sent level eeg]: ', target_string)
        return None
    # if sent_level_eeg_tensor.shape[1] < 30:
    #     return None
    
    input_sample['sent_level_EEG'] = sent_level_eeg_tensor
    #input_sample['sent_level_EEG'] = torch.randn(sent_level_eeg_tensor.size()) # random input code
    #print("NOISE:", input_sample['sent_level_EEG'])

    # get sentiment label
    # handle some wierd case
    if 'emp11111ty' in target_string:
        target_string = target_string.replace('emp11111ty','empty')
    if 'film.1' in target_string:
        target_string = target_string.replace('film.1','film.')
    
    #if target_string in ZUCO_SENTIMENT_LABELS:
    #    input_sample['sentiment_label'] = torch.tensor(ZUCO_SENTIMENT_LABELS[target_string]+1) # 0:Negative, 1:Neutral, 2:Positive
    #else:
    #    input_sample['sentiment_label'] = torch.tensor(-100) # dummy value
    input_sample['sentiment_label'] = torch.tensor(-100) # dummy value

    # get input embeddings
    word_embeddings = []

    """add CLS token embedding at the front"""
    if add_CLS_token:
        word_embeddings.append(torch.ones(105*len(bands)))

    for word in sent_obj['word']:
        # add each word's EEG embedding as Tensors
        word_level_eeg_tensor = get_word_embedding_eeg_tensor(word, eeg_type, bands = bands)
        # check none, for v2 dataset
        if word_level_eeg_tensor is None:
            return None
        # check nan:
        if torch.isnan(word_level_eeg_tensor).any():
            # print()
            # print('[NaN ERROR] problem sent:',sent_obj['content'])
            # print('[NaN ERROR] problem word:',word['content'])
            # print('[NaN ERROR] problem word feature:',word_level_eeg_tensor)
            # print()
            return None
        
        word_embeddings.append(word_level_eeg_tensor)

    # pad to max_len
    while len(word_embeddings) < max_len:
        word_embeddings.append(torch.zeros(105*len(bands)))

    if test_input=='noise':
        rand_eeg= torch.randn(torch.stack(word_embeddings).size())
        input_sample['input_embeddings'] = rand_eeg # max_len * (105*num_bands)
        # print("rand_eeg:", rand_eeg)
        # print("input_embeddings:", input_sample['input_embeddings'].shape)

    else:
        input_sample['input_embeddings'] = torch.stack(word_embeddings) # max_len * (105*num_bands)
        # print("EEG", input_sample['input_embeddings'])
    
    # mask out padding tokens
    input_sample['input_attn_mask'] = torch.zeros(max_len) # 0 is masked out

    if add_CLS_token:
        input_sample['input_attn_mask'][:len(sent_obj['word'])+1] = torch.ones(len(sent_obj['word'])+1) # 1 is not masked
    else:
        input_sample['input_attn_mask'][:len(sent_obj['word'])] = torch.ones(len(sent_obj['word'])) # 1 is not masked
    

    # mask out padding tokens reverted: handle different use case: this is for pytorch transformers
    input_sample['input_attn_mask_invert'] = torch.ones(max_len) # 1 is masked out

    if add_CLS_token:
        input_sample['input_attn_mask_invert'][:len(sent_obj['word'])+1] = torch.zeros(len(sent_obj['word'])+1) # 0 is not masked
    else:
        input_sample['input_attn_mask_invert'][:len(sent_obj['word'])] = torch.zeros(len(sent_obj['word'])) # 0 is not masked

    # mask out target padding for computing cross entropy loss
    input_sample['target_mask'] = target_tokenized['attention_mask'][0]
    input_sample['seq_len'] = len(sent_obj['word'])
    
    # clean 0 length data
    if input_sample['seq_len'] == 0:
        print('discard length zero instance: ', target_string)
        return None

    return input_sample


def create_stratified_split(input_dataset_dicts, tokenizer, subjects, train_ratio=0.8, dev_ratio=0.1, test_ratio=0.1, max_len=DEFAULT_MAX_LEN, random_state=42, text_unique=False):
    """
    Create stratified train/dev/test splits based on token distribution.
    Ensures that each token appears proportionally across all splits.

    Args:
        input_dataset_dicts: List of dataset dictionaries
        tokenizer: Tokenizer to use for extracting tokens
        subjects: List of subject IDs to include
        train_ratio: Proportion for training set (default 0.8)
        dev_ratio: Proportion for dev set (default 0.1)
        test_ratio: Proportion for test set (default 0.1)
        max_len: Maximum sequence length for tokenization
        random_state: Random seed for reproducibility
        text_unique: If True, ensures same text never appears in multiple splits (default False)

    Returns:
        train_indices: dict mapping {dataset_idx: {subject: [sentence_indices]}}
        dev_indices: dict mapping {dataset_idx: {subject: [sentence_indices]}}
        test_indices: dict mapping {dataset_idx: {subject: [sentence_indices]}}
    """

    if text_unique:
        print('[INFO] Creating stratified split with text-unique constraint (same text never in multiple splits)...')
    else:
        print('[INFO] Creating stratified split based on token distribution...')

    # Step 1: Collect all sentences and tokenize them
    if text_unique:
        # Group by text content first
        text_to_instances = defaultdict(list)  # text -> [(dataset_idx, subject, sent_idx)]
        text_to_tokens = {}  # text -> set of token IDs

    all_sentences = []  # List of (dataset_idx, subject, sent_idx) or (text_key,) for text_unique
    all_token_ids = []  # List of token ID sets

    # Track statistics
    total_sentences = 0
    filtered_sentences = 0

    for dataset_idx, input_dataset_dict in enumerate(input_dataset_dicts):
        for subject in subjects:
            # Skip if subject doesn't exist in this dataset
            if subject not in input_dataset_dict:
                continue

            for sent_idx, sent_obj in enumerate(input_dataset_dict[subject]):
                total_sentences += 1

                if sent_obj is None:
                    filtered_sentences += 1
                    continue

                # Get the content and tokenize
                target_string = sent_obj['content']

                # Handle weird cases (same as in get_input_sample)
                if 'emp11111ty' in target_string:
                    target_string = target_string.replace('emp11111ty', 'empty')
                if 'film.1' in target_string:
                    target_string = target_string.replace('film.1', 'film.')

                # Tokenize
                target_tokenized = tokenizer(target_string, padding='max_length', max_length=max_len,
                                            truncation=True, return_tensors='pt', return_attention_mask=True)
                token_ids = target_tokenized['input_ids'][0]
                attention_mask = target_tokenized['attention_mask'][0]

                # Extract non-padding tokens
                non_padding_tokens = token_ids[attention_mask == 1].tolist()

                if text_unique:
                    # Group by text content
                    text_to_instances[target_string].append((dataset_idx, subject, sent_idx))
                    if target_string not in text_to_tokens:
                        text_to_tokens[target_string] = set(non_padding_tokens)
                else:
                    # Keep individual sentences
                    all_sentences.append((dataset_idx, subject, sent_idx))
                    all_token_ids.append(set(non_padding_tokens))

    if text_unique:
        # Build sentence list from unique texts
        all_sentences = list(text_to_instances.keys())
        all_token_ids = [text_to_tokens[text] for text in all_sentences]
        valid_sentences = sum(len(instances) for instances in text_to_instances.values())
        unique_texts = len(all_sentences)
        print(f'[INFO] Total sentences: {total_sentences}')
        print(f'[INFO] Valid sentences: {valid_sentences}')
        print(f'[INFO] Unique texts: {unique_texts}')
        print(f'[INFO] Filtered out: {filtered_sentences} ({100*filtered_sentences/total_sentences:.2f}%)')
    else:
        valid_sentences = len(all_sentences)
        print(f'[INFO] Total sentences: {total_sentences}')
        print(f'[INFO] Valid sentences: {valid_sentences}')
        print(f'[INFO] Filtered out: {filtered_sentences} ({100*filtered_sentences/total_sentences:.2f}%)')

    # Step 2: Build vocabulary of all unique tokens
    all_unique_tokens = set()
    for token_set in all_token_ids:
        all_unique_tokens.update(token_set)

    token_to_idx = {token: idx for idx, token in enumerate(sorted(all_unique_tokens))}
    print(f'[INFO] Found {len(all_unique_tokens)} unique tokens')

    # Step 3: Build binary matrix (n_items x n_tokens)
    n_items = len(all_sentences)
    n_tokens = len(all_unique_tokens)

    binary_matrix = np.zeros((n_items, n_tokens), dtype=np.int8)

    for item_idx, token_set in enumerate(all_token_ids):
        for token in token_set:
            binary_matrix[item_idx, token_to_idx[token]] = 1

    print(f'[INFO] Binary matrix shape: {binary_matrix.shape}')
    print(f'[INFO] Sparsity: {100 * (1 - binary_matrix.sum() / binary_matrix.size):.2f}%')

    # Step 4: Perform stratified splitting
    # First split: train (80%) vs temp (20%)
    msss1 = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=(dev_ratio + test_ratio),
                                              random_state=random_state)
    train_indices, temp_indices = next(msss1.split(np.zeros(n_items), binary_matrix))

    # Second split: dev (50% of temp) vs test (50% of temp)
    binary_matrix_temp = binary_matrix[temp_indices]
    msss2 = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=random_state)
    dev_indices_temp, test_indices_temp = next(msss2.split(np.zeros(len(temp_indices)), binary_matrix_temp))

    # Map back to original indices
    dev_indices = temp_indices[dev_indices_temp]
    test_indices = temp_indices[test_indices_temp]

    if text_unique:
        print(f'[INFO] Split unique texts: train={len(train_indices)}, dev={len(dev_indices)}, test={len(test_indices)}')
    else:
        print(f'[INFO] Split sizes: train={len(train_indices)}, dev={len(dev_indices)}, test={len(test_indices)}')

    # Step 5: Organize indices by (dataset_idx, subject)
    train_dict = defaultdict(lambda: defaultdict(list))
    dev_dict = defaultdict(lambda: defaultdict(list))
    test_dict = defaultdict(lambda: defaultdict(list))

    if text_unique:
        # Map texts back to all their instances
        train_count = 0
        dev_count = 0
        test_count = 0

        for idx in train_indices:
            text = all_sentences[idx]
            for dataset_idx, subject, sent_idx in text_to_instances[text]:
                train_dict[dataset_idx][subject].append(sent_idx)
                train_count += 1

        for idx in dev_indices:
            text = all_sentences[idx]
            for dataset_idx, subject, sent_idx in text_to_instances[text]:
                dev_dict[dataset_idx][subject].append(sent_idx)
                dev_count += 1

        for idx in test_indices:
            text = all_sentences[idx]
            for dataset_idx, subject, sent_idx in text_to_instances[text]:
                test_dict[dataset_idx][subject].append(sent_idx)
                test_count += 1

        print(f'[INFO] Split instances: train={train_count}, dev={dev_count}, test={test_count}')

        # Verify no text overlap
        train_texts = set(all_sentences[idx] for idx in train_indices)
        dev_texts = set(all_sentences[idx] for idx in dev_indices)
        test_texts = set(all_sentences[idx] for idx in test_indices)
        assert len(train_texts & dev_texts) == 0, "Text overlap between train and dev!"
        assert len(train_texts & test_texts) == 0, "Text overlap between train and test!"
        assert len(dev_texts & test_texts) == 0, "Text overlap between dev and test!"
        print('[INFO] Verified: No text overlap between splits ✓')
    else:
        for idx in train_indices:
            dataset_idx, subject, sent_idx = all_sentences[idx]
            train_dict[dataset_idx][subject].append(sent_idx)

        for idx in dev_indices:
            dataset_idx, subject, sent_idx = all_sentences[idx]
            dev_dict[dataset_idx][subject].append(sent_idx)

        for idx in test_indices:
            dataset_idx, subject, sent_idx = all_sentences[idx]
            test_dict[dataset_idx][subject].append(sent_idx)

    return train_dict, dev_dict, test_dict


def analyze_token_distribution(input_dataset_dicts, tokenizer, subjects, train_indices_dict, dev_indices_dict, test_indices_dict, max_len=DEFAULT_MAX_LEN):
    """
    Analyze and report token distribution across train/dev/test splits.

    Args:
        input_dataset_dicts: List of dataset dictionaries
        tokenizer: Tokenizer to use
        subjects: List of subject IDs
        train_indices_dict: Training indices from stratified split
        dev_indices_dict: Dev indices from stratified split
        test_indices_dict: Test indices from stratified split
        max_len: Maximum sequence length

    Returns:
        Dictionary with distribution statistics
    """

    def get_tokens_from_indices(input_dataset_dicts, indices_dict, subjects):
        """Extract all tokens from sentences specified by indices"""
        all_tokens = Counter()
        for dataset_idx, input_dataset_dict in enumerate(input_dataset_dicts):
            for subject in subjects:
                if dataset_idx in indices_dict and subject in indices_dict[dataset_idx]:
                    for sent_idx in indices_dict[dataset_idx][subject]:
                        sent_obj = input_dataset_dict[subject][sent_idx]
                        if sent_obj is None:
                            continue
                        target_string = sent_obj['content']

                        # Handle weird cases
                        if 'emp11111ty' in target_string:
                            target_string = target_string.replace('emp11111ty', 'empty')
                        if 'film.1' in target_string:
                            target_string = target_string.replace('film.1', 'film.')

                        # Tokenize
                        target_tokenized = tokenizer(target_string, padding='max_length', max_length=max_len,
                                                    truncation=True, return_tensors='pt', return_attention_mask=True)
                        token_ids = target_tokenized['input_ids'][0]
                        attention_mask = target_tokenized['attention_mask'][0]
                        non_padding_tokens = token_ids[attention_mask == 1].tolist()

                        all_tokens.update(non_padding_tokens)
        return all_tokens

    print('\n[INFO] Analyzing token distribution across splits...')

    train_tokens = get_tokens_from_indices(input_dataset_dicts, train_indices_dict, subjects)
    dev_tokens = get_tokens_from_indices(input_dataset_dicts, dev_indices_dict, subjects)
    test_tokens = get_tokens_from_indices(input_dataset_dicts, test_indices_dict, subjects)

    train_vocab = set(train_tokens.keys())
    dev_vocab = set(dev_tokens.keys())
    test_vocab = set(test_tokens.keys())
    all_vocab = train_vocab | dev_vocab | test_vocab

    # Calculate statistics
    dev_unseen = dev_vocab - train_vocab
    test_unseen = test_vocab - train_vocab
    dev_unseen_pct = 100 * len(dev_unseen) / len(dev_vocab) if len(dev_vocab) > 0 else 0
    test_unseen_pct = 100 * len(test_unseen) / len(test_vocab) if len(test_vocab) > 0 else 0

    print(f'\n--- Token Distribution Analysis ---')
    print(f'Total unique tokens: {len(all_vocab)}')
    print(f'Train vocabulary size: {len(train_vocab)} ({100*len(train_vocab)/len(all_vocab):.1f}% of total)')
    print(f'Dev vocabulary size: {len(dev_vocab)} ({100*len(dev_vocab)/len(all_vocab):.1f}% of total)')
    print(f'Test vocabulary size: {len(test_vocab)} ({100*len(test_vocab)/len(all_vocab):.1f}% of total)')
    print(f'\nUnseen tokens in dev set: {len(dev_unseen)} ({dev_unseen_pct:.2f}% of dev vocab)')
    print(f'Unseen tokens in test set: {len(test_unseen)} ({test_unseen_pct:.2f}% of test vocab)')
    print(f'-----------------------------------\n')

    return {
        'total_vocab': len(all_vocab),
        'train_vocab': len(train_vocab),
        'dev_vocab': len(dev_vocab),
        'test_vocab': len(test_vocab),
        'dev_unseen': len(dev_unseen),
        'test_unseen': len(test_unseen),
        'dev_unseen_pct': dev_unseen_pct,
        'test_unseen_pct': test_unseen_pct
    }


class ZuCo_dataset(Dataset):
    def __init__(self, input_dataset_dicts, phase, tokenizer, subject = 'ALL', eeg_type = 'GD', bands = ['_t1','_t2','_a1','_a2','_b1','_b2','_g1','_g2'], setting = 'unique_sent', is_add_CLS_token = False, test_input='noise', max_len = DEFAULT_MAX_LEN, use_stratified_split=True, text_unique_split=False, loaded_split_indices=None):
        self.inputs = [] # This will now store metadata for lazy loading
        self.tokenizer = tokenizer
        self.eeg_type = eeg_type
        self.bands = bands
        self.is_add_CLS_token = is_add_CLS_token
        self.test_input = test_input
        self.max_len = max_len

        if not isinstance(input_dataset_dicts,list):
            input_dataset_dicts = [input_dataset_dicts]
        print(f'[INFO]loading {len(input_dataset_dicts)} task datasets')

        # Determine subjects list (needed for stratification)
        # Collect subjects from ALL datasets (union) to maximize data usage
        if subject == 'ALL':
            subjects = set()
            for dataset_dict in input_dataset_dicts:
                subjects.update(dataset_dict.keys())
            subjects = sorted(list(subjects))  # Sort for consistency
            print('[INFO]using subjects from all datasets: ', subjects)
        else:
            subjects = [subject]

        # Perform stratified split if enabled and setting is 'unique_sent'
        phase_indices_dict = None
        train_indices_dict = None
        dev_indices_dict = None
        test_indices_dict = None

        if use_stratified_split and setting == 'unique_sent':
            if loaded_split_indices is not None:
                # Use pre-computed indices from training
                print(f'[INFO] Using loaded split indices for {setting} setting')
                train_indices_dict = loaded_split_indices['train']
                dev_indices_dict = loaded_split_indices['dev']
                test_indices_dict = loaded_split_indices['test']
            else:
                # Create new split
                print(f'[INFO] Creating new stratified split for {setting} setting')
                train_indices_dict, dev_indices_dict, test_indices_dict = create_stratified_split(
                    input_dataset_dicts, tokenizer, subjects, max_len=max_len, random_state=42, text_unique=text_unique_split
                )

            # Store all split indices for later saving (during training)
            self.split_indices = {
                'train': train_indices_dict,
                'dev': dev_indices_dict,
                'test': test_indices_dict
            }

            # Select the appropriate indices based on phase
            if phase == 'train':
                phase_indices_dict = train_indices_dict
            elif phase == 'dev':
                phase_indices_dict = dev_indices_dict
            elif phase == 'test':
                phase_indices_dict = test_indices_dict
            else:
                raise ValueError(f'Unknown phase: {phase}')
        else:
            self.split_indices = None

        for dataset_idx, input_dataset_dict in enumerate(input_dataset_dicts):
            if subject == 'ALL':
                subjects = list(input_dataset_dict.keys())
                print('[INFO]using subjects: ', subjects)
            else:
                subjects = [subject]

            if subjects[0] not in input_dataset_dict:
                print(f'[WARNING] subject {subjects[0]} not found in dataset {dataset_idx}, skipping...')
                continue
            
            total_num_sentence = len(input_dataset_dict[subjects[0]])
            
            train_divider = int(0.8*total_num_sentence)
            dev_divider = train_divider + int(0.1*total_num_sentence)
            
            print(f'train divider = {train_divider}')
            print(f'dev divider = {dev_divider}')
            if setting == 'unique_sent':
                if phase_indices_dict is not None:
                    # Use stratified split
                    print(f'[INFO]initializing a {phase} set using stratified split...')
                    for key in subjects:
                        if dataset_idx in phase_indices_dict and key in phase_indices_dict[dataset_idx]:
                            indices = phase_indices_dict[dataset_idx][key]
                            for i in indices:
                                # Validate sample by calling get_input_sample, but only store the reference
                                sent_obj = input_dataset_dict[key][i]
                                if sent_obj is not None:
                                    test_sample = get_input_sample(sent_obj, self.tokenizer, eeg_type, bands=bands, add_CLS_token=is_add_CLS_token, test_input=test_input, max_len=max_len)
                                    if test_sample is not None:
                                        self.inputs.append({'sent_obj': sent_obj})
                else:
                    # Use sequential split (old method)
                    if phase == 'train':
                        print('[INFO]initializing a train set...')
                        for key in subjects:
                            for i in range(train_divider):
                                sent_obj = input_dataset_dict[key][i]
                                if sent_obj is not None:
                                    test_sample = get_input_sample(sent_obj, self.tokenizer, eeg_type, bands=bands, add_CLS_token=is_add_CLS_token, test_input=test_input, max_len=max_len)
                                    if test_sample is not None:
                                        self.inputs.append({'sent_obj': sent_obj})
                    elif phase == 'dev':
                        print('[INFO]initializing a dev set...')
                        for key in subjects:
                            for i in range(train_divider,dev_divider):
                                sent_obj = input_dataset_dict[key][i]
                                if sent_obj is not None:
                                    test_sample = get_input_sample(sent_obj, self.tokenizer, eeg_type, bands=bands, add_CLS_token=is_add_CLS_token, test_input=test_input, max_len=max_len)
                                    if test_sample is not None:
                                        self.inputs.append({'sent_obj': sent_obj})
                    elif phase == 'test':
                        print('[INFO]initializing a test set...')
                        for key in subjects:
                            for i in range(dev_divider,total_num_sentence):
                                sent_obj = input_dataset_dict[key][i]
                                if sent_obj is not None:
                                    test_sample = get_input_sample(sent_obj, self.tokenizer, eeg_type, bands=bands, add_CLS_token=is_add_CLS_token, test_input=test_input, max_len=max_len)
                                    if test_sample is not None:
                                        self.inputs.append({'sent_obj': sent_obj})

            elif setting == 'unique_subj':
                print('WARNING!!! only implemented for SR v1 dataset ')
                # subject ['ZAB', 'ZDM', 'ZGW', 'ZJM', 'ZJN', 'ZJS', 'ZKB', 'ZKH', 'ZKW'] for train
                # subject ['ZMG'] for dev
                # subject ['ZPH'] for test
                if phase == 'train':
                    print(f'[INFO]initializing a train set using {setting} setting...')
                    for i in range(total_num_sentence):
                        for key in ['ZAB', 'ZDM', 'ZGW', 'ZJM', 'ZJN', 'ZJS', 'ZKB', 'ZKH','ZKW']:
                            sent_obj = input_dataset_dict[key][i]
                            if sent_obj is not None:
                                test_sample = get_input_sample(sent_obj, self.tokenizer, eeg_type, bands=bands, add_CLS_token=is_add_CLS_token)
                                if test_sample is not None:
                                    self.inputs.append({'sent_obj': sent_obj})
                if phase == 'dev':
                    print(f'[INFO]initializing a dev set using {setting} setting...')
                    for i in range(total_num_sentence):
                        for key in ['ZMG']:
                            sent_obj = input_dataset_dict[key][i]
                            if sent_obj is not None:
                                test_sample = get_input_sample(sent_obj, self.tokenizer, eeg_type, bands=bands, add_CLS_token=is_add_CLS_token)
                                if test_sample is not None:
                                    self.inputs.append({'sent_obj': sent_obj})
                if phase == 'test':
                    print(f'[INFO]initializing a test set using {setting} setting...')
                    for i in range(total_num_sentence):
                        for key in ['ZPH']:
                            sent_obj = input_dataset_dict[key][i]
                            if sent_obj is not None:
                                test_sample = get_input_sample(sent_obj, self.tokenizer, eeg_type, bands=bands, add_CLS_token=is_add_CLS_token)
                                if test_sample is not None:
                                    self.inputs.append({'sent_obj': sent_obj})

            print('++ adding task to dataset, now we have:', len(self.inputs))

        print(f'[INFO] Lazy loading enabled. Stored {len(self.inputs)} sample references.')
        print()

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        # Lazy loading: process the sample only when requested
        # All samples in self.inputs are pre-validated during __init__
        metadata = self.inputs[idx]
        input_sample = get_input_sample(
            metadata['sent_obj'], 
            self.tokenizer, 
            eeg_type=self.eeg_type, 
            bands=self.bands, 
            add_CLS_token=self.is_add_CLS_token, 
            test_input=self.test_input, 
            max_len=self.max_len
        )
        
        return (
            input_sample['input_embeddings'], 
            input_sample['seq_len'],
            input_sample['input_attn_mask'], 
            input_sample['input_attn_mask_invert'],
            input_sample['target_ids'], 
            input_sample['target_mask'], 
            input_sample['sentiment_label'], 
            #input_sample['sent_level_EEG']
        )
        # keys: input_embeddings, input_attn_mask, input_attn_mask_invert, target_ids, target_mask, 



"""for train classifier on stanford sentiment treebank text-sentiment pairs"""
