import json
import pickle
import sys

import numpy as np
import torch
from transformers import BartTokenizer, PegasusTokenizer, T5Tokenizer

from data import ZuCo_dataset

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

    sets = {
        phase:
        ZuCo_dataset(whole_dataset_dicts, phase, tokenizer, subject = subject_choice, eeg_type = eeg_type_choice, bands = bands_choice, setting = dataset_setting, test_input=test_input)
        for phase in ['train', 'dev', 'test']
    }

    target_ids = {set_name: torch.IntTensor([sample for item in sets[set_name] for sample in item[4]]).unique() for set_name in sets}

    dev_not_in_train = np.setdiff1d(target_ids['dev'], target_ids['train'])
    test_not_in_train = np.setdiff1d(target_ids['test'], target_ids['train'])

    print(f"Number of items: {len(target_ids['train'])} in train, {len(target_ids['dev'])} in dev, {len(target_ids['test'])} in test")
    print(f"Number of dev items not in train: {len(dev_not_in_train)} ({(len(dev_not_in_train)/len(target_ids['dev']))*100:.2f}%)")
    print(f"Number of test items not in train: {len(test_not_in_train) } ({(len(test_not_in_train)/len(target_ids['test']))*100:.2f}%)")
    pass