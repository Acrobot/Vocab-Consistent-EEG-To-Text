import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
import pickle
import json
import matplotlib.pyplot as plt
from glob import glob
import time
import copy
from tqdm import tqdm
from transformers import BertLMHeadModel, BartTokenizer, BartForConditionalGeneration, BartConfig, BartForSequenceClassification, BertTokenizer, BertConfig, BertForSequenceClassification, RobertaTokenizer, RobertaForSequenceClassification, PegasusForConditionalGeneration, PegasusTokenizer, T5Tokenizer, T5ForConditionalGeneration, BertGenerationEncoder, BertGenerationDecoder, EncoderDecoderConfig, EncoderDecoderModel, get_cosine_schedule_with_warmup
from data import ZuCo_dataset
from model_decoding import BrainTranslator, BrainTranslatorNaive, BrainTranslatorBaseline, BrainTranslatorV2, T5Translator
from config import get_config

def train_model(dataloaders, device, model, criterion, optimizer, scheduler, num_epochs=25, checkpoint_path_best = './checkpoints/decoding/best/temp_decoding.pt', checkpoint_path_last = './checkpoints/decoding/last/temp_decoding.pt'):
    # modified from: https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 100000000000

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'dev']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0

            # Iterate over data.
            for input_embeddings, seq_len, input_masks, input_mask_invert, target_ids, target_mask, sentiment_labels in tqdm(dataloaders[phase]):

                # load in batch
                input_embeddings_batch = input_embeddings.to(device).float()
                input_masks_batch = input_masks.to(device)
                input_mask_invert_batch = input_mask_invert.to(device)
                target_ids_batch = target_ids.to(device)
                """replace padding ids in target_ids with -100"""
                target_ids_batch[target_ids_batch == tokenizer.pad_token_id] = -100

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
    	        # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    seq2seqLMoutput = model(input_embeddings_batch, input_masks_batch, input_mask_invert_batch, target_ids_batch)

                    """calculate loss"""
                    # logits = seq2seqLMoutput.logits # 8*48*50265
                    # logits = logits.permute(0,2,1) # 8*50265*48

                    # loss = criterion(logits, target_ids_batch_label) # calculate cross entropy loss only on encoded target parts
                    # NOTE: my criterion not used
                    loss = seq2seqLMoutput.loss # use the BART language modeling loss

                    # """check prediction, instance 0 of each batch"""
                    # print('target size:', target_ids_batch.size(), ',original logits size:', logits.size(), ',target_mask size', target_mask_batch.size())
                    # logits = logits.permute(0,2,1)
                    # for idx in [0]:
                    #     print(f'-- instance {idx} --')
                    #     # print('permuted logits size:', logits.size())
                    #     probs = logits[idx].softmax(dim = 1)
                    #     # print('probs size:', probs.size())
                    #     values, predictions = probs.topk(1)
                    #     # print('predictions before squeeze:',predictions.size())
                    #     predictions = torch.squeeze(predictions)
                    #     # print('predictions:',predictions)
                    #     # print('target mask:', target_mask_batch[idx])
                    #     # print('[DEBUG]target tokens:',tokenizer.decode(target_ids_batch_copy[idx]))
                    #     print('[DEBUG]predicted tokens:',tokenizer.decode(predictions))

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        # with torch.autograd.detect_anomaly():
                        loss.sum().backward()
                        # Clip gradients to prevent exploding gradients and loss spikes
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        optimizer.step()

                # statistics
                running_loss += loss.sum().item() * input_embeddings_batch.size()[0] # batch loss
                # print('[DEBUG]loss:',loss.item())
                # print('#################################')


            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]

            print('{} Loss: {:.4f}'.format(phase, epoch_loss))
            wandb.log({f"{phase}/loss": epoch_loss})

            # deep copy the model
            if phase == 'dev' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                '''save checkpoint'''
                torch.save(model.state_dict(), checkpoint_path_best)
                print(f'update best on dev checkpoint: {checkpoint_path_best}')
                # with torch.set_grad_enabled(False):
                #     traced_model_1 = torch.jit.trace(model, (torch.rand(1, 56, 840).to(device), torch.randint(1, 56).to(device), torch.rand(1, 56).to(device), torch.rand(1, 56).to(device)))
                #     traced_model_32 = torch.jit.trace(model, (torch.rand(32, 56, 840).to(device), torch.randint(32, 56).to(device), torch.rand(32, 56).to(device), torch.rand(32, 56).to(device)))
                # torch.jit.save(traced_model_1, checkpoint_path_best[:-3]+'_1_jit.pt')
                # torch.jit.save(traced_model_32, checkpoint_path_best[:-3]+'_32_jit.pt')
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val loss: {:4f}'.format(best_loss))
    torch.save(model.state_dict(), checkpoint_path_last)
    print(f'update last checkpoint: {checkpoint_path_last}')

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

def show_require_grad_layers(model):
    print()
    print(' require_grad layers:')
    # sanity check
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(' ', name)

def train_model_contrastive(dataloaders, device, model, criterion, optimizer, scheduler, num_epochs=25,
                             checkpoint_path_best='./checkpoints/decoding/best/temp_decoding.pt',
                             checkpoint_path_last='./checkpoints/decoding/last/temp_decoding.pt',
                             contrastive_lambda=0.3):
    """Training with noise-contrastive auxiliary loss.

    For each batch, runs both real EEG and random noise through the encoder,
    then adds a contrastive loss that pushes their representations apart.
    This forces the encoder to produce EEG-specific representations.
    """
    import torch.nn.functional as F
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 100000000000

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        for phase in ['train', 'dev']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_lm_loss = 0.0
            running_contrastive_loss = 0.0

            for input_embeddings, seq_len, input_masks, input_mask_invert, target_ids, target_mask, sentiment_labels in tqdm(dataloaders[phase]):
                input_embeddings_batch = input_embeddings.to(device).float()
                input_masks_batch = input_masks.to(device)
                input_mask_invert_batch = input_mask_invert.to(device)
                target_ids_batch = target_ids.to(device)
                target_ids_batch[target_ids_batch == tokenizer.pad_token_id] = -100

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    # Standard forward pass with EEG input
                    seq2seqLMoutput = model(input_embeddings_batch, input_masks_batch, input_mask_invert_batch, target_ids_batch)
                    lm_loss = seq2seqLMoutput.loss

                    # Noise-contrastive auxiliary loss
                    if contrastive_lambda > 0:
                        # Get encoder output for real EEG (detached to prevent gradient interference)
                        if hasattr(model, 'module'):
                            eeg_encoded = model.module.addin_forward(input_embeddings_batch, input_mask_invert_batch).detach()
                        else:
                            eeg_encoded = model.addin_forward(input_embeddings_batch, input_mask_invert_batch).detach()

                        # Generate noise and get encoder output
                        noise_input = torch.randn_like(input_embeddings_batch)
                        if hasattr(model, 'module'):
                            noise_encoded = model.module.addin_forward(noise_input, input_mask_invert_batch)
                        else:
                            noise_encoded = model.addin_forward(noise_input, input_mask_invert_batch)

                        # Pool across sequence dimension (only non-masked positions)
                        # Use input_masks_batch (1=real, 0=pad) to mask pooling
                        mask_for_pool = input_masks_batch[:, :eeg_encoded.shape[1]].unsqueeze(-1).float()
                        eeg_pooled = (eeg_encoded * mask_for_pool).sum(dim=1) / mask_for_pool.sum(dim=1).clamp(min=1)
                        noise_pooled = (noise_encoded * mask_for_pool).sum(dim=1) / mask_for_pool.sum(dim=1).clamp(min=1)

                        # Cosine similarity: we want this to be -1 (maximally different)
                        cosine_sim = F.cosine_similarity(eeg_pooled, noise_pooled, dim=1)
                        contrastive_loss = cosine_sim.mean()  # Minimize positive similarity

                        loss = lm_loss + contrastive_lambda * contrastive_loss
                    else:
                        loss = lm_loss
                        contrastive_loss = torch.tensor(0.0)

                    if phase == 'train':
                        loss.sum().backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        optimizer.step()

                running_loss += loss.sum().item() * input_embeddings_batch.size()[0]
                running_lm_loss += lm_loss.sum().item() * input_embeddings_batch.size()[0]
                running_contrastive_loss += contrastive_loss.item() * input_embeddings_batch.size()[0]

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_lm_loss = running_lm_loss / dataset_sizes[phase]
            epoch_contrastive_loss = running_contrastive_loss / dataset_sizes[phase]

            print('{} Loss: {:.4f} (LM: {:.4f}, Contrastive: {:.4f})'.format(
                phase, epoch_loss, epoch_lm_loss, epoch_contrastive_loss))
            wandb.log({
                f"{phase}/loss": epoch_loss,
                f"{phase}/lm_loss": epoch_lm_loss,
                f"{phase}/contrastive_loss": epoch_contrastive_loss,
            })

            if phase == 'dev' and epoch_lm_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), checkpoint_path_best)
                print(f'update best on dev checkpoint: {checkpoint_path_best}')
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val loss: {:4f}'.format(best_loss))
    torch.save(model.state_dict(), checkpoint_path_last)
    print(f'update last checkpoint: {checkpoint_path_last}')

    model.load_state_dict(best_model_wts)
    return model

if __name__ == '__main__':
    args = get_config('train_decoding')

    ''' config param'''
    dataset_setting = 'unique_sent'

    num_epochs_step1 = args['num_epoch_step1']
    num_epochs_step2 = args['num_epoch_step2']
    step1_lr = args['learning_rate_step1']
    step2_lr = args['learning_rate_step2']

    batch_size = args['batch_size']

    model_name = args['model_name']
    # model_name = 'BrainTranslatorNaive' # with no additional transformers
    # model_name = 'BrainTranslator'

    # task_name = 'task1'
    # task_name = 'task1_task2'
    # task_name = 'task1_task2_task3'
    # task_name = 'task1_task2_taskNRv2'
    task_name = args['task_name']
    train_input = args['train_input']
    print("train_input is:", train_input)
    save_path = args['save_path']
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    skip_step_one = args['skip_step_one']
    load_step1_checkpoint = args['load_step1_checkpoint']
    use_random_init = args['use_random_init']
    device_ids = [0] # device setting

    if use_random_init and skip_step_one:
        step2_lr = 5*1e-4

    print(f'[INFO]using model: {model_name}')

    if skip_step_one:
        save_name = f'{task_name}_finetune_{model_name}_skipstep1_b{batch_size}_{num_epochs_step1}_{num_epochs_step2}_{step1_lr}_{step2_lr}_{dataset_setting}_{train_input}'
    else:
        save_name = f'{task_name}_finetune_{model_name}_2steptraining_b{batch_size}_{num_epochs_step1}_{num_epochs_step2}_{step1_lr}_{step2_lr}_{dataset_setting}_{train_input}'

    if use_random_init:
        save_name = 'randinit_' + save_name

    save_path_best = os.path.join(save_path, 'best')
    if not os.path.exists(save_path_best):
        os.makedirs(save_path_best)

    output_checkpoint_name_best = os.path.join(save_path_best, f'{save_name}.pt')

    save_path_last = os.path.join(save_path, 'last')
    if not os.path.exists(save_path_last):
        os.makedirs(save_path_last)

    output_checkpoint_name_last = os.path.join(save_path_last, f'{save_name}.pt')

    # subject_choice = 'ALL
    subject_choice = args['subjects']
    print(f'![Debug]using {subject_choice}')
    # eeg_type_choice = 'GD
    eeg_type_choice = args['eeg_type']
    print(f'[INFO]eeg type {eeg_type_choice}')
    # bands_choice = ['_t1']
    # bands_choice = ['_t1','_t2','_a1','_a2','_b1','_b2','_g1','_g2']
    bands_choice = args['eeg_bands']
    print(f'[INFO]using bands {bands_choice}')



    ''' set random seeds '''
    seed_val = 312
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)
    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # For Python hashing
    os.environ['PYTHONHASHSEED'] = str(seed_val)


    ''' set up device '''
    # use cuda
    if torch.cuda.is_available():
        # dev = "cuda:3"
        dev = args['cuda']
    else:
        dev = "cpu"
    # CUDA_VISIBLE_DEVICES=0,1,2,3
    device = torch.device(dev)
    print(f'[INFO]using device {dev}')
    print()

    ''' set up dataloader '''
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

    print()

    """save config"""
    cfg_dir = './config/decoding/'

    if not os.path.exists(cfg_dir):
        os.makedirs(cfg_dir)

    if model_name in ['BrainTranslator','BrainTranslatorNaive', 'BrainTranslatorBaseline', 'BrainTranslatorV2']:
        tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')

    elif model_name == 'PegasusTranslator':
        tokenizer = PegasusTokenizer.from_pretrained('google/pegasus-xsum')

    elif model_name == 'T5Translator':
        tokenizer = T5Tokenizer.from_pretrained("t5-large")
        #tokenizer.set_prefix_tokens(language='english')

    # Get text_unique_split option from config (defaults to False if not specified)
    text_unique_split = args.get('text_unique_split', False)
    print(f'[INFO]text_unique_split: {text_unique_split}')

    # Get use_sequential_split option (defaults to False = stratified vocab-consistent split)
    use_sequential_split = args.get('use_sequential_split', False)
    use_stratified_split = not use_sequential_split
    print(f'[INFO]use_sequential_split: {use_sequential_split} (use_stratified_split={use_stratified_split})')

    # train dataset
    train_set = ZuCo_dataset(whole_dataset_dicts, 'train', tokenizer, subject = subject_choice, eeg_type = eeg_type_choice, bands = bands_choice, setting = dataset_setting, test_input=train_input, text_unique_split=text_unique_split, use_stratified_split=use_stratified_split)
    # dev dataset
    dev_set = ZuCo_dataset(whole_dataset_dicts, 'dev', tokenizer, subject = subject_choice, eeg_type = eeg_type_choice, bands = bands_choice, setting = dataset_setting, test_input=train_input, text_unique_split=text_unique_split, use_stratified_split=use_stratified_split)
    # test dataset
    # test_set = ZuCo_dataset(whole_dataset_dicts, 'test', tokenizer, subject = subject_choice, eeg_type = eeg_type_choice, bands = bands_choice, setting = dataset_setting)

    # Save split indices in config if stratified split was used
    if train_set.split_indices is not None:
        # Convert defaultdicts to regular dicts for JSON serialization
        split_indices_serializable = {}
        for split_name, split_dict in train_set.split_indices.items():
            split_indices_serializable[split_name] = {
                str(dataset_idx): {subject: indices for subject, indices in subjects_dict.items()}
                for dataset_idx, subjects_dict in split_dict.items()
            }
        args['split_indices'] = split_indices_serializable
        print(f'[INFO] Added split indices to config')

    # Save config with split indices
    with open(os.path.join(cfg_dir,f'{save_name}.json'), 'w') as out_config:
        json.dump(args, out_config, indent = 4)

    dataset_sizes = {'train': len(train_set), 'dev': len(dev_set)}
    print('[INFO]train_set size: ', len(train_set))
    print('[INFO]dev_set size: ', len(dev_set))
    # print('[INFO]test_set size: ', len(test_set))

    # Create generator for reproducible shuffling
    generator = torch.Generator()
    generator.manual_seed(seed_val)

    # train dataloader
    train_dataloader = DataLoader(train_set, batch_size = batch_size, shuffle=True, num_workers=1, generator=generator)
    # dev dataloader
    val_dataloader = DataLoader(dev_set, batch_size = 1, shuffle=False, num_workers=1)
    # dataloaders
    dataloaders = {'train':train_dataloader, 'dev':val_dataloader}

    ''' set up model '''
    if model_name == 'BrainTranslator':
        if use_random_init:
            config = BartConfig.from_pretrained('facebook/bart-large')
            pretrained = BartForConditionalGeneration(config)
        else:
            pretrained = BartForConditionalGeneration.from_pretrained('facebook/bart-large')

        dropout_rate = args['dropout_rate']
        model = BrainTranslator(pretrained, in_feature = 105*len(bands_choice), decoder_embedding_size = 1024, additional_encoder_nhead=8, additional_encoder_dim_feedforward = 2048, dropout=dropout_rate)


    elif model_name == 'BrainTranslatorBaseline':
        if use_random_init:
            config = BartConfig.from_pretrained('facebook/bart-large')
            pretrained = BartForConditionalGeneration(config)
        else:
            pretrained = BartForConditionalGeneration.from_pretrained('facebook/bart-large')

        dropout_rate = args['dropout_rate']
        model = BrainTranslatorBaseline(pretrained, in_feature = 105*len(bands_choice), decoder_embedding_size = 1024, additional_encoder_nhead=8, additional_encoder_dim_feedforward = 2048, dropout=dropout_rate)

    elif model_name == 'BrainTranslatorV2':
        if use_random_init:
            config = BartConfig.from_pretrained('facebook/bart-large')
            pretrained = BartForConditionalGeneration(config)
        else:
            pretrained = BartForConditionalGeneration.from_pretrained('facebook/bart-large')

        dropout_rate = args['dropout_rate']
        model = BrainTranslatorV2(pretrained, in_feature = 105*len(bands_choice), decoder_embedding_size = 1024, additional_encoder_nhead=8, additional_encoder_dim_feedforward = 2048, dropout=dropout_rate)

    elif model_name == 'BrainTranslatorNaive':
        pretrained = BartForConditionalGeneration.from_pretrained('facebook/bart-large')
        model = BrainTranslatorNaive(pretrained, in_feature = 105*len(bands_choice), decoder_embedding_size = 1024, additional_encoder_nhead=8, additional_encoder_dim_feedforward = 2048)

    elif model_name == 'PegasusTranslator':
        pretrained = PegasusForConditionalGeneration.from_pretrained('google/pegasus-xsum')
        model = BrainTranslator(pretrained, in_feature = 105*len(bands_choice), decoder_embedding_size = 1024, additional_encoder_nhead=8, additional_encoder_dim_feedforward = 2048)

    elif model_name == 'T5Translator':
        pretrained = T5ForConditionalGeneration.from_pretrained("t5-large")
        model = T5Translator(pretrained, in_feature = 105*len(bands_choice), decoder_embedding_size = 1024, additional_encoder_nhead=8, additional_encoder_dim_feedforward = 2048)

    model.to(device)
    model = torch.nn.DataParallel(model, device_ids=device_ids)

    ''' training loop '''

    ######################################################
    '''step one trainig: freeze most of BART params'''
    ######################################################

    # closely follow BART paper
    if model_name in ['BrainTranslator','BrainTranslatorNaive', 'BrainTranslatorBaseline', 'BrainTranslatorV2', 'PegasusTranslator', 'T5Translator']:
        for name, param in model.named_parameters():
            if param.requires_grad and 'pretrained' in name:
                if ('shared' in name) or ('embed_positions' in name) or ('encoder.layers.0' in name):
                    continue
                else:
                    param.requires_grad = False

    elif model_name == 'BertGeneration':
        for name, param in model.named_parameters():
            if param.requires_grad and 'pretrained' in name:
                if ('embeddings' in name) or ('encoder.layer.0' in name):
                    continue
                else:
                    param.requires_grad = False


    if skip_step_one:
        if load_step1_checkpoint:
            stepone_checkpoint = 'path_to_step_1_checkpoint.pt'
            print(f'skip step one, load checkpoint: {stepone_checkpoint}')
            model.load_state_dict(torch.load(stepone_checkpoint))
        else:
            print('skip step one, start from scratch at step two')
    else:

        ''' set up optimizer and scheduler'''
        # Use AdamW optimizer (state-of-the-art for transformers)
        optimizer_step1 = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=step1_lr,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.01
        )

        # Calculate total training steps for warmup
        steps_per_epoch_step1 = len(dataloaders['train'])
        total_steps_step1 = steps_per_epoch_step1 * num_epochs_step1
        warmup_steps_step1 = int(0.1 * total_steps_step1)  # 10% warmup

        # Use cosine schedule with warmup (smooth learning rate transitions)
        exp_lr_scheduler_step1 = get_cosine_schedule_with_warmup(
            optimizer_step1,
            num_warmup_steps=warmup_steps_step1,
            num_training_steps=total_steps_step1
        )

        print(f'[INFO] Step1 - Total steps: {total_steps_step1}, Warmup steps: {warmup_steps_step1}')

        ''' set up loss function '''
        criterion = nn.CrossEntropyLoss()

        print('=== start Step1 training ... ===')
        # print training layers
        show_require_grad_layers(model)

        if not wandb.run:
            wandb_run_name = f"Subj: {args['subjects']} | {model_name} | {task_name}"
            wandb.init(entity="YOUR_WANDB_ENTITY", project="EEG-To-Text", config=args, name=wandb_run_name,
                       settings=wandb.Settings(init_timeout=300))

        # return best loss model from step1 training
        model = train_model(dataloaders, device, model, criterion, optimizer_step1, exp_lr_scheduler_step1, num_epochs=num_epochs_step1, checkpoint_path_best = output_checkpoint_name_best, checkpoint_path_last = output_checkpoint_name_last)

    ######################################################
    '''step two trainig: update whole model for a few iterations'''
    ######################################################
    for name, param in model.named_parameters():
        param.requires_grad = True

    ''' set up optimizer and scheduler'''
    # Use AdamW optimizer
    optimizer_step2 = optim.AdamW(
        model.parameters(),
        lr=step2_lr,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01
    )

    # Calculate total training steps for warmup
    steps_per_epoch_step2 = len(dataloaders['train'])
    total_steps_step2 = steps_per_epoch_step2 * num_epochs_step2
    warmup_steps_step2 = int(0.1 * total_steps_step2)  # 10% warmup

    # Use cosine schedule with warmup (smooth learning rate transitions)
    exp_lr_scheduler_step2 = get_cosine_schedule_with_warmup(
        optimizer_step2,
        num_warmup_steps=warmup_steps_step2,
        num_training_steps=total_steps_step2
    )

    print(f'[INFO] Step2 - Total steps: {total_steps_step2}, Warmup steps: {warmup_steps_step2}')

    ''' set up loss function '''
    criterion = nn.CrossEntropyLoss()

    print()
    print('=== start Step2 training ... ===')
    # print training layers
    show_require_grad_layers(model)

    # Initialize wandb only if not already initialized (step1 may have done it)
    if not wandb.run:
        wandb_run_name = f"Subj: {args['subjects']} | {model_name} | {task_name}"
        wandb.init(entity="YOUR_WANDB_ENTITY", project="EEG-To-Text", config=args, name=wandb_run_name,
                   settings=wandb.Settings(init_timeout=300))

    '''main loop'''
    contrastive_lambda = args.get('contrastive_lambda', 0.0)
    if contrastive_lambda > 0:
        print(f'[INFO] Using noise-contrastive training with lambda={contrastive_lambda}')
        trained_model = train_model_contrastive(dataloaders, device, model, criterion, optimizer_step2, exp_lr_scheduler_step2,
                                                 num_epochs=num_epochs_step2, checkpoint_path_best=output_checkpoint_name_best,
                                                 checkpoint_path_last=output_checkpoint_name_last, contrastive_lambda=contrastive_lambda)
    else:
        trained_model = train_model(dataloaders, device, model, criterion, optimizer_step2, exp_lr_scheduler_step2, num_epochs=num_epochs_step2, checkpoint_path_best = output_checkpoint_name_best, checkpoint_path_last = output_checkpoint_name_last)

    # '''save checkpoint'''
    # torch.save(trained_model.state_dict(), os.path.join(save_path,output_checkpoint_name))
