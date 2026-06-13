"""Re-run only the deterministic teacher-forcing path and dump per-sentence token
lists, so teacher-forcing BLEU can be reproduced exactly for confidence intervals.

Why this exists: ``eval_decoding.py`` scores teacher-forcing BLEU on the *raw*
argmax token IDs, but the saved ``*_decoding_results.txt`` keeps only the decoded
string. Re-tokenising that string drifts ~1.3 BLEU below the original, so the
published TF numbers are not recoverable from the text files. The free-generation
path *is* recoverable (the eval re-encodes its own decoded string), so only TF
needs re-running.

The TF path is a single forward pass followed by an argmax (no sampling, no beam
search), so it is fully deterministic given the model and inputs. The injected
noise is drawn once at dataset construction (data.get_input_sample), before the
model's random init consumes any RNG -- so we replicate eval_decoding.py's exact
ordering (seed -> tokenizer -> dataset -> model) to reproduce the same noise.

Output: a JSON list (test-set order) of
``{target_tokens, tf_tokens, target_string, tf_string}`` where *_tokens are the
exact BART sub-word token lists fed to corpus_bleu, and *_string mirror the lines
in the saved results file for validation.

    python eval_tf_dump.py --checkpoint_path <ckpt> --config_path <train_cfg.json> \
        --test_input {EEG,noise} --out <tf_tokens.json>
"""

import argparse
import json
import os
import pickle

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BartForConditionalGeneration, BartTokenizer

from data import ZuCo_dataset
from model_decoding import BrainTranslatorBaseline

DATASET_PATHS = {
    "task1": "./dataset/ZuCo/task1-SR/pickle/task1-SR-dataset.pickle",
    "task2": "./dataset/ZuCo/task2-NR/pickle/task2-NR-dataset.pickle",
    "task3": "./dataset/ZuCo/task3-TSR/pickle/task3-TSR-dataset.pickle",
    "taskNRv2": "./dataset/ZuCo/task2-NR-2.0/pickle/task2-NR-2.0-dataset.pickle",
}


def remove_text_after_token(text, token="</s>"):
    """Drop everything from the first </s> on (matches eval_decoding.py)."""
    i = text.find(token)
    return text[:i] if i != -1 else text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-checkpoint", "--checkpoint_path", required=True)
    ap.add_argument("-conf", "--config_path", required=True)
    ap.add_argument("-test_input", "--test_input", required=True)
    ap.add_argument("-train_input", "--train_input", default="EEG")
    ap.add_argument("--out", required=True)
    args = vars(ap.parse_args())

    training_config = json.load(open(args["config_path"]))
    subject_choice = training_config["subjects"]
    eeg_type_choice = training_config["eeg_type"]
    bands_choice = training_config["eeg_bands"]
    task_name = training_config["task_name"]
    model_name = training_config["model_name"]
    dataset_setting = "unique_sent"
    assert model_name == "BrainTranslatorBaseline", f"unexpected model {model_name}"

    # --- seeds: identical to eval_decoding.py so the injected noise matches ---
    seed_val = 312
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed_val)

    device = torch.device(0 if torch.cuda.is_available() else "cpu")

    # --- data (built BEFORE the model, like eval_decoding.py, so the noise draw
    #     happens at the same RNG point) ---
    whole_dataset_dicts = []
    for key in ("task1", "task2", "task3", "taskNRv2"):
        if key in task_name:
            with open(DATASET_PATHS[key], "rb") as handle:
                whole_dataset_dicts.append(pickle.load(handle))

    tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")

    loaded_split_indices = None
    if "split_indices" in training_config:
        split_indices = training_config["split_indices"]
        loaded_split_indices = {
            split_name: {
                int(dataset_idx): subjects_dict
                for dataset_idx, subjects_dict in split_indices[split_name].items()
            }
            for split_name in ("train", "dev", "test")
        }
    use_sequential_split = training_config.get("use_sequential_split", False)
    use_stratified_split = not use_sequential_split

    test_set = ZuCo_dataset(
        whole_dataset_dicts,
        "test",
        tokenizer,
        subject=subject_choice,
        eeg_type=eeg_type_choice,
        bands=bands_choice,
        setting=dataset_setting,
        test_input=args["test_input"],
        loaded_split_indices=loaded_split_indices,
        use_stratified_split=use_stratified_split,
    )
    print(f"[INFO] test_set size: {len(test_set)}", flush=True)
    test_dataloader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=1)

    # --- model (created after the dataset, exactly as in eval_decoding.py) ---
    pretrained_bart = BartForConditionalGeneration.from_pretrained("facebook/bart-large")
    model = BrainTranslatorBaseline(
        pretrained_bart,
        in_feature=105 * len(bands_choice),
        decoder_embedding_size=1024,
        additional_encoder_nhead=8,
        additional_encoder_dim_feedforward=2048,
    )
    state_dict = torch.load(args["checkpoint_path"])
    new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()

    # --- teacher-forcing forward pass + token dump ---
    results = []
    with torch.no_grad():
        for (
            input_embeddings,
            seq_len,
            input_masks,
            input_mask_invert,
            target_ids,
            target_mask,
            sentiment_labels,
        ) in tqdm(test_dataloader):
            input_embeddings_batch = input_embeddings.to(device).float()
            input_masks_batch = input_masks.to(device)
            target_ids_batch = target_ids.to(device)
            input_mask_invert_batch = input_mask_invert.to(device)

            # references, taken before pad ids are masked (matches eval_decoding.py)
            target_tokens = tokenizer.convert_ids_to_tokens(
                target_ids_batch[0].tolist(), skip_special_tokens=True
            )
            target_string = tokenizer.decode(target_ids_batch[0], skip_special_tokens=True)

            target_ids_batch[target_ids_batch == tokenizer.pad_token_id] = -100

            seq2seqLMoutput = model(
                input_embeddings_batch,
                input_masks_batch,
                input_mask_invert_batch,
                target_ids_batch,
            )
            logits_previous = seq2seqLMoutput.logits
            probs_previous = logits_previous[0].softmax(dim=1)
            _, predictions_previous = probs_previous.topk(1)
            predictions_previous = torch.squeeze(predictions_previous)
            tf_string = remove_text_after_token(
                tokenizer.decode(predictions_previous).split("</s></s>")[0].replace("<s>", "")
            )
            predictions_previous = predictions_previous.tolist()
            truncated = []
            for t in predictions_previous:
                if t != tokenizer.eos_token_id:
                    truncated.append(t)
                else:
                    break
            tf_tokens = tokenizer.convert_ids_to_tokens(truncated, skip_special_tokens=True)

            results.append(
                {
                    "target_tokens": target_tokens,
                    "tf_tokens": tf_tokens,
                    "target_string": target_string,
                    "tf_string": tf_string,
                }
            )

    os.makedirs(os.path.dirname(args["out"]), exist_ok=True)
    with open(args["out"], "w") as f:
        json.dump(results, f)
    print(f"[INFO] wrote {len(results)} records to {args['out']}", flush=True)


if __name__ == "__main__":
    main()
