# Vocab-Consistent-EEG-To-Text

Vocabulary-consistent evaluation and contrastive learning for EEG-to-text generation on ZuCo. This repository contains the decoding implementation, dataset preprocessing, evaluation and statistical analysis for the associated research article.

## Provenance

Based on [NeuSpeech/EEG-To-Text](https://github.com/NeuSpeech/EEG-To-Text) and [MikeWangWZHL/EEG-To-Text](https://github.com/MikeWangWZHL/EEG-To-Text). Cite Wang & Ji (AAAI 2022), *Open Vocabulary Electroencephalography-to-Text Decoding and Zero-Shot Sentiment Classification*, and Jo et al., *Are EEG-to-Text Models Working?* / *Evaluating EEG-to-text models through noise-based performance analysis* (Scientific Reports, DOI 10.1038/s41598-025-29587-x), alongside the associated article when available.

Upstream licensing terms have not been explicitly established. No blanket open-source license is assigned to inherited code; upstream attribution is retained. Third-party libraries, pretrained models and datasets have their own terms.

## Scope and history

The relevant original Git history is retained, with manuscripts, datasets, checkpoints and generated artifacts removed throughout history. Empty commits are pruned and hashes consequently differ from the working repository. Original author names, commit email addresses and dates are preserved. This is a prepared research release, not a claim of complete end-to-end validation.

## Setup

Use Python 3.9. `environment.yml` installs the original PyTorch 1.9 / CUDA 11.1 stack and the complete direct dependency list in `requirements.txt`. Run the command from the repository root so the requirements path resolves. `environment_working_bart.yaml` records the historical working environment; `environment_original.yml` is upstream provenance, not the recommended setup. GPU drivers must support the selected CUDA runtime.

```bash
conda env create -f environment.yml
conda activate vocab-consistent-eeg-to-text
```

For a CPU-only setup, create a Python 3.9 environment, install PyTorch 1.9 for CPU following the PyTorch installation instructions, then run `python -m pip install -r requirements.txt`. The full BART-large paper model is computationally expensive on CPU.

Activate the intended environment before running Python/shell entry points. Sweep scripts use the active interpreter. Set `WANDB_ENTITY` and optionally `WANDB_PROJECT` for online training/sweeps. Do not commit API keys; authenticate locally with `wandb login`.

## Data

Download the official [ZuCo 1.0](https://osf.io/q3zws/) and [ZuCo 2.0](https://osf.io/2urht/) releases. Place their Matlab files under `dataset/ZuCo/task1-SR/Matlab_files`, `task2-NR/Matlab_files`, `task3-TSR/Matlab_files` if needed, and `task2-NR-2.0/Matlab_files`. Preprocess with:

```bash
bash scripts/prepare_dataset.sh
```

This repository does not redistribute raw EEG, reference-sentence text or prediction dumps. The `dataset_analysis` package provides the additional OOV comparison tooling; see its README for Alice EEG, DERCo and ROAMM sources.

## Final paper runs

| Model | Run | Configuration |
|---|---|---|
| Contrastive | ihig0y4v | configs/contrastive.json |
| Vocabulary-consistent baseline | 00ksermq | configs/vocab.json |
| Corrected literature baseline | mymf64l7 | configs/literature.json |
| Single-subject ZDM | slwo549q | configs/zdm.json |

The three stratified configurations use explicit indices in `splits/`; the literature baseline uses the sequential rule described in `splits/literature_rule.json`. Dataset order is ZuCo 1.0 SR, ZuCo 1.0 NR, ZuCo 2.0 NR. Stored indices precede sample filtering: cohort 11,386/1,341/1,462 and ZDM 598/51/51 train/dev/test positions. Evaluation contains 1,387 stratified-cohort records and 49 ZDM records after filtering.

These configurations were recovered from the original W&B run records, avoiding overwritten loose config files. Contrastive and vocabulary-consistent stored split indices match exactly. Checkpoint hashes and integrity results are in `manifest.json`; weights are not bundled and no download location is claimed yet.

## Recompute reported metrics

The `statistics/` files contain per-record sufficient statistics. Python and NumPy suffice to verify the packaged point estimates without EEG datasets or checkpoints:

```bash
python analysis/reproduce_metrics.py
python analysis/confidence_intervals.py analyse --n-boot 2000 --n-ar 10000 --seed 42
```

Only point estimates were recomputed during release preparation. Confidence intervals/significance can be regenerated with the second command. Preserve per-record order for paired tests. Heavy tokenizer/scoring dependencies are needed only for regenerating sufficient statistics from prediction files, which are not bundled.

## Rerun checkpoint evaluation

Supply a checkpoint listed in `manifest.json`. The following wrapper combines the matching recovered config with its exact stored split and writes an evaluation-only config:

```bash
python scripts/evaluate_paper_model.py --model contrastive --checkpoint /path/to/model.pt --input EEG
python scripts/evaluate_paper_model.py --model contrastive --checkpoint /path/to/model.pt --input noise
```

Use `vocab`, `literature` or `zdm` for the other models. Datasets, compatible dependencies and checkpoints must be supplied separately. GPU inference/training was not rerun during preparation, so the numerical-statistics verification does not establish full end-to-end reproduction.

## Training

After downloading and preprocessing the three paper tasks, train the selected contrastive configuration from the repository root:

```bash
# Disable external experiment logging, or set WANDB_ENTITY for online logging.
export WANDB_MODE=disabled
python train_decoding.py \
  --model_name BrainTranslatorBaseline --task_name task1_task2_taskNRv2 \
  --one_step --pretrained --not_load_step1_checkpoint \
  --num_epoch_step1 20 --num_epoch_step2 30 \
  --learning_rate_step1 0.00002 --learning_rate_step2 0.0004851820064384097 \
  --dropout_rate 0.25306155198637037 --contrastive_lambda 0.329048196487127 \
  --batch_size 8 --subjects ALL --train_input EEG \
  --save_path ./checkpoints/contrastive-retrained
```

BART-large weights/tokenizer are downloaded from Hugging Face on first use, or must already be cached. Training writes generated configs into `config/decoding/` and weights into the requested checkpoint directory. Choose a new output directory for each run. The entry point creates stratified splits using its recorded seed; exact paper splits are preserved in `splits/` for evaluation. Training does not currently accept saved split files as CLI input. A full paper run used an RTX 4080; suitable GPU memory and substantial training time are required. CPU fallback is supported but impractical for full BART-large training.

To check the training loops without a GPU, datasets, model downloads or external logging:

```bash
python scripts/smoke_train.py
```

This runs the real ZuCo sample-loading/batching code and both baseline and contrastive training/validation loops with synthetic features, a tiny tokenizer and a miniature random BART. It checks finite parameters, optimizer updates, checkpoint writing and reload, and removes its temporary outputs. It does not reproduce the paper's metrics or establish full BART-large/GPU readiness.

Core training is in `train_decoding.py`; sweep definitions and runners are under `scripts/`. For the corrected literature baseline use `sweep_litbaseline_btb_config.yaml`, not the superseded experimental-architecture Jo sweep. Configurations describe the original selected runs; retraining may vary across seeds and software/hardware versions.
