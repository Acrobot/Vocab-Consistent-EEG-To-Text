"""CPU check of real dataset batching, baseline/contrastive loops and checkpoints.

Uses synthetic EEG-shaped samples and a miniature random BART model. Does not
validate paper-model convergence or require/download ZuCo or pretrained weights.
"""
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
os.environ["WANDB_MODE"] = "disabled"
os.environ.setdefault("WANDB_SILENT", "true")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import BartConfig, BartForConditionalGeneration
import wandb
from data import ZuCo_dataset
from model_decoding import BrainTranslatorBaseline
import train_decoding as training

class SmokeTokenizer:
    """Deterministic tiny tokenizer for synthetic samples only."""
    pad_token_id = 1
    def __call__(self, text, max_length, **kwargs):
        ids = [0, 3, 4, 5, 2] + [1] * (max_length - 5)
        ids = torch.tensor([ids])
        return {"input_ids": ids, "attention_mask": (ids != 1).long()}

def main():
    torch.set_num_threads(2)
    torch.manual_seed(312)
    rng = np.random.default_rng(312)
    bands = ["_t1", "_t2", "_a1", "_a2", "_b1", "_b2", "_g1", "_g2"]
    samples = []
    for _ in range(3):
        words = [{"content": "test", "word_level_EEG": {"GD": {"GD" + band: rng.normal(size=105) for band in bands}}} for _ in range(3)]
        samples.append({"content": "a test sentence", "word": words,
                        "sentence_level_EEG": {"mean" + band: rng.normal(size=105) for band in bands}})
    split = {"train": {0: {"SMOKE": [0, 1]}}, "dev": {0: {"SMOKE": [2]}}, "test": {0: {"SMOKE": []}}}
    tokenizer = SmokeTokenizer()
    datasets = {phase: ZuCo_dataset([{"SMOKE": samples}], phase, tokenizer,
        subject="SMOKE", test_input="EEG", max_len=8, loaded_split_indices=split) for phase in ["train", "dev"]}
    training.tokenizer = tokenizer
    training.dataset_sizes = {phase: len(ds) for phase, ds in datasets.items()}
    loaders = {phase: DataLoader(ds, batch_size=1, shuffle=False, num_workers=0) for phase, ds in datasets.items()}
    assert training.dataset_sizes == {"train": 2, "dev": 1}
    with tempfile.TemporaryDirectory(prefix="eeg-training-smoke-") as temporary:
        run = wandb.init(mode="disabled", dir=temporary, project="training-smoke")
        for name, penalty in [("baseline", 0.0), ("contrastive", 0.329)]:
            bart = BartForConditionalGeneration(BartConfig(vocab_size=16, d_model=32,
                encoder_layers=1, decoder_layers=1, encoder_attention_heads=4,
                decoder_attention_heads=4, encoder_ffn_dim=64, decoder_ffn_dim=64,
                max_position_embeddings=16, pad_token_id=1, bos_token_id=0, eos_token_id=2,
                decoder_start_token_id=2, dropout=0.0))
            model = BrainTranslatorBaseline(bart, in_feature=840, decoder_embedding_size=32,
                additional_encoder_dim_feedforward=64, dropout=0.0)
            before = model.fc1.weight.detach().clone()
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
            best = str(Path(temporary) / (name + "-best.pt"))
            last = str(Path(temporary) / (name + "-last.pt"))
            kwargs = dict(dataloaders=loaders, device=torch.device("cpu"), model=model,
                criterion=torch.nn.CrossEntropyLoss(), optimizer=optimizer, scheduler=scheduler,
                num_epochs=1, checkpoint_path_best=best, checkpoint_path_last=last)
            if penalty:
                model = training.train_model_contrastive(**kwargs, contrastive_lambda=penalty)
            else:
                model = training.train_model(**kwargs)
            assert not torch.equal(before, model.fc1.weight.detach()), "Optimizer did not update weights"
            assert all(torch.isfinite(p).all() for p in model.parameters()), "Non-finite parameters"
            assert Path(best).stat().st_size > 0 and Path(last).stat().st_size > 0
            # Only load the checkpoint produced by this test, never external pickle files.
            restored = torch.load(best, map_location="cpu")
            model.load_state_dict(restored)
            print(name + ": batching, forward/backward, optimizer, validation and checkpoint reload passed")
        run.finish()
    print("Training smoke check passed; temporary checkpoints removed.")

if __name__ == "__main__":
    main()
