import torch
import torch.nn as nn
from model_decoding import BrainTranslator
from transformers import BartForConditionalGeneration

# Constants to match training config
IN_FEATURE = 840
DEC_EMB_SIZE = 1024

# Path to a "best" checkpoint
checkpoint_path = 'checkpoints/decoding_fixed/best/task1_task2_taskNRv2_finetune_BrainTranslator_skipstep1_b8_20_30_2e-05_1.785e-05_unique_sent_EEG.pt'

def analyze():
    # Load pretrained BART to initialize BrainTranslator
    # (Doesn't need to be the actual trained BART weights for this structural test)
    pretrained = BartForConditionalGeneration.from_pretrained('facebook/bart-large')
    model = BrainTranslator(pretrained, in_feature=IN_FEATURE, decoder_embedding_size=DEC_EMB_SIZE)

    # Load state dict
    try:
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        # Handle DataParallel prefix
        new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(new_state_dict)
        print(f"Successfully loaded checkpoint: {checkpoint_path}")
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return

    # Check additional_encoder weights
    print("\n--- Additional Encoder Weights ---")
    for name, param in model.additional_encoder.named_parameters():
        if 'weight' in name:
            print(f"{name}: mean={param.data.mean():.6f}, std={param.data.std():.6f}, max={param.data.max():.6f}, min={param.data.min():.6f}")

    # Check fc1 weights
    print("\n--- FC1 Weights ---")
    print(f"fc1.weight: mean={model.fc1.weight.data.mean():.6f}, std={model.fc1.weight.data.std():.6f}")
    print(f"fc1.bias: mean={model.fc1.bias.data.mean():.6f}, std={model.fc1.bias.data.std():.6f}")

    # Test with random input
    dummy_input = torch.randn(1, 10, IN_FEATURE)
    dummy_mask = torch.zeros(1, 10).bool()
    
    with torch.no_grad():
        encoded = model.addin_forward(dummy_input, dummy_mask)
        print("\n--- Addin Forward Output (ReLU activated) ---")
        print(f"Encoded shape: {encoded.shape}")
        print(f"Encoded stats: mean={encoded.mean():.6f}, std={encoded.std():.4f}, zeros={(encoded == 0).sum().item()}/{encoded.numel()} ({(encoded == 0).sum().item()/encoded.numel():.1%})")

analyze()
