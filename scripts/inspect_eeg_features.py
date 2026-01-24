import pickle
import torch

def normalize_1d(input_tensor):
    mean = torch.mean(input_tensor)
    std = torch.std(input_tensor)
    if std == 0: return input_tensor - mean
    input_tensor = (input_tensor - mean)/std
    return input_tensor

dataset_path = 'dataset/ZuCo/task1-SR/pickle/task1-SR-dataset.pickle'
with open(dataset_path, 'rb') as h:
    data = pickle.load(h)

# Inspect first subject, first sentence
subject = list(data.keys())[0]
sentence = data[subject][0]

print(f"Subject: {subject}")
print(f"Sentence: {sentence['content']}")

# Look at word-level EEG
word = sentence['word'][0]
eeg = word['word_level_EEG']['GD']
bands = ['GD_t1','GD_t2','GD_a1','GD_a2','GD_b1','GD_b2','GD_g1','GD_g2']

features = []
for b in bands:
    feat_np = eeg[b]
    feat = torch.from_numpy(feat_np)
    features.append(feat)
    print(f"Band {b} shape: {feat.shape}, Stats: mean={feat.mean():.4f}, std={feat.std():.4f}, range=[{feat.min():.4f}, {feat.max():.4f}]")

all_feat = torch.cat(features)
norm_feat = normalize_1d(all_feat)

print(f"\nConcatenated feature (8*105={len(all_feat)}) stats:")
print(f"Original: mean={all_feat.mean():.4f}, std={all_feat.std():.4f}")
print(f"Normalized: mean={norm_feat.mean():.4f}, std={norm_feat.std():.4f}")

# Look at multiple words to see variance
print("\nComparing first 5 words' mean/std values (Original):")
for i in range(min(5, len(sentence['word']))):
    w = sentence['word'][i]
    if 'word_level_EEG' not in w or 'GD' not in w['word_level_EEG']:
        continue
    eeg_w = w['word_level_EEG']['GD']
    f_w = torch.cat([torch.from_numpy(eeg_w[b]) for b in bands])
    print(f"Word {i} ('{w['content']}'): mean={f_w.mean():.4f}, std={f_w.std():.4f}")

