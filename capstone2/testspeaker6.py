import torch
import os
import warnings
import logging
import random
import numpy as np
import matplotlib.pyplot as plt
import itertools
from speechbrain.inference.speaker import EncoderClassifier
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import roc_curve, auc

# 1. Silence warnings and logs
warnings.filterwarnings("ignore")
logging.getLogger("speechbrain").setLevel(logging.ERROR)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# 2. Setup Paths
YAML_DIR = "C:/coding/SpeechBrain/SpeakerRec_ecapa_tdnn"
CKPT_DIR = "C:/coding/SpeechBrain/SpeakerRec_ecapa_tdnn/save/CKPT+2026-03-14+10-48-44+00"
DATASET_DIR = "C:/coding/SpeechBrain/train-clean-wb"
PLOT_OUTPUT = "C:/coding/SpeechBrain/roc_curve.png"

# 3. Load Model
print("Loading model...")
classifier = EncoderClassifier.from_hparams(
    source=CKPT_DIR,
    hparams_file=os.path.join(YAML_DIR, "hyperparams.yaml"),
    savedir=CKPT_DIR,
    overrides={"mean_var_norm": None}
)

def collect_audio_files(speaker_path):
    audio_files = []
    for root, _, files in os.walk(speaker_path):
        for f in files:
            if f.endswith('.flac'):
                audio_files.append(os.path.join(root, f))
    return audio_files

def get_embedding(path):
    signal = classifier.load_audio(path)
    embeddings = classifier.encode_batch(signal)
    return embeddings.squeeze().cpu().numpy().flatten()

# 4. Prepare Speakers list
speakers = [s for s in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, s))]
print(f"Found {len(speakers)} speakers. Starting evaluation...")

all_scores = []
all_labels = []
num_negative_samples = 5 

# 5. Evaluation Loop
for i, spk in enumerate(speakers):
    spk_path = os.path.join(DATASET_DIR, spk)
    files = collect_audio_files(spk_path)
    
    if len(files) < 2: continue

    # Precompute all embeddings for the current speaker to save time
    embeddings = [get_embedding(f) for f in files]

    # Positive Pairs: Test all unique combinations of files for this speaker
    for emb1, emb2 in itertools.combinations(embeddings, 2):
        score_pos = cosine_similarity([emb1], [emb2])[0][0]
        all_scores.append(score_pos)
        all_labels.append(1)

    # Negative Pairs: Compare each embedding of this speaker to N other random speakers
    other_indices = [idx for idx in range(len(speakers)) if idx != i]
    
    for emb in embeddings:
        sampled_others = random.sample(other_indices, min(num_negative_samples, len(other_indices)))
        
        for other_idx in sampled_others:
            other_spk_path = os.path.join(DATASET_DIR, speakers[other_idx])
            other_files = collect_audio_files(other_spk_path)
            if other_files:
                emb_other = get_embedding(random.choice(other_files))
                score_neg = cosine_similarity([emb], [emb_other])[0][0]
                all_scores.append(score_neg)
                all_labels.append(0)

    if (i + 1) % 10 == 0:
        print(f"Processed {i+1}/{len(speakers)} speakers...")

# 6. Calculate Metrics
fpr, tpr, thresholds = roc_curve(all_labels, all_scores)
roc_auc = auc(fpr, tpr)

fnr = 1 - tpr
eer_idx = np.nanargmin(np.absolute((fnr - fpr)))
eer = fpr[eer_idx]
eer_threshold = thresholds[eer_idx]

print(f"\n--- Final Metrics ---")
print(f"ROC AUC: {roc_auc:.4f}")
print(f"EER: {eer:.4f} ({eer*100:.2f}%)")
print(f"Optimal Threshold: {eer_threshold:.4f}")

# 7. Plotting the ROC Curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--') # Diagonal random line
plt.scatter(eer, 1-eer, color='red', label=f'EER = {eer*100:.2f}%') # Mark EER point

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (FAR)')
# FNR = 1 - TPR, but ROC usually plots True Positive Rate
plt.ylabel('True Positive Rate (1 - FRR)')
plt.title('Receiver Operating Characteristic (ROC) - Speaker Verification')
plt.legend(loc="lower right")
plt.grid(alpha=0.3)

plt.savefig(PLOT_OUTPUT)
print(f"ROC Curve saved to: {PLOT_OUTPUT}")
plt.show()
