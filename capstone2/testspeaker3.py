import torch
import os
import warnings
import logging
import numpy as np
from speechbrain.inference.speaker import EncoderClassifier
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import roc_curve, auc

# 1. Silence warnings
warnings.filterwarnings("ignore")
logging.getLogger("speechbrain").setLevel(logging.ERROR)

# 2. Setup Paths
YAML_DIR = r"C:\coding\SpeechBrain\SpeakerRec_ecapa_tdnn"
CKPT_DIR = r"C:\coding\SpeechBrain\SpeakerRec_ecapa_tdnn\save\CKPT+2026-03-14+10-48-44+00"
DATASET_DIR = "C:/coding/SpeechBrain/test-clean-wb"

# 3. Load Model
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
    # Ensure it returns a 1D array
    return embeddings.squeeze().cpu().numpy().flatten()

# 4. Prepare Speakers list (This was missing!)
speakers = [s for s in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, s))]
print(f"Found {len(speakers)} speakers. Starting evaluation...")

all_scores = []
all_labels = []

# 5. Evaluation Loop
for i, spk in enumerate(speakers):
    spk_path = os.path.join(DATASET_DIR, spk)
    files = collect_audio_files(spk_path)
    
    if len(files) < 2: continue

    # Positive Pair (Same Speaker)
    emb1 = get_embedding(files[0])
    emb2 = get_embedding(files[1])
    score_pos = cosine_similarity([emb1], [emb2])[0][0]
    all_scores.append(score_pos)
    all_labels.append(1)

    # Negative Pair (Different Speaker)
    other_idx = (i + 1) % len(speakers)
    other_spk_path = os.path.join(DATASET_DIR, speakers[other_idx])
    other_files = collect_audio_files(other_spk_path)
    
    if other_files:
        emb_other = get_embedding(other_files[0])
        score_neg = cosine_similarity([emb1], [emb_other])[0][0]
        all_scores.append(score_neg)
        all_labels.append(0)

# 6. Calculate AUC and EER
fpr, tpr, thresholds = roc_curve(all_labels, all_scores, pos_label=1)
roc_auc = auc(fpr, tpr)

# EER calculation
fnr = 1 - tpr
eer_threshold = thresholds[np.nanargmin(np.absolute((fnr - fpr)))]
eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]

print(f"\n--- Metrics ---")
print(f"Total Trials: {len(all_scores)}")
print(f"ROC AUC: {roc_auc:.4f}")
print(f"EER: {eer:.4f} ({eer*100:.2f}%)")
print(f"Optimal EER Threshold: {eer_threshold:.4f}")
