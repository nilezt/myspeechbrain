import torch
import os
import warnings
import logging
import random
import numpy as np
from speechbrain.inference.speaker import EncoderClassifier
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import roc_curve, auc

# 1. Silence warnings and logs for a clean output
warnings.filterwarnings("ignore")
logging.getLogger("speechbrain").setLevel(logging.ERROR)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# 2. Setup Paths - Using forward slashes for Windows compatibility
YAML_DIR = "C:/coding/SpeechBrain/SpeakerRec_ecapa_tdnn"
CKPT_DIR = "C:/coding/SpeechBrain/SpeakerRec_ecapa_tdnn/save/CKPT+2026-03-14+10-48-44+00"
DATASET_DIR = "C:/coding/SpeechBrain/test-clean-wb"

# 3. Load Model
print("Loading model...")
classifier = EncoderClassifier.from_hparams(
    source=CKPT_DIR,
    hparams_file=os.path.join(YAML_DIR, "hyperparams.yaml"),
    savedir=CKPT_DIR,
    overrides={"mean_var_norm": None}
)

def collect_audio_files(speaker_path):
    """Recursively finds all .flac files in LibriSpeech folder structure."""
    audio_files = []
    for root, _, files in os.walk(speaker_path):
        for f in files:
            if f.endswith('.flac'):
                audio_files.append(os.path.join(root, f))
    return audio_files

def get_embedding(path):
    """Loads audio and extracts a flattened 1D embedding vector."""
    signal = classifier.load_audio(path)
    embeddings = classifier.encode_batch(signal)
    return embeddings.squeeze().cpu().numpy().flatten()

# 4. Prepare Speakers list
speakers = [s for s in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, s))]
print(f"Found {len(speakers)} speakers. Starting evaluation...")

all_scores = []
all_labels = []
num_negative_samples = 5  # Each speaker is compared against 5 different random speakers

# 5. Evaluation Loop
for i, spk in enumerate(speakers):
    spk_path = os.path.join(DATASET_DIR, spk)
    files = collect_audio_files(spk_path)
    
    if len(files) < 2: 
        continue

    # A. Positive Pair (Same Speaker - Label: 1)
    # Compare first two unique files of the same speaker
    emb1 = get_embedding(files[0])
    emb2 = get_embedding(files[1])
    score_pos = cosine_similarity([emb1], [emb2])[0][0]
    all_scores.append(score_pos)
    all_labels.append(1)

    # B. Negative Pairs (Different Speakers - Label: 0)
    # Pick N random speakers that are NOT the current speaker
    other_indices = [idx for idx in range(len(speakers)) if idx != i]
    sampled_others = random.sample(other_indices, min(num_negative_samples, len(other_indices)))
    
    for other_idx in sampled_others:
        other_spk_path = os.path.join(DATASET_DIR, speakers[other_idx])
        other_files = collect_audio_files(other_spk_path)
        
        if other_files:
            # Compare original speaker (emb1) with a random file from a different speaker
            emb_other = get_embedding(random.choice(other_files))
            score_neg = cosine_similarity([emb1], [emb_other])[0][0]
            all_scores.append(score_neg)
            all_labels.append(0)

    # Progress tracking
    if (i + 1) % 10 == 0:
        print(f"Processed {i+1}/{len(speakers)} speakers...")

# 6. Calculate Metrics (AUC and EER)
all_scores = np.array(all_scores)
all_labels = np.array(all_labels)

fpr, tpr, thresholds = roc_curve(all_labels, all_scores, pos_label=1)
roc_auc = auc(fpr, tpr)

# EER calculation: where False Positive Rate (FPR) equals False Negative Rate (FNR)
fnr = 1 - tpr
eer_idx = np.nanargmin(np.absolute((fnr - fpr)))
eer = fpr[eer_idx]
eer_threshold = thresholds[eer_idx]

print(f"\n--- Final Performance Metrics ---")
print(f"Total Trials: {len(all_scores)} (Pos: {sum(all_labels)}, Neg: {len(all_labels)-sum(all_labels)})")
print(f"ROC AUC: {roc_auc:.4f}")
print(f"EER: {eer:.4f} ({eer*100:.2f}%)")
print(f"Optimal EER Threshold: {eer_threshold:.4f}")
