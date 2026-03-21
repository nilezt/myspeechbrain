import torch
import os
import warnings
import logging
from speechbrain.inference.speaker import EncoderClassifier
from sklearn.metrics.pairwise import cosine_similarity

# 1. Silencing the noise
warnings.filterwarnings("ignore")
logging.getLogger("speechbrain").setLevel(logging.ERROR)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" 

YAML_DIR = r"C:\coding\SpeechBrain\SpeakerRec_ecapa_tdnn"
CKPT_DIR = r"C:\coding\SpeechBrain\SpeakerRec_ecapa_tdnn\save\CKPT+2026-03-14+10-48-44+00"
DATASET_DIR = "C:/coding/SpeechBrain/test-clean-wb"

# Load Model
classifier = EncoderClassifier.from_hparams(
    source=CKPT_DIR,
    hparams_file=os.path.join(YAML_DIR, "hyperparams.yaml"),
    savedir=CKPT_DIR,
    overrides={"mean_var_norm": None}
)

def collect_audio_files(speaker_path):
    """Deep search for .flac files in LibriSpeech subfolders"""
    audio_files = []
    for root, _, files in os.walk(speaker_path):
        for f in files:
            if f.endswith('.flac'):
                audio_files.append(os.path.join(root, f))
    return audio_files

def get_embedding(path):
    signal = classifier.load_audio(path)
    embeddings = classifier.encode_batch(signal)
    return embeddings[0].squeeze().cpu().numpy()

# Setup data
speakers = [s for s in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, s))]
print(f"Found {len(speakers)} speakers.")

tp, fp, tn, fn = 0, 0, 0, 0
threshold = 0.25 # Adjust based on your model's performance

print("Processing...")
for i, spk in enumerate(speakers):
    spk_path = os.path.join(DATASET_DIR, spk)
    files = collect_audio_files(spk_path)
    
    if len(files) < 2: continue

    # A. Same Speaker Test
    emb1 = get_embedding(files[0])
    emb2 = get_embedding(files[1])
    score = cosine_similarity([emb1], [emb2])[0][0]
    
    print(f"Speaker {spk} - Similarity Score: {score:.4f}")
    print(f"Speaker {spk} - Prediction: {'Same Speaker' if score >= threshold else 'Different Speakers'}")


    if score >= threshold: tp += 1
    else: fn += 1

    # B. Different Speaker Test
    other_spk = speakers[(i + 1) % len(speakers)]
    other_files = collect_audio_files(os.path.join(DATASET_DIR, other_spk))
    
    if other_files:
        emb_other = get_embedding(other_files[0])
        score_neg = cosine_similarity([emb1], [emb_other])[0][0]

        print(f"Speaker {spk} vs {other_spk} - Similarity Score: {score_neg:.4f}")
        print(f"Speaker {spk} vs {other_spk} - Prediction: {'Same Speaker' if score_neg >= threshold else 'Different Speakers'}")   


        if score_neg < threshold: tn += 1
        else: fp += 1

print(f"\n--- Results at Threshold {threshold} ---")
print(f"TP: {tp} | TN: {tn} | FP: {fp} | FN: {fn}")
