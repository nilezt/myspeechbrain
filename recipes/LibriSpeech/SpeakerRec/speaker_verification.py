#!/usr/bin/python3
"""
Speaker Verification on FULL train-clean-100 + train-clean (total 291 unseen speakers)
Auto-creates CSV + loads your exact 360h checkpoint
Metrics: EER, AUC, F1@EER, minDCF
"""

import os
import random
import argparse
import torch
import numpy as np
import pandas as pd
import torchaudio
from tqdm import tqdm
from collections import defaultdict
from sklearn.metrics import roc_auc_score, roc_curve, f1_score

import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml
from speechbrain.utils.metric_stats import minDCF

# ====================== AUTO CREATE FULL CSV (no split) ======================
def create_full_librispeech_csv(data_root, subset="train-clean-100"):
    base = os.path.join(data_root, "LibriSpeech", subset)
    if not os.path.exists(base):
        raise FileNotFoundError(f"Cannot find {base}. Please put LibriSpeech/train-clean-100 inside {data_root}")

    csv_path = os.path.join("/tmp", f"full_{subset}.csv")
    if os.path.exists(csv_path):
        print(f"Using existing full CSV: {csv_path}")
        return csv_path

    print(f"Creating full annotation CSV from {base} ...")
    rows = []
    for speaker in sorted(os.listdir(base)):
        if not speaker.isdigit():
            continue
        spk_path = os.path.join(base, speaker)
        for chapter in sorted(os.listdir(spk_path)):
            chap_path = os.path.join(spk_path, chapter)
            for fname in sorted(os.listdir(chap_path)):
                if fname.endswith(".flac"):
                    wav = os.path.join(chap_path, fname)
                    info = torchaudio.info(wav)
                    duration = info.num_frames / info.sample_rate
                    utt_id = f"{speaker}-{chapter}-{fname[:-5]}"
                    rows.append({
                        "ID": utt_id,
                        "wav": wav,
                        "spk_id": speaker,
                        "duration": duration
                    })

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    print(f"✅ Created full CSV with {len(df)} utterances → {csv_path}")
    return csv_path


# ====================== AUDIO PIPELINE (same as training) ======================
def audio_pipeline(wav, duration, hparams):
    snt_len_sample = int(hparams["sample_rate"] * hparams["sentence_len"])
    duration_sample = int(duration * hparams["sample_rate"])

    if hparams.get("random_chunk", True):
        start = random.randint(0, max(0, duration_sample - snt_len_sample))
        stop = start + snt_len_sample
    else:
        start = 0
        stop = snt_len_sample

    num_frames = stop - start
    if stop > duration_sample:
        sig, _ = torchaudio.load(wav)
        while sig.shape[1] < num_frames:
            sig = torch.cat([sig, sig], dim=1)
        sig = sig[:, :num_frames]
    else:
        sig, _ = torchaudio.load(wav, num_frames=num_frames, frame_offset=start)

    return sig.squeeze(0)


# ====================== MAIN ======================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hparams_file", type=str, required=True)
    parser.add_argument("--ckpt_path", type=str, required=True, help="Full path to your best .ckpt file")
    parser.add_argument("--data_root", type=str, default="/content/SpeechBrain/dataset")
    parser.add_argument("--num_trials", type=int, default=10000)
    parser.add_argument("--add_wham_noise", action="store_true")
    parser.add_argument("--wham_dir", type=str, default=None)
    args = parser.parse_args()

    # Load hyperparams
    with open(args.hparams_file, encoding="utf-8") as f:
        hparams = load_hyperpyyaml(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hparams["random_chunk"] = False  # deterministic for verification

    # Load modules
    compute_features = hparams["modules"]["compute_features"]
    mean_var_norm = hparams["modules"]["mean_var_norm"]
    embedding_model = hparams["modules"]["embedding_model"]

    # Load your checkpoint (works with any SpeechBrain .ckpt)
    print(f"Loading checkpoint: {args.ckpt_path}")
    ckpt = torch.load(args.ckpt_path, map_location=device)
    for name, module in hparams["modules"].items():
        if name in ckpt:
            module.load_state_dict(ckpt[name])
        elif "modules" in ckpt and name in ckpt["modules"]:
            module.load_state_dict(ckpt["modules"][name])
    for m in hparams["modules"].values():
        m.to(device)
        m.eval()

    # Create full CSV for train-clean-100
    csv_path = create_full_librispeech_csv(args.data_root)

    df = pd.read_csv(csv_path)

    # Extract embeddings
    emb_dict = {}
    spk_dict = {}
    print("Extracting embeddings from ALL train-clean-100 utterances...")
    with torch.no_grad():
        for _, row in tqdm(df.iterrows(), total=len(df)):
            sig = audio_pipeline(row["wav"], row["duration"], hparams)
            if args.add_wham_noise and args.wham_dir:
                sig = add_wham_noise(sig, hparams, args.wham_dir)  # defined below

            sig = sig.unsqueeze(0).to(device)
            feats = compute_features(sig)
            feats = mean_var_norm(feats, torch.ones(1, device=device))
            emb = embedding_model(feats).squeeze(0).cpu().numpy()

            emb_dict[row["ID"]] = emb
            spk_dict[row["ID"]] = row["spk_id"]

    # Generate trials
    spk_to_utts = defaultdict(list)
    for utt, spk in spk_dict.items():
        spk_to_utts[spk].append(utt)

    print(f"Generating {args.num_trials} target + {args.num_trials} impostor trials...")
    random.seed(42)
    speakers = list(spk_to_utts.keys())
    pairs = []
    for _ in range(args.num_trials):
        # Target
        spk = random.choice(speakers)
        utts = spk_to_utts[spk]
        if len(utts) >= 2:
            u1, u2 = random.sample(utts, 2)
            pairs.append((u1, u2, 1))
        # Impostor
        spk1, spk2 = random.sample(speakers, 2)
        u1 = random.choice(spk_to_utts[spk1])
        u2 = random.choice(spk_to_utts[spk2])
        pairs.append((u1, u2, 0))

    random.shuffle(pairs)

    # Compute scores
    scores = []
    labels = []
    print("Computing cosine similarities...")
    for u1, u2, lab in tqdm(pairs):
        sim = np.dot(emb_dict[u1], emb_dict[u2]) / (np.linalg.norm(emb_dict[u1]) * np.linalg.norm(emb_dict[u2]) + 1e-6)
        scores.append(sim)
        labels.append(lab)

    scores = np.array(scores)
    labels = np.array(labels)

    # ====================== METRICS ======================
    auc = roc_auc_score(labels, scores)

    fpr, tpr, thresholds = roc_curve(labels, scores)
    eer = (fpr[np.nanargmin(np.abs(fpr - (1 - tpr)))] + (1 - tpr[np.nanargmin(np.abs(fpr - (1 - tpr)))])) / 2
    eer_threshold = thresholds[np.nanargmin(np.abs(fpr - (1 - tpr)))]
    pred = (scores >= eer_threshold).astype(int)
    f1 = f1_score(labels, pred)

    pos_scores = scores[labels == 1]
    neg_scores = scores[labels == 0]
    mindcf = minDCF(pos_scores, neg_scores, p_target=0.01)

    print("\n" + "="*60)
    print("SPEAKER VERIFICATION RESULTS (train-clean-100 — unseen speakers)")
    print("="*60)
    print(f"EER          : {eer*100:.3f} %")
    print(f"AUC          : {auc:.4f}")
    print(f"F1 @ EER     : {f1:.4f}")
    print(f"minDCF       : {mindcf:.4f}  (p_target=0.01)")
    print(f"Threshold    : {eer_threshold:.4f}")
    print("="*60)