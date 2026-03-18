import torchaudio
import torch
import os


from speechbrain.inference.speaker import SpeakerRecognition

YAML_DIR = "/home/nilezt/speechBrain/SpeakerRec_ecapa_tdnn"
CKPT_DIR = "/home/nilezt/speechBrain/SpeakerRec_ecapa_tdnn/save/CKPT+2026-03-08+10-02-51+00"

verification = SpeakerRecognition.from_hparams(
    source=CKPT_DIR,
    hparams_file=os.path.join(YAML_DIR, "hyperparams.yaml"),
    savedir=CKPT_DIR,
    # This prevents the script from looking for the missing .ckpt file
    overrides={"mean_var_norm": None}
)

# Test Files
SPKR1_AUDIO1 = "/home/nilezt/speechBrain/100/121669/100-121669-0000.flac"
SPKR1_AUDIO2 = "/home/nilezt/speechBrain/100/121669/100-121669-0001.flac"

SPKR2_AUDIO1 = "/home/nilezt/speechBrain/122/121729/122-121729-0000.flac"
SPKR2_AUDIO2 = "/home/nilezt/speechBrain/122/121729/122-121729-0001.flac"


# 2. Pass the threshold into the verify_files method
score, prediction = verification.verify_files(
    SPKR1_AUDIO2, 
    SPKR2_AUDIO2
)


print(f"\n--- RESULTS ---")
print(f"Similarity Score: {score.item():.4f}")
print(f"Same Speaker: {prediction.item()}")
