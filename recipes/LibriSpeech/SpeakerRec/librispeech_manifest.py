import os
import hyperpyyaml

# Define the path to the librispeech_prepare.py script
LIBRISPEECH_PREPARE_SCRIPT = "/workspaces/myspeechbrain/recipes/LibriSpeech/SpeakerRec/librispeech_prepare.py"

# Define the path to the LibriSpeech recipes directory
LIBRI_SPEECH_RECIPES_PATH = "/workspaces/myspeechbrain/speechbrain/recipes/LibriSpeech/"

# Define the source data path for LibriSpeech train-clean-100 (as per task description)
SOURCE_DATA_PATH = "/workspaces/dataset/dev-clean/"

# Import the prepare_librispeech function directly
import sys
sys.path.insert(0, "/workspaces/myspeechbrain")  # Add speechbrain root to path
sys.path.insert(0, os.path.dirname(LIBRISPEECH_PREPARE_SCRIPT)) # Add script directory to path
from librispeech_prepare import prepare_librispeech

LIBRISPEECH_HPARAMS_FILE ="/workspaces/myspeechbrain/recipes/LibriSpeech/SpeakerRec/hparams/train_ecapa_tdnn.yaml"

# Define the new target data path for manifests. Moving it outside of SOURCE_DATA_PATH
# to prevent conflicts with librispeech_prepare.py's parsing of speaker directories.
TARGET_DATA_PATH_LIBRISPEECH_MANIFESTS = "/workspaces/dataset_manifests/"

# Ensure the output directory for the manifest files exists
print(f"Creating directory if not exists: {TARGET_DATA_PATH_LIBRISPEECH_MANIFESTS}")
#os.mkdir -p "{TARGET_DATA_PATH_LIBRISPEECH_MANIFESTS}"
os.makedirs(TARGET_DATA_PATH_LIBRISPEECH_MANIFESTS,exist_ok=True)

# Load the hyperparameters from the LIBRISPEECH_HPARAMS_FILE to extract sentence_len
# Pass SOURCE_DATA_PATH as an override for the 'data_folder' placeholder,
# assuming it exists and is required by the hparams file
print(f"Loading hparams from: {LIBRISPEECH_HPARAMS_FILE} with data_folder override: {SOURCE_DATA_PATH}")
with open(LIBRISPEECH_HPARAMS_FILE) as fin:
    hparams_libri = hyperpyyaml.load_hyperpyyaml(fin, overrides={'data_folder': SOURCE_DATA_PATH})

# Extract the sentence_len value (Note: This is used for filtering data *later*, not by prepare_librispeech itself)
sentence_len_libri = hparams_libri.get('sentence_len', 4.0) # Default to 4.0 if not found
print(f"Extracted sentence_len from LibriSpeech hparams: {sentence_len_libri}")

# Call the prepare_librispeech function directly
print("Generating manifest files using librispeech_prepare.py...")
prepare_librispeech(
    data_folder=SOURCE_DATA_PATH,
    save_folder=TARGET_DATA_PATH_LIBRISPEECH_MANIFESTS
)

# Verify that the manifest files have been created
print(f"Verifying manifest files in: {TARGET_DATA_PATH_LIBRISPEECH_MANIFESTS}")
#ls -F "{TARGET_DATA_PATH_LIBRISPEECH_MANIFESTS}"