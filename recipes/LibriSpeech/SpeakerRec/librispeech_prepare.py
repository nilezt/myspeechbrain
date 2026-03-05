import os
import random
import soundfile as sf
import glob
import logging

logger = logging.getLogger(__name__)

def prepare_librispeech(data_folder, save_folder, splits=["train", "dev", "test"], split_ratio=[80, 10, 10], seed=1234):
    """
    Prepares the CSV files for the Librispeech dataset.
    
    Arguments:
    data_folder : str
        Path to the folder where the original Librispeech dataset is stored.
    save_folder : str
        The directory where to store the csv files.
    splits : list
        List of splits to create. Default: ["train", "dev", "test"]
    split_ratio : list
        Ratio of data for each split. Default: [80, 10, 10]
    seed : int
        Seed for reproducibility.
    """

    # Create save folder if needed
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    data = []
    
    # Check if data_folder exists
    if not os.path.exists(data_folder):
        raise ValueError(f"Data folder {data_folder} does not exist.")

    speakers = [d for d in os.listdir(data_folder) if os.path.isdir(os.path.join(data_folder, d))]
    if len(speakers) == 0:
         raise ValueError(f"No speakers found in {data_folder}. Check path.")
         
    # Check if this phase is already done (if so, skip)
    # We check for the first split only
    if os.path.isfile(os.path.join(save_folder, splits[0] + ".csv")):
        logger.info("Librispeech CSVs found, skipping data prep.")
        return len(speakers)

    logger.info(f"Scanning {data_folder}...")

    # LibriSpeech structure: SpeakerID / ChapterID / filename.flac
    # We expect data_folder to point to e.g. .../dev-clean
    
    for speaker_id in speakers:
        speaker_path = os.path.join(data_folder, speaker_id)
            
        for chapter_id in os.listdir(speaker_path):
            chapter_path = os.path.join(speaker_path, chapter_id)
            if not os.path.isdir(chapter_path): continue
                
            for file in os.listdir(chapter_path):
                if file.endswith(".flac") or file.endswith(".wav"):
                    # Found an audio file!
                    full_path = os.path.join(chapter_path, file)
                    
                    info = sf.info(full_path)
                    duration = info.duration
                    
                    # Create unique ID and entry
                    file_id = os.path.splitext(file)[0]
                    # CSV format: ID, duration, wav, spk_id
                    data.append(f"{file_id},{duration:.2f},{full_path},{speaker_id}")

    # Shuffle and Split
    random.seed(seed)
    random.shuffle(data)
    
    n = len(data)
    if sum(split_ratio) != 100:
        raise ValueError("Split ratios must sum to 100")
        
    n_train = int(n * split_ratio[0] / 100)
    n_dev = int(n * split_ratio[1] / 100)
    
    splits_data = {}
    splits_data["train"] = data[:n_train]
    splits_data["dev"] = data[n_train:n_train+n_dev]
    splits_data["test"] = data[n_train+n_dev:]

    header = "ID,duration,wav,spk_id"
    
    for i, split_name in enumerate(splits):
        if i == 0: lines = splits_data["train"]
        elif i == 1: lines = splits_data["dev"]
        elif i == 2: lines = splits_data["test"]
        else: break
        
        out_path = os.path.join(save_folder, split_name + ".csv")
        with open(out_path, "w") as f:
            f.write(header + "\n")
            f.write("\n".join(lines))
        logger.info(f"Saved {split_name}: {len(lines)} files")
        
    return len(speakers)
