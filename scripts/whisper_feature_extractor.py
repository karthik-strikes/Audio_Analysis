#!/usr/bin/env python3
"""
Whisper Feature Extraction Pipeline

This script extracts audio features from MP3 files using OpenAI's Whisper model.
The extracted features are encoder embeddings from the Whisper model, which capture
rich acoustic and linguistic representations of speech.

Features:
- Processes MP3 files from a specified directory
- Extracts Whisper encoder features (last_hidden_state)
- Saves features as NumPy arrays (.npy files)
- Tracks processed files in CSV to avoid reprocessing
- Uses librosa for audio preprocessing and resampling

Output:
- Feature files: {filename}_whisper_features.npy
- Tracking CSV: processed_wm_files.csv

Author: Audio Feature Extraction Team
Date: 2024
"""

import os
import csv
from typing import Tuple, Optional
from glob import glob

import numpy as np
import pandas as pd
import torch
import librosa
from transformers import WhisperProcessor, WhisperModel


# ============================================================================
# CONFIGURATION
# ============================================================================

# Whisper model configuration
WHISPER_MODEL_NAME = "openai/whisper-medium"

# Directory paths
WHISPER_FEATURES_DIR = "/nlp/data/karthik9/sandata/whisper_large"
MP3_FILES_DIR = "/nlp/data/karthik9/New_audio"
PROCESSED_FILES_CSV = "/nlp/data/karthik9/processed_wm_files.csv"

# Audio preprocessing
TARGET_SAMPLE_RATE = 16000  # Hz


# ============================================================================
# FILE TRACKING FUNCTIONS
# ============================================================================

def is_file_processed(filename: str, csv_path: str) -> bool:
    """
    Check if a file has already been processed by looking in the tracking CSV.

    Args:
        filename (str): Name of the file to check
        csv_path (str): Path to the CSV tracking file

    Returns:
        bool: True if file is already processed, False otherwise
    """
    if not os.path.exists(csv_path):
        return False

    try:
        with open(csv_path, mode='r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                if row.get('Filename') == filename:
                    return True
    except Exception as e:
        print(f"Warning: Error reading CSV file: {e}")

    return False


def add_processed_file(filename: str, features_path: str, csv_path: str) -> None:
    """
    Add a processed file entry to the tracking CSV.

    Args:
        filename (str): Name of the processed file
        features_path (str): Path to the saved features file
        csv_path (str): Path to the CSV tracking file
    """
    file_exists = os.path.exists(csv_path)

    with open(csv_path, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=['Filename', 'FeaturesFilePath'])

        # Write header if file is new
        if not file_exists:
            writer.writeheader()

        writer.writerow({
            'Filename': filename,
            'FeaturesFilePath': features_path
        })


# ============================================================================
# AUDIO PREPROCESSING
# ============================================================================

def preprocess_audio(audio_path: str, target_sr: int = TARGET_SAMPLE_RATE) -> Tuple[torch.Tensor, int]:
    """
    Load and preprocess audio file with resampling.

    Args:
        audio_path (str): Path to audio file
        target_sr (int): Target sample rate in Hz (default: 16000)

    Returns:
        Tuple[torch.Tensor, int]: (waveform tensor, sample rate)

    Raises:
        FileNotFoundError: If audio file doesn't exist
        Exception: If audio loading fails
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    print(f"  Loading audio: {os.path.basename(audio_path)}")

    # Load audio with librosa (automatically resamples to target_sr)
    waveform, sample_rate = librosa.load(audio_path, sr=target_sr)

    # Convert to tensor and add batch dimension
    waveform_tensor = torch.tensor(waveform).unsqueeze(0)

    print(f"  ✓ Audio loaded: {waveform_tensor.shape[1]} samples at {sample_rate} Hz")

    return waveform_tensor, sample_rate


# ============================================================================
# FEATURE EXTRACTION
# ============================================================================

def extract_whisper_features(
    waveform: torch.Tensor,
    sample_rate: int,
    model_name: str = WHISPER_MODEL_NAME
) -> np.ndarray:
    """
    Extract features from audio waveform using Whisper encoder.

    This function loads the Whisper model, processes the audio, and extracts
    the encoder's last hidden state, which contains rich acoustic-linguistic features.

    Args:
        waveform (torch.Tensor): Audio waveform tensor (1, num_samples)
        sample_rate (int): Sample rate of the waveform
        model_name (str): Hugging Face model identifier

    Returns:
        np.ndarray: Extracted features of shape (sequence_length, hidden_size)

    Note:
        - For whisper-medium: hidden_size = 1024
        - sequence_length depends on audio duration
    """
    print("  Loading Whisper model...")
    processor = WhisperProcessor.from_pretrained(model_name)
    model = WhisperModel.from_pretrained(model_name)
    model.eval()  # Set to evaluation mode

    print("  Processing audio input...")
    # Convert waveform to input features
    input_features = processor(
        waveform.squeeze().numpy(),
        sampling_rate=sample_rate,
        return_tensors="pt"
    ).input_features

    print("  Extracting features...")
    with torch.no_grad():
        # Get encoder outputs
        encoder_outputs = model.get_encoder()(input_features)
        features = encoder_outputs.last_hidden_state

    print(f"  ✓ Features extracted: shape {features.shape}")

    # Convert to NumPy and remove batch dimension
    features_np = features.squeeze(0).numpy()

    return features_np


def save_features(features: np.ndarray, filename: str, output_dir: str) -> str:
    """
    Save extracted features to a NumPy file.

    Args:
        features (np.ndarray): Extracted features
        filename (str): Base filename (without extension)
        output_dir (str): Directory to save features

    Returns:
        str: Path to saved features file

    Raises:
        OSError: If directory creation or file saving fails
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Create output file path
    output_filename = f"{filename}_whisper_features.npy"
    output_path = os.path.join(output_dir, output_filename)

    # Save features
    np.save(output_path, features)

    print(f"  ✓ Features saved: {output_path}")

    return output_path


# ============================================================================
# MAIN PROCESSING PIPELINE
# ============================================================================

def process_single_file(
    audio_path: str,
    filename: str,
    output_dir: str,
    csv_path: str
) -> bool:
    """
    Process a single audio file: extract and save Whisper features.

    Args:
        audio_path (str): Path to audio file
        filename (str): Base filename (without extension)
        output_dir (str): Directory to save features
        csv_path (str): Path to tracking CSV

    Returns:
        bool: True if processing succeeded, False otherwise
    """
    print(f"\nProcessing: {filename}")

    try:
        # Check if already processed
        if is_file_processed(filename, csv_path):
            print("  ⊙ Already processed - skipping")
            return True

        # Preprocess audio
        waveform, sample_rate = preprocess_audio(audio_path)

        # Extract features
        features = extract_whisper_features(waveform, sample_rate)

        # Save features
        features_path = save_features(features, filename, output_dir)

        # Record in CSV
        add_processed_file(filename, features_path, csv_path)

        print(f"  ✓ Successfully processed: {features.shape}")
        return True

    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def process_all_files(
    mp3_dir: str,
    output_dir: str,
    csv_path: str
) -> dict:
    """
    Process all MP3 files in a directory.

    Args:
        mp3_dir (str): Directory containing MP3 files
        output_dir (str): Directory to save features
        csv_path (str): Path to tracking CSV

    Returns:
        dict: Processing statistics
    """
    # Find all MP3 files
    mp3_files = glob(os.path.join(mp3_dir, '*.mp3'))

    if not mp3_files:
        print(f"No MP3 files found in: {mp3_dir}")
        return {'total': 0, 'processed': 0, 'skipped': 0, 'errors': 0}

    print(f"\n{'='*70}")
    print(f"Found {len(mp3_files)} MP3 files")
    print(f"{'='*70}")

    stats = {
        'total': len(mp3_files),
        'processed': 0,
        'skipped': 0,
        'errors': 0
    }

    for audio_path in mp3_files:
        # Extract filename without extension
        filename = os.path.splitext(os.path.basename(audio_path))[0]

        # Check if already processed
        if is_file_processed(filename, csv_path):
            stats['skipped'] += 1
            continue

        # Process file
        success = process_single_file(audio_path, filename, output_dir, csv_path)

        if success:
            stats['processed'] += 1
        else:
            stats['errors'] += 1

    return stats


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_mp3_files(folder_path: str) -> list:
    """
    Get list of MP3 files in a folder.

    Args:
        folder_path (str): Path to folder

    Returns:
        list: List of MP3 file paths
    """
    return glob(os.path.join(folder_path, '*.mp3'))


def create_dataframe_from_mp3_files(mp3_folder: str) -> pd.DataFrame:
    """
    Create DataFrame with MP3 file information.

    Args:
        mp3_folder (str): Path to folder containing MP3 files

    Returns:
        pd.DataFrame: DataFrame with 'Filepath' and 'Filename' columns
    """
    mp3_files = get_mp3_files(mp3_folder)
    filenames = [os.path.splitext(os.path.basename(f))[0] for f in mp3_files]

    return pd.DataFrame({
        'Filepath': mp3_files,
        'Filename': filenames
    })


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""
    print("\n" + "="*70)
    print("WHISPER FEATURE EXTRACTION PIPELINE")
    print("="*70)

    print("\nConfiguration:")
    print(f"  Model: {WHISPER_MODEL_NAME}")
    print(f"  MP3 directory: {MP3_FILES_DIR}")
    print(f"  Output directory: {WHISPER_FEATURES_DIR}")
    print(f"  Tracking CSV: {PROCESSED_FILES_CSV}")
    print(f"  Target sample rate: {TARGET_SAMPLE_RATE} Hz")

    # Verify input directory exists
    if not os.path.exists(MP3_FILES_DIR):
        print(f"\n✗ Error: MP3 directory not found: {MP3_FILES_DIR}")
        return

    # Process all files
    stats = process_all_files(MP3_FILES_DIR, WHISPER_FEATURES_DIR, PROCESSED_FILES_CSV)

    # Print summary
    print("\n" + "="*70)
    print("PROCESSING COMPLETE")
    print("="*70)
    print(f"Total files: {stats['total']}")
    print(f"Newly processed: {stats['processed']}")
    print(f"Already processed (skipped): {stats['skipped']}")
    print(f"Errors: {stats['errors']}")
    print(f"Features saved in: {WHISPER_FEATURES_DIR}")
    print(f"Tracking CSV: {PROCESSED_FILES_CSV}")
    print("="*70)


if __name__ == '__main__':
    main()
