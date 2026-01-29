#!/usr/bin/env python3
"""
Audio Utilities Module

Common audio processing functions including:
- Audio format conversion
- Audio preprocessing
- Feature computation
"""

import os
import numpy as np
import torch
import torchaudio
from pydub import AudioSegment
from config import Config


def convert_to_wav(audio_file_path: str) -> str:
    """
    Convert audio file to WAV format if not already in WAV.
    
    Args:
        audio_file_path (str): Path to audio file
        
    Returns:
        str: Path to WAV file
    """
    if not audio_file_path.endswith('.wav'):
        print(f"  Converting {os.path.basename(audio_file_path)} to WAV...")
        audio = AudioSegment.from_file(audio_file_path)
        wav_path = audio_file_path.rsplit('.', 1)[0] + '.wav'
        audio.export(wav_path, format='wav')
        return wav_path
    return audio_file_path


def preprocess_audio(audio_file_path: str, target_sample_rate: int = None):
    """
    Preprocess audio file: convert to WAV and resample to target sample rate.
    
    Args:
        audio_file_path (str): Path to audio file
        target_sample_rate (int): Target sample rate (default: from Config)
        
    Returns:
        tuple: (waveform, sample_rate)
    """
    if target_sample_rate is None:
        target_sample_rate = Config.TARGET_SAMPLE_RATE
    
    # Convert to WAV if necessary
    if not audio_file_path.endswith('.wav'):
        print(f"  Converting to WAV...")
        audio = AudioSegment.from_file(audio_file_path)
        new_file_path = os.path.splitext(audio_file_path)[0] + ".wav"
        audio.export(new_file_path, format="wav")
        audio_file_path = new_file_path
    
    # Load audio with torchaudio
    waveform, sampling_rate = torchaudio.load(audio_file_path)
    
    # Resample if necessary
    if sampling_rate != target_sample_rate:
        print(f"  Resampling from {sampling_rate} Hz to {target_sample_rate} Hz...")
        resample_transform = torchaudio.transforms.Resample(
            orig_freq=sampling_rate, 
            new_freq=target_sample_rate
        )
        waveform = resample_transform(waveform)
    
    return waveform, target_sample_rate


def compute_mean_median(file_path: str):
    """
    Compute per-dimension mean and median for a NumPy array.
    
    Args:
        file_path (str): Path to .npy file
        
    Returns:
        tuple: (mean_array, median_array)
    """
    data = np.load(file_path)
    
    if data.ndim == 1:
        return data, data
    
    mean = np.mean(data, axis=0)
    median = np.median(data, axis=0)
    
    return mean, median


def load_numpy_to_columns(npy_path: str) -> dict:
    """
    Load 1024-dimensional Trill embedding and expand into dictionary of columns.
    
    Args:
        npy_path (str): Path to .npy file
        
    Returns:
        dict: Dictionary with keys 'dim0' through 'dim1023'
    """
    array = np.load(npy_path)
    return {f"dim{i}": array[i] for i in range(len(array))}


def convert_features_to_float(features: dict) -> dict:
    """
    Convert all feature values to Python float.
    
    Args:
        features (dict): Dictionary with feature values
        
    Returns:
        dict: Dictionary with float values
    """
    return {key: float(value) for key, value in features.items()}
