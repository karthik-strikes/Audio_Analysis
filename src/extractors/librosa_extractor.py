#!/usr/bin/env python3
"""
Librosa Feature Extractor

Extracts audio features using Librosa including:
- MFCCs (Mel-frequency cepstral coefficients)
- Chroma features
- Spectral contrast
- Tonnetz features
"""

import librosa
from src.audio_utils import convert_features_to_float


class LibrosaExtractor:
    """Librosa feature extraction class."""
    
    def __init__(self):
        """Initialize Librosa extractor."""
        print("Librosa extractor initialized")
    
    def extract_features(self, audio_path: str) -> dict:
        """
        Extract Librosa features from audio file.
        
        Extracts:
        - 13 MFCCs (Mel-frequency cepstral coefficients)
        - 12 Chroma features (pitch class profiles)
        - 7 Spectral contrast bands
        - 6 Tonnetz features (tonal centroid features)
        
        Args:
            audio_path (str): Path to audio file
            
        Returns:
            dict: Dictionary with 38 feature values
        """
        # Load audio
        y, sr = librosa.load(audio_path, sr=None)
        
        # Extract features
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr)
        
        # Aggregate features by taking mean across time frames
        mfccs_mean = mfccs.mean(axis=1)
        chroma_mean = chroma.mean(axis=1)
        spectral_contrast_mean = spectral_contrast.mean(axis=1)
        tonnetz_mean = tonnetz.mean(axis=1)
        
        # Create features dictionary
        features = {
            "mfcc_1": mfccs_mean[0], "mfcc_2": mfccs_mean[1], "mfcc_3": mfccs_mean[2],
            "mfcc_4": mfccs_mean[3], "mfcc_5": mfccs_mean[4], "mfcc_6": mfccs_mean[5],
            "mfcc_7": mfccs_mean[6], "mfcc_8": mfccs_mean[7], "mfcc_9": mfccs_mean[8],
            "mfcc_10": mfccs_mean[9], "mfcc_11": mfccs_mean[10], "mfcc_12": mfccs_mean[11],
            "mfcc_13": mfccs_mean[12],
            "chroma_1": chroma_mean[0], "chroma_2": chroma_mean[1], "chroma_3": chroma_mean[2],
            "chroma_4": chroma_mean[3], "chroma_5": chroma_mean[4], "chroma_6": chroma_mean[5],
            "chroma_7": chroma_mean[6], "chroma_8": chroma_mean[7], "chroma_9": chroma_mean[8],
            "chroma_10": chroma_mean[9], "chroma_11": chroma_mean[10], "chroma_12": chroma_mean[11],
            "spectral_contrast_1": spectral_contrast_mean[0],
            "spectral_contrast_2": spectral_contrast_mean[1],
            "spectral_contrast_3": spectral_contrast_mean[2],
            "spectral_contrast_4": spectral_contrast_mean[3],
            "spectral_contrast_5": spectral_contrast_mean[4],
            "spectral_contrast_6": spectral_contrast_mean[5],
            "spectral_contrast_7": spectral_contrast_mean[6],
            "tonnetz_1": tonnetz_mean[0], "tonnetz_2": tonnetz_mean[1],
            "tonnetz_3": tonnetz_mean[2], "tonnetz_4": tonnetz_mean[3],
            "tonnetz_5": tonnetz_mean[4], "tonnetz_6": tonnetz_mean[5]
        }
        
        # Convert to Python floats
        features = convert_features_to_float(features)
        
        return features
    
    def process_file(self, audio_path: str, filename: str) -> dict:
        """
        Process a single audio file and extract Librosa features.
        
        Args:
            audio_path (str): Path to audio file
            filename (str): Base filename
            
        Returns:
            dict: Extracted features (38 features)
        """
        print(f"  Processing with Librosa: {filename}")
        
        # Extract features
        features = self.extract_features(audio_path)
        
        print(f"    ✓ Extracted {len(features)} features")
        
        return features
