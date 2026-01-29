#!/usr/bin/env python3
"""
OpenSmile Feature Extractor

Extracts acoustic features using OpenSMILE (eGeMAPSv02 feature set).
"""

import json
import opensmile
from src.audio_utils import convert_to_wav, preprocess_audio


class OpenSmileExtractor:
    """OpenSmile feature extraction class."""
    
    def __init__(self):
        """Initialize OpenSmile extractor."""
        print("Initializing OpenSmile...")
        self.smile = opensmile.Smile(
            feature_set=opensmile.FeatureSet.eGeMAPSv02,
            feature_level=opensmile.FeatureLevel.Functionals,
        )
        print(f"  ✓ OpenSmile initialized with {self.smile.feature_names.shape[0]} features")
    
    def extract_features(self, audio_path: str) -> dict:
        """
        Extract OpenSmile features from audio file.
        
        Args:
            audio_path (str): Path to audio file
            
        Returns:
            dict: Dictionary of feature values
        """
        # Convert to WAV if necessary
        audio_path = convert_to_wav(audio_path)
        
        # Preprocess audio
        waveform, sample_rate = preprocess_audio(audio_path)
        
        # Extract features
        features_df = self.smile.process_file(audio_path)
        
        # Convert to dictionary
        features = features_df.iloc[0].to_dict()
        
        return features
    
    def features_to_json(self, features: dict) -> str:
        """
        Convert features dictionary to JSON string.
        
        Args:
            features (dict): Features dictionary
            
        Returns:
            str: JSON string
        """
        return json.dumps(features)
    
    def process_file(self, audio_path: str, filename: str) -> dict:
        """
        Process a single audio file and extract OpenSmile features.
        
        Args:
            audio_path (str): Path to audio file
            filename (str): Base filename
            
        Returns:
            dict: Extracted features
        """
        print(f"  Processing with OpenSmile: {filename}")
        
        # Extract features
        features = self.extract_features(audio_path)
        
        print(f"    ✓ Extracted {len(features)} features")
        
        return features
