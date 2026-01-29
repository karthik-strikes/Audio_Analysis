#!/usr/bin/env python3
"""
Whisper Feature Extractor

Extracts audio embeddings using OpenAI's Whisper model.
"""

import os
import numpy as np
import torch
from transformers import WhisperProcessor, WhisperModel
from config import Config
from src.audio_utils import preprocess_audio


class WhisperExtractor:
    """Whisper feature extraction class."""
    
    def __init__(self, model_name: str = None):
        """
        Initialize Whisper extractor.
        
        Args:
            model_name (str): Whisper model name (default: from Config)
        """
        self.model_name = model_name or Config.WHISPER_MODEL
        self.processor = None
        self.model = None
        self.features_dir = Config.WHISPER_FEATURES_DIR
        
        # Ensure features directory exists
        self.features_dir.mkdir(parents=True, exist_ok=True)
    
    def load_model(self):
        """Load Whisper processor and model."""
        print(f"Loading Whisper model: {self.model_name}...")
        self.processor = WhisperProcessor.from_pretrained(self.model_name)
        self.model = WhisperModel.from_pretrained(self.model_name)
        self.model.eval()
        print("  ✓ Whisper model loaded")
    
    def is_processed(self, filename: str) -> bool:
        """
        Check if features exist for a given file.
        
        Args:
            filename (str): Base filename without extension
            
        Returns:
            bool: True if features exist
        """
        feature_file = self.features_dir / f"{filename}_whisper.npy"
        return feature_file.exists()
    
    def extract_features(self, waveform: torch.Tensor, sample_rate: int) -> np.ndarray:
        """
        Extract Whisper features from audio waveform.
        
        Args:
            waveform (torch.Tensor): Audio waveform
            sample_rate (int): Sample rate
            
        Returns:
            np.ndarray: Extracted features of shape (sequence_length, hidden_size)
        """
        if self.processor is None or self.model is None:
            self.load_model()
        
        # Process audio and prepare input tensors
        input_features = self.processor(
            waveform.squeeze().numpy(), 
            sampling_rate=sample_rate, 
            return_tensors="pt"
        ).input_features

        # Extract features
        with torch.no_grad():
            encoder_outputs = self.model.get_encoder()(input_features)
            features = encoder_outputs.last_hidden_state

        # Convert to NumPy array
        features_np = features.squeeze().numpy()
        
        return features_np
    
    def save_features(self, features: np.ndarray, filename: str) -> str:
        """
        Save extracted features to file.
        
        Args:
            features (np.ndarray): Extracted features
            filename (str): Base filename
            
        Returns:
            str: Path to saved features file
        """
        base_name = filename.rsplit('.mp3', 1)[0]
        features_file_path = self.features_dir / f"{base_name}_whisper.npy"
        
        np.save(features_file_path, features)
        
        return str(features_file_path)
    
    def process_file(self, audio_path: str, filename: str) -> str:
        """
        Process a single audio file and extract Whisper features.
        
        Args:
            audio_path (str): Path to audio file
            filename (str): Base filename
            
        Returns:
            str: Path to saved features file
        """
        print(f"  Processing with Whisper: {filename}")
        
        # Check if already processed
        if self.is_processed(filename):
            print(f"    ⊙ Already processed")
            return str(self.features_dir / f"{filename}_whisper.npy")
        
        # Preprocess audio
        waveform, sample_rate = preprocess_audio(audio_path)
        
        # Extract features
        features = self.extract_features(waveform, sample_rate)
        print(f"    Feature shape: {features.shape}")
        
        # Save features
        features_path = self.save_features(features, filename)
        print(f"    ✓ Saved to {features_path}")
        
        return features_path
    
    def compute_statistics(self, features_path: str) -> tuple:
        """
        Compute mean and median statistics for features.
        
        Args:
            features_path (str): Path to features file
            
        Returns:
            tuple: (mean_features, median_features)
        """
        features = np.load(features_path)
        
        if features.ndim == 1:
            return features, features
        
        mean = np.mean(features, axis=0)
        median = np.median(features, axis=0)
        
        return mean, median
