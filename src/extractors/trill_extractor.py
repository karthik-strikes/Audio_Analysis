#!/usr/bin/env python3
"""
Trill Feature Extractor

Handles loading and processing of pre-computed Trill embeddings.
"""

import os
import numpy as np
from pathlib import Path
from config import Config


class TrillExtractor:
    """Trill feature extraction class."""
    
    def __init__(self):
        """Initialize Trill extractor."""
        self.features_dir = Config.TRILL_FEATURES_DIR
        print(f"Trill extractor initialized")
        print(f"  Features directory: {self.features_dir}")
    
    def find_trill_file(self, filename: str) -> str:
        """
        Find Trill embedding file for a given filename.
        
        Args:
            filename (str): Base filename
            
        Returns:
            str: Path to Trill file or None
        """
        # Try different naming patterns
        patterns = [
            f"{filename}_wb_mean_trill_embedding.npy",
            f"{filename}_std_trill_embedding.npy",
            f"{filename}_trill_embedding.npy"
        ]
        
        for pattern in patterns:
            trill_path = self.features_dir / pattern
            if trill_path.exists():
                return str(trill_path)
        
        return None
    
    def load_embedding(self, trill_path: str) -> np.ndarray:
        """
        Load Trill embedding from file.
        
        Args:
            trill_path (str): Path to Trill embedding file
            
        Returns:
            np.ndarray: Trill embedding array
        """
        embedding = np.load(trill_path)
        
        # If multi-dimensional, take mean
        if embedding.ndim > 1:
            embedding = np.mean(embedding, axis=0)
        
        return embedding
    
    def process_file(self, filename: str) -> np.ndarray:
        """
        Load Trill embedding for a file.
        
        Args:
            filename (str): Base filename
            
        Returns:
            np.ndarray: Trill embedding or None if not found
        """
        trill_path = self.find_trill_file(filename)
        
        if trill_path is None:
            print(f"  ⊗ No Trill embedding found for: {filename}")
            return None
        
        print(f"  Loading Trill: {filename}")
        embedding = self.load_embedding(trill_path)
        print(f"    ✓ Loaded embedding: {embedding.shape}")
        
        return embedding
    
    def embedding_to_dict(self, embedding: np.ndarray) -> dict:
        """
        Convert embedding array to dictionary with column names.
        
        Args:
            embedding (np.ndarray): Embedding array
            
        Returns:
            dict: Dictionary with keys 'trill_0', 'trill_1', etc.
        """
        return {f"trill_{i}": float(embedding[i]) for i in range(len(embedding))}
