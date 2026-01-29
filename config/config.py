#!/usr/bin/env python3
"""
Configuration Management for Audio Analysis Pipeline

This module handles all configuration settings including:
- Database connections
- File paths
- Model parameters
- Processing options
"""

import os
from pathlib import Path
from typing import Dict
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Config:
    """Central configuration class for the audio analysis pipeline."""
    
    # ========================================================================
    # Database Configuration
    # ========================================================================
    DB_HOST = os.getenv('DB_HOST', 'localhost')
    DB_DATABASE = os.getenv('DB_DATABASE', 'audio_analysis')
    DB_USERNAME = os.getenv('DB_USERNAME', 'root')
    DB_PASSWORD = os.getenv('DB_PASSWORD', '')
    
    # ========================================================================
    # Directory Paths
    # ========================================================================
    DATA_DIR = Path(os.getenv('DATA_DIR', '.'))
    AUDIO_ROOT = Path(os.getenv('AUDIO_ROOT', '/sandata/karthik9/nwa'))
    WHISPER_FEATURES_DIR = Path(os.getenv('WHISPER_FEATURES_DIR', '/sandata/karthik9/whisper_new'))
    TRILL_FEATURES_DIR = Path(os.getenv('TRILL_FEATURES_DIR', '/sandata/karthik9/mstd_embeddings'))
    
    # ========================================================================
    # Input Files
    # ========================================================================
    ID_MAPPING_FILE = DATA_DIR / os.getenv('ID_MAPPING_FILE', 'Klaatch Sanitized TextCEL 2021 to 2023 (1).xlsx')
    DEMOGRAPHICS_FILE = DATA_DIR / os.getenv('DEMOGRAPHICS_FILE', 'Demographics_Klaatch Sanitized TextCEL 2021 to 2023.xlsx')
    TRANSCRIPTS_FILE = DATA_DIR / os.getenv('TRANSCRIPTS_FILE', 'Klaatch_transcripts.csv')
    NEW_TRANSCRIPTS_FILE = DATA_DIR / os.getenv('NEW_TRANSCRIPTS_FILE', 'new_transcripts.csv')
    
    # ========================================================================
    # Output Files
    # ========================================================================
    PROCESSED_DATA_FILE = DATA_DIR / os.getenv('PROCESSED_DATA_FILE', 'processed_klaatch_data.csv')
    
    # ========================================================================
    # Database Tables
    # ========================================================================
    DB_TABLES = {
        'main_data': 'merged_data',
        'whisper_mean': 'feat$whisper_mean_n$merged_data$message_id',
        'whisper_median': 'feat$whisper_median_n$merged_data$message_id',
        'opensmile': 'feat$opensmile_n$merged_data$message_id',
        'librosa': 'feat$librosa_n$merged_data$message_id',
        'trill': 'feat$trill_mstd$merged_data$message_id',
        'audio_processing': 'new_audio_processing_status'
    }
    
    # ========================================================================
    # Audio Processing Parameters
    # ========================================================================
    TARGET_SAMPLE_RATE = int(os.getenv('TARGET_SAMPLE_RATE', '16000'))
    WHISPER_MODEL = os.getenv('WHISPER_MODEL', 'openai/whisper-base')
    
    # ========================================================================
    # Text Preprocessing Patterns
    # ========================================================================
    TEXT_REMOVAL_PATTERNS = [
        'lisa\w*', 'lisa',
        'speaker1', 'speaker2',
        'strongly agree', 'strongly disagree',
        'strongly_agree', 'strongly_disagree',
        'agree\w*', 'disagree\w*',
        'neutral', 'Speaker', 'Speaker\w*',
        'nn\w*', 'nnspeaker\w*',
        "don't\w*", 'not\w*', 'strongly\w*', 
        'definitely\w*', 'neither\w*',
        '!', '\$', ',', '\.', '\.\.', 
        '\.\s*mm hmm', '\.\.',
    ]
    
    TIMESTAMP_PATTERN = r'\[\d{2}:\d{2}:\d{2}\]'
    
    @classmethod
    def get_db_config(cls) -> Dict[str, str]:
        """
        Get database configuration as a dictionary.
        
        Returns:
            Dict[str, str]: Database connection parameters
        """
        return {
            'host': cls.DB_HOST,
            'database': cls.DB_DATABASE,
            'user': cls.DB_USERNAME,
            'password': cls.DB_PASSWORD
        }
    
    @classmethod
    def validate_paths(cls) -> bool:
        """
        Validate that required directories exist.
        
        Returns:
            bool: True if all paths are valid
        """
        required_paths = [
            cls.DATA_DIR,
        ]
        
        for path in required_paths:
            if not path.exists():
                print(f"Warning: Path does not exist: {path}")
                return False
        
        return True
    
    @classmethod
    def create_output_dirs(cls):
        """Create output directories if they don't exist."""
        cls.WHISPER_FEATURES_DIR.mkdir(parents=True, exist_ok=True)
        print(f"✓ Ensured output directory exists: {cls.WHISPER_FEATURES_DIR}")
