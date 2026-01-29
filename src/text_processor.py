#!/usr/bin/env python3
"""
Text Preprocessing Module

Handles text cleaning and preprocessing operations including:
- Pattern removal
- Timestamp cleaning
- Whitespace normalization
"""

import re
import pandas as pd
from config import Config


def clean_text(text: str, patterns: list = None, timestamp_pattern: str = None) -> str:
    """
    Clean text by removing specified patterns and timestamps.
    
    Args:
        text (str): Text to clean
        patterns (list): List of regex patterns to remove (default: from Config)
        timestamp_pattern (str): Timestamp regex pattern (default: from Config)
        
    Returns:
        str: Cleaned text
    """
    if text is None or pd.isna(text):
        return ""
    
    if patterns is None:
        patterns = Config.TEXT_REMOVAL_PATTERNS
    
    if timestamp_pattern is None:
        timestamp_pattern = Config.TIMESTAMP_PATTERN
    
    # Escape patterns that need it
    escaped_patterns = []
    for pattern in patterns:
        if '\\w' in pattern:
            escaped_patterns.append(pattern)  # Keep regex operators
        else:
            escaped_patterns.append(re.escape(pattern))  # Escape others
    
    # Build combined pattern
    words_regex = r'\b(?:' + '|'.join(escaped_patterns) + r')\b'
    full_regex_pattern = f'(?:{words_regex})|(?:{timestamp_pattern})'
    
    # Apply cleaning
    cleaned = re.sub(full_regex_pattern, '', str(text), flags=re.IGNORECASE)
    
    # Clean up extra whitespace
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    
    return cleaned


def preprocess_text_column(df: pd.DataFrame, text_column: str = 'Text') -> pd.DataFrame:
    """
    Preprocess text column in a DataFrame with comprehensive pattern removal.
    
    Args:
        df (pd.DataFrame): DataFrame containing text data
        text_column (str): Name of text column to process
        
    Returns:
        pd.DataFrame: DataFrame with cleaned text and text length column
    """
    print(f"Preprocessing text in column '{text_column}'...")
    
    if text_column not in df.columns:
        print(f"Warning: '{text_column}' column not found")
        return df
    
    # Create a copy
    result_df = df.copy()
    
    # Apply text cleaning
    result_df[text_column] = result_df[text_column].apply(clean_text)
    
    # Calculate text length
    result_df['Text_Length'] = result_df[text_column].str.len()
    
    # Statistics
    original_count = df[text_column].notna().sum()
    cleaned_count = (result_df['Text_Length'] > 0).sum()
    empty_count = (result_df['Text_Length'] == 0).sum()
    
    print(f"  Original text samples: {original_count}")
    print(f"  Cleaned text samples: {cleaned_count}")
    print(f"  Empty after cleaning: {empty_count}")
    
    return result_df


def extract_metadata_from_filename(filename: str) -> dict:
    """
    Extract metadata from filename.
    Expected format: {ID}_{Date}_{Time}.mp3
    
    Args:
        filename (str): Filename to parse
        
    Returns:
        dict: Dictionary with 'klaatch_id' and 'date' keys
    """
    parts = filename.replace('.mp3', '').split('_')
    
    metadata = {
        'klaatch_id': parts[0] if len(parts) > 0 else 'unknown',
        'date': parts[1] if len(parts) > 1 else 'unknown'
    }
    
    return metadata
