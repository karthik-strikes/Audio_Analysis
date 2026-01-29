#!/usr/bin/env python3
"""
Data Loading Module

Handles loading and merging data from multiple sources:
- ID mappings
- Demographics
- Transcripts
- Audio files
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from config import Config


def load_id_mapping() -> pd.DataFrame:
    """
    Load ID mapping from Excel file.
    
    Returns:
        pd.DataFrame: ID mapping dataframe
    """
    print("Loading ID mapping...")
    df = pd.read_excel(Config.ID_MAPPING_FILE, sheet_name='Klaatch Sanitized')
    print(f"  ✓ Loaded {len(df)} ID mappings")
    return df


def load_demographics() -> pd.DataFrame:
    """
    Load demographics data from Excel file.
    
    Returns:
        pd.DataFrame: Demographics dataframe
    """
    print("Loading demographics...")
    df = pd.read_excel(Config.DEMOGRAPHICS_FILE)
    print(f"  ✓ Loaded demographics for {len(df)} participants")
    return df


def load_transcripts() -> pd.DataFrame:
    """
    Load transcripts from CSV file.
    
    Returns:
        pd.DataFrame: Transcripts dataframe
    """
    print("Loading transcripts...")
    df = pd.read_csv(Config.TRANSCRIPTS_FILE)
    
    # Clean and standardize IDs
    df['New ID'] = pd.to_numeric(df['New ID'], errors='coerce').replace(0, np.nan)
    df['New ID'] = df['New ID'].astype(str)
    df['Old ID'] = df['Old ID'].astype(int).astype(str)
    
    print(f"  ✓ Loaded {len(df)} transcripts")
    print(f"  Transcripts with New ID: {df['New ID'].notna().sum()}")
    
    return df


def load_new_transcripts() -> pd.DataFrame:
    """
    Load additional transcripts if available.
    
    Returns:
        pd.DataFrame: New transcripts dataframe or empty dataframe
    """
    if Config.NEW_TRANSCRIPTS_FILE.exists():
        print("Loading additional transcripts...")
        df = pd.read_csv(Config.NEW_TRANSCRIPTS_FILE)
        print(f"  ✓ Loaded {len(df)} additional transcripts")
        return df
    else:
        print(f"  No additional transcripts file found")
        return pd.DataFrame()


def merge_transcripts_with_demographics(transcripts_df: pd.DataFrame, 
                                       demographics_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge transcripts with demographics data.
    
    Args:
        transcripts_df (pd.DataFrame): Transcripts dataframe
        demographics_df (pd.DataFrame): Demographics dataframe
        
    Returns:
        pd.DataFrame: Merged dataframe
    """
    print("Merging transcripts with demographics...")
    
    merged_df = pd.merge(
        transcripts_df, 
        demographics_df, 
        left_on='New ID', 
        right_on='New ID', 
        how='left'
    )
    
    print(f"  ✓ Merged dataset: {len(merged_df)} records")
    print(f"  Missing demographics: {merged_df['Age'].isna().sum()} records")
    
    return merged_df


def discover_audio_files() -> pd.DataFrame:
    """
    Scan audio root directory and create dataframe of audio files.
    
    Returns:
        pd.DataFrame: Dataframe with audio file information
    """
    print(f"Scanning audio files in {Config.AUDIO_ROOT}...")
    
    file_data = []
    
    for root, dirs, files in os.walk(Config.AUDIO_ROOT):
        for file in files:
            if file.endswith('.mp3'):
                file_path = os.path.join(root, file)
                
                # Extract metadata from filename
                # Expected format: {ID}_{Date}_{Time}.mp3
                parts = file.replace('.mp3', '').split('_')
                
                if len(parts) >= 2:
                    klaatch_id = parts[0]
                    date = parts[1] if len(parts) > 1 else 'unknown'
                    
                    file_data.append({
                        'Filename': file.replace('.mp3', ''),
                        'Filepath': file_path,
                        'Original_Filename': file,
                        'KlaatchID': klaatch_id,
                        'Date': date
                    })
    
    df_audio = pd.DataFrame(file_data)
    df_audio = df_audio.drop_duplicates(subset='Filename', keep='first')
    
    print(f"  ✓ Found {len(df_audio)} unique audio files")
    print(f"  Unique participants: {df_audio['KlaatchID'].nunique()}")
    
    return df_audio


def merge_metadata_with_audio(metadata_df: pd.DataFrame, 
                              audio_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge metadata with audio file paths.
    
    Args:
        metadata_df (pd.DataFrame): Metadata dataframe
        audio_df (pd.DataFrame): Audio files dataframe
        
    Returns:
        pd.DataFrame: Merged dataframe
    """
    print("Merging metadata with audio file paths...")
    
    # Extract date from metadata filenames for consistency
    metadata_df['Date'] = metadata_df['Filename'].apply(
        lambda x: x.split('_')[1] if '_' in str(x) else 'unknown'
    )
    
    merged_df = pd.merge(metadata_df, audio_df, on='Filename', how='left')
    
    print(f"  ✓ Merged dataset: {len(merged_df)} records")
    print(f"  Records with audio files: {merged_df['Filepath'].notna().sum()}")
    print(f"  Records missing audio: {merged_df['Filepath'].isna().sum()}")
    
    return merged_df


def load_and_merge_all_data() -> pd.DataFrame:
    """
    Load and merge all data sources into a single dataframe.
    
    Returns:
        pd.DataFrame: Complete merged dataset
    """
    print("="*70)
    print("LOADING AND MERGING DATA")
    print("="*70)
    
    # Load all data sources
    id_mapping = load_id_mapping()
    demographics = load_demographics()
    transcripts = load_transcripts()
    
    # Merge transcripts with demographics
    metadata_df = merge_transcripts_with_demographics(transcripts, demographics)
    
    # Discover audio files
    audio_df = discover_audio_files()
    
    # Merge everything together
    merged_df = merge_metadata_with_audio(metadata_df, audio_df)
    
    # Load additional transcripts if available
    new_transcripts = load_new_transcripts()
    if len(new_transcripts) > 0:
        merged_df = merged_df.merge(
            new_transcripts,
            on='Filename',
            how='left',
            suffixes=('', '_new')
        )
    
    print("\n" + "="*70)
    print(f"FINAL DATASET: {len(merged_df)} records")
    print("="*70)
    
    return merged_df
