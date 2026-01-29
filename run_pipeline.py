#!/usr/bin/env python3
"""
Audio Feature Extraction Pipeline - Main Orchestrator

This script orchestrates the complete audio feature extraction pipeline:
1. Data loading and merging
2. Text preprocessing
3. Feature extraction (Whisper, OpenSmile, Librosa, Trill)
4. Database storage

Usage:
    python run_pipeline.py [options]

Options:
    --load-only         Only load and merge data (skip feature extraction)
    --extract-whisper   Extract Whisper features
    --extract-opensmile Extract OpenSmile features
    --extract-librosa   Extract Librosa features
    --load-trill        Load existing Trill features
    --insert-db         Insert features into database
    --all               Run all steps
"""

import argparse
import sys
import pandas as pd
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from config import Config
from src.data_loader import load_and_merge_all_data
from src.text_processor import preprocess_text_column
from src.database import (
    test_connection, insert_data, insert_feature_table_into_db,
    connect_to_db, is_file_processed_opensmile, is_file_processed_librosa
)
from src.extractors import (
    WhisperExtractor, OpenSmileExtractor, 
    LibrosaExtractor, TrillExtractor
)


def setup_pipeline():
    """Initialize pipeline and validate configuration."""
    print("\n" + "="*70)
    print("AUDIO FEATURE EXTRACTION PIPELINE")
    print("="*70)
    
    print("\nConfiguration:")
    print(f"  Data directory: {Config.DATA_DIR}")
    print(f"  Audio root: {Config.AUDIO_ROOT}")
    print(f"  Whisper features: {Config.WHISPER_FEATURES_DIR}")
    print(f"  Database: {Config.DB_HOST}/{Config.DB_DATABASE}")
    
    # Test database connection
    print("\nTesting database connection...")
    if not test_connection():
        print("Warning: Database connection failed. Database operations will not work.")
    
    # Create output directories
    Config.create_output_dirs()
    
    return True


def load_and_preprocess_data():
    """Load all data and preprocess text."""
    print("\n" + "="*70)
    print("STEP 1: DATA LOADING AND PREPROCESSING")
    print("="*70)
    
    # Load and merge all data
    merged_df = load_and_merge_all_data()
    
    # Preprocess text
    print("\nPreprocessing text...")
    merged_df = preprocess_text_column(merged_df, 'Text')
    
    # Save processed data
    print(f"\nSaving processed data to {Config.PROCESSED_DATA_FILE}...")
    merged_df.to_csv(Config.PROCESSED_DATA_FILE, index=False)
    print(f"  ✓ Saved {len(merged_df)} records")
    
    return merged_df


def extract_whisper_features(df: pd.DataFrame):
    """Extract Whisper features from audio files."""
    print("\n" + "="*70)
    print("STEP 2: WHISPER FEATURE EXTRACTION")
    print("="*70)
    
    extractor = WhisperExtractor()
    
    processed_count = 0
    skipped_count = 0
    error_count = 0
    
    whisper_data = []
    
    for idx, row in df.iterrows():
        if pd.isna(row.get('Filepath')):
            continue
        
        filename = row['Filename']
        filepath = row['Filepath']
        
        try:
            features_path = extractor.process_file(filepath, filename)
            
            # Compute statistics
            mean_feat, median_feat = extractor.compute_statistics(features_path)
            
            whisper_data.append({
                'group_id': filename,
                'mean_features': mean_feat,
                'median_features': median_feat
            })
            
            processed_count += 1
            
        except Exception as e:
            print(f"  ✗ Error processing {filename}: {e}")
            error_count += 1
    
    print(f"\nWhisper extraction complete:")
    print(f"  Processed: {processed_count}")
    print(f"  Errors: {error_count}")
    
    return whisper_data


def extract_opensmile_features(df: pd.DataFrame):
    """Extract OpenSmile features from audio files."""
    print("\n" + "="*70)
    print("STEP 3: OPENSMILE FEATURE EXTRACTION")
    print("="*70)
    
    extractor = OpenSmileExtractor()
    connection = connect_to_db()
    
    processed_count = 0
    skipped_count = 0
    error_count = 0
    
    for idx, row in df.iterrows():
        if pd.isna(row.get('Filepath')):
            continue
        
        filename = row['Filename']
        filepath = row['Filepath']
        
        # Check if already processed
        if is_file_processed_opensmile(filename, connection):
            print(f"  ⊙ Skipping {filename} (already processed)")
            skipped_count += 1
            continue
        
        try:
            features = extractor.process_file(filepath, filename)
            
            # Store in database (implement as needed)
            # insert_opensmile_features(connection, filename, features)
            
            processed_count += 1
            
        except Exception as e:
            print(f"  ✗ Error processing {filename}: {e}")
            error_count += 1
    
    connection.close()
    
    print(f"\nOpenSmile extraction complete:")
    print(f"  Processed: {processed_count}")
    print(f"  Skipped: {skipped_count}")
    print(f"  Errors: {error_count}")


def extract_librosa_features(df: pd.DataFrame):
    """Extract Librosa features from audio files."""
    print("\n" + "="*70)
    print("STEP 4: LIBROSA FEATURE EXTRACTION")
    print("="*70)
    
    extractor = LibrosaExtractor()
    connection = connect_to_db()
    
    processed_count = 0
    skipped_count = 0
    error_count = 0
    
    for idx, row in df.iterrows():
        if pd.isna(row.get('Filepath')):
            continue
        
        filename = row['Filename']
        filepath = row['Filepath']
        
        # Check if already processed
        if is_file_processed_librosa(filename, connection):
            print(f"  ⊙ Skipping {filename} (already processed)")
            skipped_count += 1
            continue
        
        try:
            features = extractor.process_file(filepath, filename)
            
            # Store in database (implement as needed)
            # insert_librosa_features(connection, filename, features)
            
            processed_count += 1
            
        except Exception as e:
            print(f"  ✗ Error processing {filename}: {e}")
            error_count += 1
    
    connection.close()
    
    print(f"\nLibrosa extraction complete:")
    print(f"  Processed: {processed_count}")
    print(f"  Skipped: {skipped_count}")
    print(f"  Errors: {error_count}")


def load_trill_features(df: pd.DataFrame):
    """Load pre-computed Trill features."""
    print("\n" + "="*70)
    print("STEP 5: LOADING TRILL FEATURES")
    print("="*70)
    
    extractor = TrillExtractor()
    
    loaded_count = 0
    missing_count = 0
    
    trill_data = []
    
    for idx, row in df.iterrows():
        filename = row['Filename']
        
        try:
            embedding = extractor.process_file(filename)
            
            if embedding is not None:
                trill_data.append({
                    'group_id': filename,
                    'embedding': embedding
                })
                loaded_count += 1
            else:
                missing_count += 1
                
        except Exception as e:
            print(f"  ✗ Error loading {filename}: {e}")
            missing_count += 1
    
    print(f"\nTrill loading complete:")
    print(f"  Loaded: {loaded_count}")
    print(f"  Missing: {missing_count}")
    
    return trill_data


def insert_features_to_database(whisper_data=None, opensmile_data=None, 
                                librosa_data=None, trill_data=None):
    """Insert extracted features into database."""
    print("\n" + "="*70)
    print("STEP 6: INSERTING FEATURES INTO DATABASE")
    print("="*70)
    
    connection = connect_to_db()
    
    # Insert Whisper features
    if whisper_data:
        print("\nInserting Whisper features...")
        
        # Create mean features dataframe
        mean_records = []
        for record in whisper_data:
            for i, val in enumerate(record['mean_features']):
                mean_records.append({
                    'group_id': record['group_id'],
                    'feat': f'whisper_mean_{i}',
                    'value': float(val)
                })
        
        if mean_records:
            whisper_mean_df = pd.DataFrame(mean_records)
            insert_feature_table_into_db(
                connection, whisper_mean_df, Config.DB_TABLES['whisper_mean']
            )
        
        # Create median features dataframe
        median_records = []
        for record in whisper_data:
            for i, val in enumerate(record['median_features']):
                median_records.append({
                    'group_id': record['group_id'],
                    'feat': f'whisper_median_{i}',
                    'value': float(val)
                })
        
        if median_records:
            whisper_median_df = pd.DataFrame(median_records)
            insert_feature_table_into_db(
                connection, whisper_median_df, Config.DB_TABLES['whisper_median']
            )
    
    # Insert Trill features
    if trill_data:
        print("\nInserting Trill features...")
        
        trill_records = []
        for record in trill_data:
            for i, val in enumerate(record['embedding']):
                trill_records.append({
                    'group_id': record['group_id'],
                    'feat': f'trill_mean_{i}',
                    'value': float(val)
                })
        
        if trill_records:
            trill_df = pd.DataFrame(trill_records)
            insert_feature_table_into_db(
                connection, trill_df, Config.DB_TABLES['trill']
            )
    
    connection.close()
    print("\n✓ Database insertion complete")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Audio Feature Extraction Pipeline'
    )
    parser.add_argument('--load-only', action='store_true',
                       help='Only load and merge data')
    parser.add_argument('--extract-whisper', action='store_true',
                       help='Extract Whisper features')
    parser.add_argument('--extract-opensmile', action='store_true',
                       help='Extract OpenSmile features')
    parser.add_argument('--extract-librosa', action='store_true',
                       help='Extract Librosa features')
    parser.add_argument('--load-trill', action='store_true',
                       help='Load existing Trill features')
    parser.add_argument('--insert-db', action='store_true',
                       help='Insert features into database')
    parser.add_argument('--all', action='store_true',
                       help='Run all steps')
    
    args = parser.parse_args()
    
    # If no arguments, show help
    if len(sys.argv) == 1:
        parser.print_help()
        return
    
    # Setup
    setup_pipeline()
    
    # Load and preprocess data
    merged_df = load_and_preprocess_data()
    
    if args.load_only:
        print("\n✓ Data loading complete. Exiting.")
        return
    
    # Initialize storage for extracted features
    whisper_data = None
    trill_data = None
    
    # Extract features based on arguments
    if args.extract_whisper or args.all:
        whisper_data = extract_whisper_features(merged_df)
    
    if args.extract_opensmile or args.all:
        extract_opensmile_features(merged_df)
    
    if args.extract_librosa or args.all:
        extract_librosa_features(merged_df)
    
    if args.load_trill or args.all:
        trill_data = load_trill_features(merged_df)
    
    # Insert into database
    if args.insert_db or args.all:
        insert_features_to_database(
            whisper_data=whisper_data,
            trill_data=trill_data
        )
    
    # Final summary
    print("\n" + "="*70)
    print("PIPELINE EXECUTION COMPLETE")
    print("="*70)
    print(f"Total records processed: {len(merged_df)}")
    print(f"Processed data saved to: {Config.PROCESSED_DATA_FILE}")
    print("="*70)


if __name__ == "__main__":
    main()
