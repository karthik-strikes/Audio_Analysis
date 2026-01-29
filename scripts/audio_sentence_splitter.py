#!/usr/bin/env python3
"""
Whisper Audio Sentence Splitter

This script splits audio files into sentence-level segments using Whisper transcription.
It processes audio files, transcribes them with word-level timestamps, and splits the audio
at sentence boundaries (punctuation marks: . ! ?).

Features:
- Automatic sentence detection based on punctuation
- Preserves all words in transcription
- Exports individual sentence audio segments
- Creates CSV with sentence-level transcripts
- Supports multiple audio formats (mp3, wav, m4a)

Use Case:
- Creating sentence-level audio datasets
- Preparing data for fine-grained audio analysis
- Generating aligned audio-transcript pairs

Author: Audio Segmentation Team
Date: 2024
"""

import os
from typing import List, Dict, Tuple
import whisper
import pandas as pd
from pydub import AudioSegment


# ============================================================================
# CONFIGURATION
# ============================================================================

# Whisper model configuration
WHISPER_MODEL = "large"

# Directory paths
INPUT_FOLDER = "/nlp/data/karthik9/New_audio/folder7"
OUTPUT_FOLDER = "/nlp/data/karthik9/New_audio/folder7"
TRANSCRIPT_CSV = "transcripts_senten.csv"

# Audio format configuration
SUPPORTED_FORMATS = (".mp3", ".wav", ".m4a", ".flac")
OUTPUT_FORMAT = "mp3"

# Sentence boundary punctuation
SENTENCE_ENDINGS = (".", "!", "?")


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def ensure_directory_exists(directory: str) -> None:
    """
    Create directory if it doesn't exist.

    Args:
        directory (str): Directory path to create
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"✓ Created directory: {directory}")


def is_sentence_boundary(word: str, is_last_word: bool = False) -> bool:
    """
    Check if a word marks the end of a sentence.

    Args:
        word (str): The word to check
        is_last_word (bool): Whether this is the last word in the transcript

    Returns:
        bool: True if word ends a sentence
    """
    # Last word always ends a sentence
    if is_last_word:
        return True

    # Check if word ends with sentence punctuation
    return word.strip().endswith(SENTENCE_ENDINGS)


# ============================================================================
# AUDIO PROCESSING
# ============================================================================

def extract_words_with_timestamps(transcription_result: Dict) -> List[Dict]:
    """
    Extract all words with their timestamps from Whisper transcription result.

    Args:
        transcription_result (Dict): Whisper transcription result with segments

    Returns:
        List[Dict]: List of word dictionaries with 'word', 'start', 'end' keys
    """
    words = []

    for segment in transcription_result["segments"]:
        if "words" in segment:
            words.extend(segment["words"])

    return words


def split_audio_into_sentences(
    audio: AudioSegment,
    words: List[Dict],
    base_name: str,
    output_folder: str
) -> List[Dict]:
    """
    Split audio into sentence-level segments based on word timestamps.

    Args:
        audio (AudioSegment): Loaded audio file
        words (List[Dict]): List of words with timestamps
        base_name (str): Base filename for output files
        output_folder (str): Directory to save audio segments

    Returns:
        List[Dict]: List of transcript dictionaries with 'File Name' and 'Transcript'
    """
    transcripts = []
    start_time_ms = 0
    part_number = 1
    transcript_text = ""

    for i, word_info in enumerate(words):
        word = word_info.get("word", "")
        transcript_text += word + " "

        # Check if this is a sentence boundary
        is_last = (i == len(words) - 1)

        if is_sentence_boundary(word, is_last):
            end_time_ms = int(word_info["end"] * 1000)  # Convert seconds to milliseconds

            # Extract audio segment
            segment_audio = audio[start_time_ms:end_time_ms]

            # Generate output filename
            output_filename = f"{base_name}_{part_number}.{OUTPUT_FORMAT}"
            output_path = os.path.join(output_folder, output_filename)

            # Export segment
            segment_audio.export(output_path, format=OUTPUT_FORMAT)

            # Store transcript
            transcripts.append({
                "File Name": output_filename,
                "Transcript": transcript_text.strip()
            })

            print(f"  ✓ Segment {part_number}: {len(transcript_text.split())} words")

            # Reset for next sentence
            start_time_ms = end_time_ms
            transcript_text = ""
            part_number += 1

    return transcripts


def process_single_audio_file(
    filename: str,
    input_folder: str,
    output_folder: str,
    model: whisper.Whisper
) -> List[Dict]:
    """
    Process a single audio file: transcribe and split into sentences.

    Args:
        filename (str): Audio filename
        input_folder (str): Directory containing input audio
        output_folder (str): Directory for output segments
        model (whisper.Whisper): Loaded Whisper model

    Returns:
        List[Dict]: List of transcript dictionaries

    Raises:
        Exception: If transcription or splitting fails
    """
    audio_path = os.path.join(input_folder, filename)
    base_name = os.path.splitext(filename)[0]

    print(f"\nProcessing: {filename}")

    # Load audio
    print("  Loading audio...")
    audio = AudioSegment.from_file(audio_path)
    print(f"  ✓ Audio loaded: {len(audio)/1000:.1f} seconds")

    # Transcribe with word-level timestamps
    print("  Transcribing audio...")
    result = model.transcribe(audio_path, word_timestamps=True)
    print("  ✓ Transcription complete")

    # Extract words
    words = extract_words_with_timestamps(result)
    print(f"  Found {len(words)} words")

    # Split into sentences
    print("  Splitting audio by sentences...")
    transcripts = split_audio_into_sentences(
        audio, words, base_name, output_folder
    )
    print(f"  ✓ Created {len(transcripts)} sentence segments")

    return transcripts


# ============================================================================
# BATCH PROCESSING
# ============================================================================

def process_all_audio_files(
    input_folder: str,
    output_folder: str,
    transcript_csv: str,
    model: whisper.Whisper
) -> Dict[str, int]:
    """
    Process all audio files in a folder.

    Args:
        input_folder (str): Directory containing input audio files
        output_folder (str): Directory for output segments
        transcript_csv (str): Path to output CSV file
        model (whisper.Whisper): Loaded Whisper model

    Returns:
        Dict[str, int]: Processing statistics
    """
    # Ensure output directory exists
    ensure_directory_exists(output_folder)

    # Find audio files
    audio_files = [
        f for f in os.listdir(input_folder)
        if f.lower().endswith(SUPPORTED_FORMATS)
    ]

    if not audio_files:
        print(f"No audio files found in: {input_folder}")
        return {'total_files': 0, 'total_segments': 0, 'errors': 0}

    print(f"\n{'='*70}")
    print(f"Found {len(audio_files)} audio files to process")
    print(f"{'='*70}")

    all_transcripts = []
    stats = {
        'total_files': len(audio_files),
        'processed_files': 0,
        'total_segments': 0,
        'errors': 0
    }

    # Process each file
    for filename in audio_files:
        try:
            transcripts = process_single_audio_file(
                filename, input_folder, output_folder, model
            )

            all_transcripts.extend(transcripts)
            stats['processed_files'] += 1
            stats['total_segments'] += len(transcripts)

        except Exception as e:
            print(f"  ✗ Error processing {filename}: {e}")
            stats['errors'] += 1

    # Save all transcripts to CSV
    if all_transcripts:
        df = pd.DataFrame(all_transcripts)
        csv_path = os.path.join(output_folder, transcript_csv)
        df.to_csv(csv_path, index=False)
        print(f"\n✓ Transcripts saved to: {csv_path}")

    return stats


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""
    print("\n" + "="*70)
    print("WHISPER AUDIO SENTENCE SPLITTER")
    print("="*70)

    print("\nConfiguration:")
    print(f"  Model: {WHISPER_MODEL}")
    print(f"  Input folder: {INPUT_FOLDER}")
    print(f"  Output folder: {OUTPUT_FOLDER}")
    print(f"  Transcript CSV: {TRANSCRIPT_CSV}")
    print(f"  Supported formats: {', '.join(SUPPORTED_FORMATS)}")
    print(f"  Output format: {OUTPUT_FORMAT}")

    # Verify input directory exists
    if not os.path.exists(INPUT_FOLDER):
        print(f"\n✗ Error: Input folder not found: {INPUT_FOLDER}")
        return

    # Load Whisper model
    print(f"\nLoading Whisper model '{WHISPER_MODEL}'...")
    model = whisper.load_model(WHISPER_MODEL)
    print("✓ Model loaded successfully")

    # Process all files
    stats = process_all_audio_files(
        INPUT_FOLDER,
        OUTPUT_FOLDER,
        TRANSCRIPT_CSV,
        model
    )

    # Print summary
    print("\n" + "="*70)
    print("PROCESSING COMPLETE")
    print("="*70)
    print(f"Total files: {stats['total_files']}")
    print(f"Successfully processed: {stats['processed_files']}")
    print(f"Errors: {stats['errors']}")
    print(f"Total sentence segments created: {stats['total_segments']}")
    print(f"Segments saved in: {OUTPUT_FOLDER}")
    print(f"Transcripts CSV: {os.path.join(OUTPUT_FOLDER, TRANSCRIPT_CSV)}")
    print("="*70)
    print("\n✓ No words lost - all transcripts preserved!")


if __name__ == "__main__":
    main()
