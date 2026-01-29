#!/usr/bin/env python3
"""
Audio Silence Processor with Whisper Transcription

This script processes audio files to:
1. Transcribe audio using OpenAI's Whisper model
2. Identify specific words/patterns (e.g., "strongly agree", "disagree")
3. Replace identified words with silence in the audio
4. Export modified audio and transcripts

The script is useful for anonymizing or censoring specific content in audio recordings
while maintaining transcript records.

Author: Audio Processing Team
Date: 2024
"""

import os
import re
import csv
from typing import List, Dict, Tuple
from pydub import AudioSegment
from pydub.generators import Sine
import whisper


# ============================================================================
# CONFIGURATION
# ============================================================================

# Model configuration
WHISPER_MODEL = "small.en"

# Directory paths
INPUT_FOLDER = "/nlp/data/karthik9/Zip/T1 2021"
OUTPUT_FOLDER = "/nlp/data/karthik9/New_audio"
CSV_FILE = "/nlp/data/karthik9/transcripts1.csv"

# Patterns to identify and silence
CENSORED_PATTERNS = [
    r'\bstrongly agree\w*\b',
    r'\bstrongly disagree\w*\b',
    r'\bstrongly_agree\w*\b',
    r'\bstrongly_disagree\w*\b',
    r'\bagree\w*\b',       # Matches agree, agreed, agreement, agreeable, agreeing
    r'\bdisagree\w*\b',    # Matches disagree, disagreed, disagreement, etc.
    r"\bdon't\w*\b",       # Matches don't and its extensions
    r'\bnot\w*\b',         # Matches not and its extensions
    r'\bstrongly\w*\b',    # Matches strongly and extensions
    r'\bdefinitely\w*\b',  # Matches definitely and extensions
    r'\bneither\w*\b'      # Matches neither and extensions
]

# Audio settings
BEEP_DURATION_MS = 500  # Duration of beep in milliseconds (currently unused)
BEEP_FREQUENCY_HZ = 1000  # Beep frequency in Hz (currently unused)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def initialize_csv_file(csv_path: str) -> None:
    """
    Initialize CSV file with headers if it doesn't exist.

    Args:
        csv_path (str): Path to the CSV file
    """
    if not os.path.exists(csv_path):
        with open(csv_path, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(["Filename", "Transcript"])
        print(f"✓ Created CSV file: {csv_path}")


def find_pattern_matches(segments: List[Dict], patterns: List[str]) -> List[Dict]:
    """
    Find words matching specified patterns in transcription segments.

    Args:
        segments (List[Dict]): Whisper transcription segments with word-level timestamps
        patterns (List[str]): Regex patterns to match

    Returns:
        List[Dict]: List of matched words with timestamps
            Each dict contains: {'word': str, 'start': float, 'end': float}
    """
    matches = []

    for segment in segments:
        for word_info in segment.get('words', []):
            word = word_info.get('word', '')
            for pattern in patterns:
                if re.search(pattern, word, re.IGNORECASE):
                    matches.append({
                        'word': word,
                        'start': word_info['start'],
                        'end': word_info['end']
                    })
                    break  # Only match once per word

    return matches


def silence_audio_segments(audio: AudioSegment, matches: List[Dict]) -> AudioSegment:
    """
    Replace specified audio segments with silence.

    Args:
        audio (AudioSegment): Original audio
        matches (List[Dict]): List of segments to silence with 'start' and 'end' times

    Returns:
        AudioSegment: Modified audio with silenced segments
    """
    modified_audio = audio

    # Sort matches by start time (descending) to modify from end to start
    # This prevents timing shifts during modification
    sorted_matches = sorted(matches, key=lambda x: x['start'], reverse=True)

    for match in sorted_matches:
        start_ms = int(match['start'] * 1000)  # Convert seconds to milliseconds
        end_ms = int(match['end'] * 1000)
        duration_ms = end_ms - start_ms

        # Create silent segment
        silent_segment = AudioSegment.silent(duration=duration_ms)

        # Replace audio segment with silence
        modified_audio = modified_audio[:start_ms] + silent_segment + modified_audio[end_ms:]

    return modified_audio


def append_to_csv(csv_path: str, filename: str, transcript: str) -> None:
    """
    Append transcript entry to CSV file.

    Args:
        csv_path (str): Path to CSV file
        filename (str): Audio filename
        transcript (str): Transcription text
    """
    with open(csv_path, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow([filename, transcript])


def ensure_directory_exists(directory: str) -> None:
    """
    Create directory if it doesn't exist.

    Args:
        directory (str): Directory path
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"✓ Created directory: {directory}")


# ============================================================================
# MAIN PROCESSING FUNCTION
# ============================================================================

def process_audio_file(
    audio_path: str,
    output_folder: str,
    model: whisper.Whisper,
    patterns: List[str]
) -> Tuple[str, int]:
    """
    Process a single audio file: transcribe, identify patterns, and silence matches.

    Args:
        audio_path (str): Path to input audio file
        output_folder (str): Directory for output files
        model (whisper.Whisper): Loaded Whisper model
        patterns (List[str]): Regex patterns to identify and silence

    Returns:
        Tuple[str, int]: (transcript text, number of matches found)
    """
    filename = os.path.basename(audio_path)
    print(f"\nProcessing: {filename}")

    # Transcribe with word-level timestamps
    print("  Transcribing audio...")
    result = model.transcribe(audio_path, word_timestamps=True)
    transcript = result['text']

    # Find pattern matches
    print("  Identifying patterns...")
    matches = find_pattern_matches(result['segments'], patterns)
    print(f"  Found {len(matches)} matches to silence")

    if matches:
        # Load audio
        print("  Loading audio...")
        audio = AudioSegment.from_file(audio_path)

        # Silence matched segments
        print("  Silencing matched segments...")
        modified_audio = silence_audio_segments(audio, matches)

        # Generate output path
        base_name, ext = os.path.splitext(filename)
        output_filename = f"{base_name}_wb.mp3"
        output_path = os.path.join(output_folder, output_filename)

        # Export modified audio
        print("  Exporting modified audio...")
        modified_audio.export(output_path, format="mp3")
        print(f"✓ Saved: {output_path}")
    else:
        print("  No matches found - skipping audio modification")

    return transcript, len(matches)


def process_all_audio_files(
    input_folder: str,
    output_folder: str,
    csv_file: str,
    model: whisper.Whisper,
    patterns: List[str]
) -> Dict[str, int]:
    """
    Process all audio files in the input folder.

    Args:
        input_folder (str): Directory containing input audio files
        output_folder (str): Directory for output files
        csv_file (str): Path to CSV file for transcripts
        model (whisper.Whisper): Loaded Whisper model
        patterns (List[str]): Regex patterns to identify and silence

    Returns:
        Dict[str, int]: Statistics about processing
    """
    # Ensure output directory exists
    ensure_directory_exists(output_folder)

    # Initialize CSV
    initialize_csv_file(csv_file)

    # Get list of audio files
    audio_extensions = ('.wav', '.mp3', '.m4a', '.flac')
    audio_files = [f for f in os.listdir(input_folder) if f.lower().endswith(audio_extensions)]

    print(f"\n{'='*70}")
    print(f"Found {len(audio_files)} audio files to process")
    print(f"{'='*70}")

    # Process statistics
    stats = {
        'total_files': len(audio_files),
        'processed': 0,
        'total_matches': 0,
        'errors': 0
    }

    # Process each file
    for file_name in audio_files:
        audio_path = os.path.join(input_folder, file_name)

        try:
            transcript, num_matches = process_audio_file(
                audio_path, output_folder, model, patterns
            )

            # Append transcript to CSV
            append_to_csv(csv_file, file_name, transcript)

            # Update statistics
            stats['processed'] += 1
            stats['total_matches'] += num_matches

        except Exception as e:
            print(f"✗ Error processing {file_name}: {e}")
            stats['errors'] += 1

    return stats


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""
    print("\n" + "="*70)
    print("AUDIO SILENCE PROCESSOR WITH WHISPER TRANSCRIPTION")
    print("="*70)

    print("\nConfiguration:")
    print(f"  Model: {WHISPER_MODEL}")
    print(f"  Input folder: {INPUT_FOLDER}")
    print(f"  Output folder: {OUTPUT_FOLDER}")
    print(f"  CSV file: {CSV_FILE}")
    print(f"  Patterns to censor: {len(CENSORED_PATTERNS)}")

    # Load Whisper model
    print(f"\nLoading Whisper model '{WHISPER_MODEL}'...")
    model = whisper.load_model(WHISPER_MODEL)
    print("✓ Model loaded successfully")

    # Process all files
    stats = process_all_audio_files(
        INPUT_FOLDER,
        OUTPUT_FOLDER,
        CSV_FILE,
        model,
        CENSORED_PATTERNS
    )

    # Print summary
    print("\n" + "="*70)
    print("PROCESSING COMPLETE")
    print("="*70)
    print(f"Total files: {stats['total_files']}")
    print(f"Processed: {stats['processed']}")
    print(f"Errors: {stats['errors']}")
    print(f"Total matches silenced: {stats['total_matches']}")
    print(f"Transcripts saved to: {CSV_FILE}")
    print("="*70)


if __name__ == "__main__":
    main()
