# Audio Feature Extraction Pipeline

**Version 1.0.0** | Production-Ready Audio Analysis Tool

A comprehensive pipeline for extracting acoustic and linguistic features from audio data using state-of-the-art models including Whisper, OpenSmile, Librosa, and Trill.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Features](#features)
- [Pipeline Stages](#pipeline-stages)
- [Troubleshooting](#troubleshooting)
- [Advanced Usage](#advanced-usage)
- [Version History](#version-history)

---

## Overview

This pipeline processes audio files from the Klaatch dataset (2021-2023) and extracts multiple types of features:

- **Whisper Features**: Speech embeddings using OpenAI's Whisper model
- **OpenSmile Features**: 88 acoustic features (eGeMAPSv02 feature set)
- **Librosa Features**: 38 audio features (MFCCs, Chroma, Spectral Contrast, Tonnetz)
- **Trill Features**: Audio embeddings from Google's Trill model

### Key Capabilities

✅ **Modular Architecture** - Clean separation of concerns  
✅ **Command-Line Interface** - Easy execution with various options  
✅ **Multiple Feature Extractors** - 4 different extraction methods  
✅ **Database Integration** - MySQL storage and retrieval  
✅ **Error Handling** - Graceful failures with informative messages  
✅ **GPU Support** - Automatic CUDA detection for faster processing  
✅ **Skip Processed Files** - Avoids reprocessing existing features  

---

## Quick Start

Get up and running in **5 minutes**!

### 1. Initial Setup

```bash
# Navigate to the project directory
cd /home/karthik9/Audio_Analysis

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Linux/Mac
# or: venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Verify Python version (should be 3.9+)
python --version

# Check if CUDA is available (optional, for GPU acceleration)
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

### 2. Configure Environment

```bash
# Copy the example environment file
cp .env.example .env

# Edit with your paths and credentials
nano .env  # or use your preferred editor

# Verify configuration loads correctly
python -c "from config import Config; print('✓ Configuration loaded')"
```

Update these key settings:
```bash
AUDIO_ROOT=/path/to/your/audio/files
DB_HOST=localhost
DB_DATABASE=audio_analysis
DB_USERNAME=your_username
DB_PASSWORD=your_password
```

### 3. Verify Installation

```bash
# Test configuration
python -c "from config import Config; print('✓ Configuration loaded')"

# Test database connection (optional)
python -c "from src.database import test_connection; test_connection()"

# Verify key modules can be imported
python -c "import torch, transformers, librosa, opensmile; print('✓ All modules available')"
```

### 4. Run Your First Pipeline

💡 **Tip**: Start with `--load-only` to verify everything works before feature extraction.

**Option A: Just Load and Prepare Data**
```bash
python run_pipeline.py --load-only
```
This will load, merge, and preprocess all data.

**Option B: Extract Whisper Features**
```bash
python run_pipeline.py --extract-whisper
```

**Option C: Run Everything**
```bash
python run_pipeline.py --all
```
⚠️ **Warning**: This will take a long time for large datasets!

---

## Project Structure

```
Audio_Analysis/
├── 📋 Configuration
│   ├── .env.example              # Environment variables template
│   ├── .gitignore               # Git ignore patterns
│   ├── requirements.txt         # Python dependencies
│   └── config/
│       ├── __init__.py
│       └── config.py            # Centralized configuration
│
├── 🔧 Source Code (src/)
│   ├── __init__.py
│   ├── audio_utils.py           # Audio processing utilities
│   ├── data_loader.py           # Data loading and merging
│   ├── database.py              # Database operations
│   ├── text_processor.py        # Text preprocessing
│   └── extractors/
│       ├── __init__.py
│       ├── whisper_extractor.py    # Whisper features
│       ├── opensmile_extractor.py  # OpenSmile features
│       ├── librosa_extractor.py    # Librosa features
│       └── trill_extractor.py      # Trill features
│
├── 🚀 Executable Scripts
│   ├── run_pipeline.py          # Main pipeline (CLI)
│   └── setup.py                 # Setup & installation
│
├── 📜 Standalone Scripts (scripts/)
│   ├── audio_pattern_silencer.py
│   ├── audio_sentence_splitter.py
│   └── whisper_feature_extractor.py
│
├── 📓 Jupyter Notebooks (notebooks/)
│   ├── README.md
│   ├── Audio Feature Extraction Pipeline.ipynb
│   ├── Features Analysis.ipynb
│   └── ... (more notebooks)
│
└── 📚 README.md                 # This file
```

**Code Statistics:**
- 1,200+ lines of organized code
- 13 Python modules
- 3 standalone scripts
- 6 Jupyter notebooks
- Comprehensive documentation

---

## Installation

### System Requirements
- Python 3.9+
- CUDA-capable GPU (recommended for Whisper)
- MySQL database

### Step-by-Step Installation

#### 1. Navigate to the repository
```bash
cd /home/karthik9/Audio_Analysis
```

#### 2. Create virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows
```

#### 3. Install dependencies
```bash
pip install -r requirements.txt
```

#### 4. Configure environment
```bash
cp .env.example .env
nano .env  # Edit paths and credentials
```

#### 5. Verify installation
```bash
# Check Python version
python --version  # Should be 3.9+

# Test imports
python -c "from config import Config; print('✓ Ready')"

# Test database (optional)
python -c "from src.database import test_connection; test_connection()"
```

### Python Dependencies

Core packages (see `requirements.txt` for complete list):
- **Core**: numpy, pandas, torch, transformers
- **Audio**: librosa, soundfile, pydub, opensmile
- **ML**: scikit-learn
- **Database**: mysql-connector-python
- **Config**: python-dotenv

---

## Configuration

Edit the `.env` file to set your specific paths and database credentials:

```bash
# Database Configuration
DB_HOST=localhost
DB_DATABASE=audio_analysis
DB_USERNAME=your_username
DB_PASSWORD=your_password

# Directory Paths
DATA_DIR=.
AUDIO_ROOT=/path/to/audio/files
WHISPER_FEATURES_DIR=/path/to/whisper/output
TRILL_FEATURES_DIR=/path/to/trill/embeddings

# Input Files
ID_MAPPING_FILE=Klaatch Sanitized TextCEL 2021 to 2023 (1).xlsx
DEMOGRAPHICS_FILE=Demographics_Klaatch Sanitized TextCEL 2021 to 2023.xlsx
TRANSCRIPTS_FILE=Klaatch_transcripts.csv
NEW_TRANSCRIPTS_FILE=new_transcripts.csv

# Processing Options
TARGET_SAMPLE_RATE=16000
WHISPER_MODEL=openai/whisper-base
```

---

## Usage

### Main Pipeline Commands

The main pipeline offers flexible execution options:

#### Load and preprocess data only
```bash
python run_pipeline.py --load-only
```
- Loads all Excel and CSV files
- Merges demographics with transcripts
- Discovers audio files
- Cleans and preprocesses text
- Saves to `processed_klaatch_data.csv`

#### Extract specific features
```bash
# Whisper features
python run_pipeline.py --extract-whisper

# OpenSmile features
python run_pipeline.py --extract-opensmile

# Librosa features
python run_pipeline.py --extract-librosa

# Load existing Trill features
python run_pipeline.py --load-trill
```

#### Database operations
```bash
# Insert features into database
python run_pipeline.py --insert-db
```

#### Run all steps
```bash
python run_pipeline.py --all
```

#### Combine multiple steps
```bash
python run_pipeline.py --extract-whisper --load-trill --insert-db
```

### Standalone Scripts

The pipeline also includes standalone scripts for specific tasks:

#### Audio Pattern Silencer
Transcribes audio and silences specific words/patterns:
```bash
python scripts/audio_pattern_silencer.py
```

#### Audio Sentence Splitter
Splits audio files into sentence-level segments:
```bash
python scripts/audio_sentence_splitter.py
```

#### Whisper Feature Extractor
Extracts Whisper features from MP3 files:
```bash
python scripts/whisper_feature_extractor.py
```

### Using Jupyter Notebooks

```bash
# Start Jupyter
jupyter notebook notebooks/

# Or use JupyterLab
jupyter lab
```

See `notebooks/README.md` for details on each notebook.

### Common Workflows

**Workflow 1: Data Preparation Only**
```bash
python setup.py
python run_pipeline.py --load-only
```

**Workflow 2: Feature Extraction**
```bash
# Extract one feature type at a time for large datasets
python run_pipeline.py --extract-whisper
python run_pipeline.py --extract-librosa
python run_pipeline.py --load-trill
python run_pipeline.py --insert-db
```

**Workflow 3: Complete Pipeline**
```bash
python run_pipeline.py --all
```

---

## Features

### 🔧 Core Modules

**Data Loading & Merging** (`src/data_loader.py`)
- ID mapping loader
- Demographics loader
- Transcript loader
- Audio file discovery
- Complete data merging pipeline

**Text Preprocessing** (`src/text_processor.py`)
- Pattern removal (speaker labels, timestamps, etc.)
- Whitespace normalization
- Configurable cleaning rules

**Audio Processing** (`src/audio_utils.py`)
- Format conversion (MP3 → WAV)
- Audio preprocessing and resampling
- Feature computation utilities

**Database Operations** (`src/database.py`)
- Connection management
- Feature insertion functions
- Processing status tracking

### 🎵 Feature Extractors

**1. Whisper Extractor** (`src/extractors/whisper_extractor.py`)
- Model: OpenAI Whisper (configurable)
- Output: Encoder embeddings
- Statistics: Mean and median features

**2. OpenSmile Extractor** (`src/extractors/opensmile_extractor.py`)
- Feature Set: eGeMAPSv02
- Output: 88 acoustic features
- Includes: Pitch, energy, spectral features, voice quality

**3. Librosa Extractor** (`src/extractors/librosa_extractor.py`)
- 13 MFCCs (Mel-frequency cepstral coefficients)
- 12 Chroma features (pitch class profiles)
- 7 Spectral contrast bands
- 6 Tonnetz features (tonal centroids)
- **Total: 38 features**

**4. Trill Extractor** (`src/extractors/trill_extractor.py`)
- Pre-computed embedding loading
- Multiple naming pattern support
- Dimension: Typically 1024

---

## Pipeline Stages

### Stage 1: Data Loading and Merging

The pipeline loads data from multiple sources:
- **ID mappings** (Excel)
- **Demographics data** (Excel)
- **Transcripts** (CSV)
- **Audio files** (MP3)

All sources are merged into a single dataset with consistent identifiers.

### Stage 2: Text Preprocessing

Transcripts are cleaned by removing:
- Speaker labels (e.g., "Speaker1", "Speaker2")
- Timestamps (e.g., `[00:24:00]`)
- Response options (e.g., "strongly agree", "disagree")
- Special characters and extra whitespace

### Stage 3: Feature Extraction

#### Whisper Features
- **Model**: `openai/whisper-base` (configurable)
- **Output**: Encoder embeddings (sequence_length × hidden_size)
- **Statistics**: Mean and median features computed
- **Format**: NumPy arrays saved as `.npy` files

#### OpenSmile Features
- **Feature Set**: eGeMAPSv02 (Geneva Minimalistic Acoustic Parameter Set)
- **Output**: 88 acoustic features
- **Includes**: Pitch, energy, spectral features, voice quality
- **Format**: JSON or database columns

#### Librosa Features
- **Output**: 38 features
  - 13 MFCCs (Mel-frequency cepstral coefficients)
  - 12 Chroma features (pitch class profiles)
  - 7 Spectral contrast bands
  - 6 Tonnetz features (tonal centroids)
- **Format**: Individual database columns

#### Trill Features
- **Source**: Pre-computed embeddings loaded from disk
- **Dimension**: Typically 1024
- **Processing**: Mean pooling applied if multi-dimensional

### Stage 4: Database Storage

Features are stored in MySQL database with the following tables:

**Main Data Table:**
```sql
CREATE TABLE merged_data (
    message_id VARCHAR(255) PRIMARY KEY,
    message TEXT,
    KlaatchID VARCHAR(50),
    Date VARCHAR(50),
    CEL_Total FLOAT,
    CELVAL1 FLOAT,
    CELVAL2 FLOAT,
    CELVAL3 FLOAT,
    Age INT
);
```

**Feature Tables:**
- `feat$whisper_mean_n$merged_data$message_id`: Whisper mean features
- `feat$whisper_median_n$merged_data$message_id`: Whisper median features
- `feat$opensmile_n$merged_data$message_id`: OpenSmile features
- `feat$librosa_n$merged_data$message_id`: Librosa features
- `feat$trill_mstd$merged_data$message_id`: Trill features

---

## Troubleshooting

### "Module not found" errors

```bash
# Reinstall dependencies
pip install -r requirements.txt

# Verify virtual environment is activated
which python  # Should point to venv
```

### Database Connection Issues

```bash
# Test connection
python -c "from src.database import test_connection; test_connection()"

# Verify MySQL is running
sudo systemctl status mysql

# Check credentials in .env file
cat .env | grep DB_
```

### "Audio files not found"

```bash
# Verify the path
ls /path/to/your/audio/files/*.mp3

# Update AUDIO_ROOT in .env
nano .env
```

### Memory Issues with Whisper

```bash
# Use a smaller Whisper model in .env:
WHISPER_MODEL=openai/whisper-tiny

# Or process fewer files at a time
# Edit run_pipeline.py to process in batches
```

### Import Errors

```bash
# Ensure all dependencies are installed
pip install -r requirements.txt

# Check Python version (should be 3.9+)
python --version

# Activate virtual environment if using one
source venv/bin/activate
```

### Out of Memory Errors

**Solutions:**
1. Use smaller Whisper model (`whisper-tiny` or `whisper-small`)
2. Process files in smaller batches
3. Reduce batch size in processing loops
4. Ensure sufficient GPU memory (for CUDA)

### Missing Audio Files

**Checks:**
1. Verify `AUDIO_ROOT` path in `.env`
2. Ensure audio files are in MP3 format
3. Check file naming: `{ID}_{Date}_{Time}.mp3`
4. Verify file permissions

---

## Advanced Usage

### Custom Feature Extractors

Create custom feature extractors by extending the base pattern:

```python
from src.extractors import WhisperExtractor

class CustomExtractor(WhisperExtractor):
    def extract_features(self, waveform, sample_rate):
        # Your custom extraction logic
        features = your_model(waveform)
        return features
```

Add to `src/extractors/__init__.py`:
```python
from .custom_extractor import CustomExtractor

__all__ = [..., 'CustomExtractor']
```

### Batch Processing

For large datasets, process in batches:

```python
from src.data_loader import load_and_merge_all_data
from src.extractors import WhisperExtractor

# Load data
df = load_and_merge_all_data()

# Initialize extractor
extractor = WhisperExtractor()

# Process in batches
batch_size = 100
for i in range(0, len(df), batch_size):
    batch = df.iloc[i:i+batch_size]
    
    for idx, row in batch.iterrows():
        if pd.notna(row['Filepath']):
            features_path = extractor.process_file(
                row['Filepath'], 
                row['Filename']
            )
```

### Adding New Data Sources

1. **Add loading function** to `src/data_loader.py`:
```python
def load_new_source() -> pd.DataFrame:
    """Load new data source."""
    df = pd.read_csv('new_source.csv')
    return df
```

2. **Update merge logic** in `load_and_merge_all_data()`:
```python
new_data = load_new_source()
merged_df = merged_df.merge(new_data, on='id', how='left')
```

### Updating Configuration

1. Add to `.env.example`:
```bash
NEW_SETTING=default_value
```

2. Add to `config/config.py`:
```python
class Config:
    NEW_SETTING = os.getenv('NEW_SETTING', 'default_value')
```

3. Use in code:
```python
from config import Config
value = Config.NEW_SETTING
```

### Performance Optimization

**Tips for Large Datasets:**

1. **Use GPU**: Much faster for Whisper
   ```bash
   # Check CUDA availability
   python -c "import torch; print(torch.cuda.is_available())"
   ```

2. **Batch Processing**: Process files in batches

3. **Skip Processed**: The pipeline automatically skips already-processed files

4. **Parallel Processing**: Consider using multiprocessing for CPU-bound tasks

5. **Monitor Progress**: Track processing with logging

---

## Version History

### Version 1.0.0 (2026-01-29) - Initial Release

**What Was Created:**
- Complete refactor from Jupyter notebook to production code
- Modular architecture with separated concerns
- 1,200+ lines of organized Python code
- Comprehensive documentation
- Command-line interface
- 4 feature extraction methods

**Key Additions:**
- ✅ Configuration management via environment variables
- ✅ Database integration (MySQL)
- ✅ Error handling and validation
- ✅ Processing status tracking
- ✅ Skip already-processed files
- ✅ GPU support (CUDA)
- ✅ Organized notebooks in `notebooks/` folder
- ✅ Standalone scripts in `scripts/` folder

**Modules Created:**
- `config/config.py` - Centralized configuration
- `src/data_loader.py` - Data loading and merging
- `src/text_processor.py` - Text preprocessing
- `src/audio_utils.py` - Audio utilities
- `src/database.py` - Database operations
- `src/extractors/whisper_extractor.py` - Whisper features
- `src/extractors/opensmile_extractor.py` - OpenSmile features
- `src/extractors/librosa_extractor.py` - Librosa features
- `src/extractors/trill_extractor.py` - Trill features

**Migration from Notebook:**
- Converted notebook cells to modular Python modules
- Extracted configuration to environment variables
- Separated data loading, processing, and extraction
- Added command-line interface
- Improved error handling and progress tracking

---

## Output Files

- **`processed_klaatch_data.csv`**: Merged and preprocessed dataset
- **Whisper features**: `<WHISPER_FEATURES_DIR>/<filename>_whisper.npy`
- **Database**: Features stored in MySQL tables

---

## Citation

If you use this pipeline in your research, please cite:

```
[Add citation information here]
```

---

## License

[Add license information here]

---

## Contact

[Add contact information here]

---

## Acknowledgments

This pipeline uses the following open-source projects:
- **OpenAI Whisper** - Speech recognition and feature extraction
- **OpenSMILE** - Audio feature extraction toolkit
- **Librosa** - Audio and music analysis
- **Google Trill** - Audio embedding model
- **PyTorch** - Deep learning framework
- **Transformers (Hugging Face)** - Pre-trained models

---

## Status

**Production-Ready ✅**

Your audio analysis pipeline is ready to use!

**Last Updated**: 2026-01-29  
**Version**: 1.0.0
