# Multimodal Loneliness Prediction Pipeline

**Version 1.0.0** | Complete Analysis Pipeline from Audio/Text Features to Prediction

A comprehensive end-to-end pipeline for predicting loneliness from multimodal features (audio + text). This repository contains the complete workflow from feature extraction through statistical analysis to machine learning prediction models.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Analysis Workflow](#analysis-workflow)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Features](#features)
- [Notebooks](#notebooks)
- [Troubleshooting](#troubleshooting)
- [Advanced Usage](#advanced-usage)
- [Version History](#version-history)

---

## Overview

This pipeline processes audio and text data from the Klaatch dataset (2021-2023) to predict emotional loneliness scores. It combines acoustic features, linguistic features, and demographic data through a multi-stage analysis workflow.

### Feature Types

**Audio Features:**
- **Whisper Features**: Speech embeddings using OpenAI's Whisper model
- **OpenSmile Features**: 88 acoustic features (eGeMAPSv02 feature set)
- **Librosa Features**: 38 audio features (MFCCs, Chroma, Spectral Contrast, Tonnetz)
- **Trill Features**: Audio embeddings from Google's Trill model

**Text/Linguistic Features** (via [DLATK](https://dlatk.github.io/dlatk/)):
- **LIWC2022**: Linguistic Inquiry and Word Count categories
- **LDA Topics**: Latent Dirichlet Allocation topic modeling
- **N-grams**: 1-3 gram language features with PMI filtering

**Demographic Features:**
- Age, Gender, Race
- CEL (UCLA Loneliness Scale) scores

### Key Capabilities

✅ **Complete Analysis Pipeline** - From raw audio to prediction models
✅ **Multimodal Features** - Audio + Text + Demographics
✅ **DLATK Integration** - Linguistic feature extraction via DLATK commands
✅ **Propensity Score Matching** - Fair demographic subgroup analysis
✅ **Participant-Level CV** - Prevents data leakage in cross-validation
✅ **Statistical Analysis** - Correlation tables with Bonferroni correction
✅ **Multiple ML Models** - ExtraTrees regression for prediction
✅ **Database Integration** - MySQL storage and retrieval
✅ **GPU Support** - Automatic CUDA detection for faster processing

---

## Analysis Workflow

The complete analysis follows this sequence:

### 1️⃣ **Feature Extraction** → `Audio Feature Extraction Pipeline.ipynb`
Extract acoustic features from audio files:
- Whisper embeddings
- OpenSmile acoustic features
- Librosa audio features
- Trill embeddings
- Store in MySQL database

### 2️⃣ **Propensity Score Matching** → `Propensity Score Matching Analysis.ipynb`
Create balanced demographic subgroups:
- Calculate propensity scores using logistic regression
- 1:1 nearest neighbor matching
- Generate balanced datasets: `stratified_male`, `stratified_female`, `stratified_black`, `stratified_white`
- Ensure fair model evaluation across demographics

### 3️⃣ **Data Stratification & Cross-Validation** → `Data Stratification for Cross-Validation.ipynb`
Prepare data for participant-level cross-validation:
- Split by participant ID (not message ID) to prevent data leakage
- Create stratified folds for training/testing
- Balance demographic groups across folds

### 4️⃣ **Feature Analysis (DLATK)** → `Features Analysis.ipynb`
Extract and analyze linguistic features using [DLATK](https://dlatk.github.io/dlatk/):
- **LIWC2022**: Psychological language categories
- **LDA Topics**: Topic modeling on text data
- **N-grams**: 1-3 gram features with PMI filtering
- Run analysis on **total dataset** and **stratified subgroups** (male, female, black, white)
- Generate correlation tables (Tables 2, 3, S1-S9)
- Statistical analysis with Bonferroni correction

**Note:** This notebook contains DLATK command-line calls in cells. See [DLATK documentation](https://dlatk.github.io/dlatk/) for command syntax.

### 5️⃣ **Prediction Models** → `Predicting Loneliness from Multimodal Features.ipynb`
Train and evaluate machine learning models:
- ExtraTrees regression with hyperparameter tuning
- Feature combination strategies (Combined Text, Combined Audio, Multimodal)
- Participant-level cross-validation
- Feature importance analysis
- Performance evaluation by demographic subgroups
- Generate results tables (Table 4, prediction metrics)
- Pearson correlations with confidence intervals  

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
├── 📓 Jupyter Notebooks (notebooks/) - Complete Analysis Pipeline
│   ├── README.md
│   ├── 1. Audio Feature Extraction Pipeline.ipynb
│   ├── 2. Propensity Score Matching Analysis.ipynb
│   ├── 3. Data Stratification for Cross-Validation.ipynb
│   ├── 4. Features Analysis.ipynb (DLATK commands)
│   ├── 5. Predicting Loneliness from Multimodal Features.ipynb
│   └── 6. Topic Messages Analysis.ipynb
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
- **Text/Linguistic**: DLATK (Differential Language Analysis ToolKit) - see [installation guide](https://dlatk.github.io/dlatk/install.html)
- **ML**: scikit-learn
- **Database**: mysql-connector-python
- **Config**: python-dotenv

**Note on DLATK:** DLATK is required for linguistic feature extraction (LIWC, LDA, N-grams) in the `Features Analysis.ipynb` notebook. Install separately following the [DLATK installation guide](https://dlatk.github.io/dlatk/install.html).

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

---

## Notebooks

The `notebooks/` folder contains the complete analysis pipeline as Jupyter notebooks. Each notebook represents a stage in the workflow and can be run independently or in sequence.

### Running Notebooks

```bash
# Start Jupyter
jupyter notebook notebooks/

# Or use JupyterLab
jupyter lab
```

### Notebook Workflow (in order)

#### 1. Audio Feature Extraction Pipeline.ipynb
**Stage**: Feature Extraction (Audio)

**What it does:**
- Loads and merges data from multiple sources (demographics, transcripts, audio files)
- Extracts Whisper embeddings from audio
- Extracts OpenSmile acoustic features (88 features)
- Extracts Librosa audio features (38 features)
- Loads Trill pre-computed embeddings
- Stores all features in MySQL database

**Output:**
- Database tables with audio features
- Processed data ready for analysis

---

#### 2. Propensity Score Matching Analysis.ipynb
**Stage**: Demographic Balancing

**What it does:**
- Calculates propensity scores using logistic regression
- Performs 1:1 nearest neighbor matching
- Creates balanced demographic subgroups:
  - `stratified_male` / `stratified_female`
  - `stratified_black` / `stratified_white`
- Evaluates model fairness across subgroups
- Exports matched datasets to database

**Output:**
- 4 balanced demographic datasets
- Subgroup performance metrics (MAE, R²)

**Key Functions:**
- `calculate_propensity_scores()` - Propensity score calculation
- `perform_propensity_matching()` - 1:1 matching algorithm
- `evaluate_subgroup_performance()` - Model evaluation by group

---

#### 3. Data Stratification for Cross-Validation.ipynb
**Stage**: Cross-Validation Setup

**What it does:**
- Creates participant-level train/test splits (NOT message-level)
- Prevents data leakage by keeping all messages from one participant in the same fold
- Generates stratified folds for cross-validation
- Verifies balance across demographic groups

**Output:**
- Cross-validation fold assignments
- Balanced training/testing splits

**Important:** Splits by `klaatch_id` (participant ID), ensuring messages from the same person don't appear in both train and test sets.

---

#### 4. Features Analysis.ipynb
**Stage**: Linguistic Feature Extraction & Statistical Analysis

**What it does:**
- Extracts linguistic features using **DLATK** ([Differential Language Analysis ToolKit](https://dlatk.github.io/dlatk/))
- Runs DLATK commands for:
  - **LIWC2022**: Linguistic Inquiry and Word Count categories
  - **LDA Topics**: Latent Dirichlet Allocation topic modeling
  - **N-grams**: 1-3 gram language features with PMI filtering
- Analyzes features for:
  - **Total dataset** (all participants)
  - **Stratified subgroups** (male, female, black, white)
- Computes correlations with Bonferroni correction
- Generates correlation tables (Tables 2, 3, S1-S9)

**Output:**
- LIWC feature tables in database
- LDA topic distributions
- N-gram features (1-3 grams with PMI ≥ 6.0)
- Statistical correlation tables

**Note on DLATK:**
This notebook contains DLATK command-line calls within cells. DLATK is a command-line tool, so the notebook preserves the exact commands used. Example:
```bash
dlatkInterface.py -d audio_analysis -t merged_data -c message_id \
    --add_liwc --liwc_table feat$cat_LIWC2022_lw$merged_data$message_id$1gra \
    --outcome_table stratified_female --outcomes CEL_Total
```

See [DLATK documentation](https://dlatk.github.io/dlatk/) for command syntax and options.

**Some duplicate code may exist across notebooks** - this is intentional for reproducibility and to keep each notebook self-contained.

---

#### 5. Predicting Loneliness from Multimodal Features.ipynb
**Stage**: Machine Learning Prediction

**What it does:**
- Combines all features (audio + text + demographics)
- Creates feature combinations:
  - **Combined (Text)**: LIWC + LDA + N-grams
  - **Combined (Audio)**: Whisper + OpenSmile + Librosa + Trill
  - **Multimodal**: Text + Audio combined
- Trains ExtraTrees regression models with hyperparameter tuning
- Performs participant-level cross-validation
- Calculates feature importance
- Evaluates performance across demographic subgroups
- Computes Pearson correlations with confidence intervals
- Generates prediction results tables (Table 4)

**Output:**
- Trained prediction models
- Feature importance rankings
- Performance metrics (R², MAE, Pearson r)
- Results tables for publication

**Key Analyses:**
- Baseline model comparisons
- Feature ablation studies
- Subgroup performance evaluation
- Statistical significance testing

---

#### 6. Topic Messages Analysis.ipynb
**Stage**: Exploratory Analysis (Optional)

**What it does:**
- Topic modeling and message clustering
- Thematic analysis of conversation content
- Topic distribution visualization

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
