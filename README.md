# Multimodal Loneliness Prediction Pipeline

A complete end-to-end pipeline for predicting emotional loneliness from conversational audio and text data. This repository contains the full research workflow from our study analyzing the Klaatch dataset (2021-2023), combining acoustic features, linguistic patterns, and demographic factors to predict UCLA Loneliness Scale scores.

**Key Contribution:** Fair, multimodal prediction using propensity score matching for demographic balance and participant-level cross-validation to prevent data leakage.

---

## Overview

Loneliness is a significant public health concern, particularly among older adults. This pipeline addresses the challenge of predicting emotional loneliness using multimodal features extracted from conversational audio recordings.

### What This Pipeline Does

1. **Extracts multimodal features** from audio conversations:
   - **Audio**: Whisper embeddings, OpenSmile acoustics, Librosa features, Trill embeddings
   - **Text**: LIWC2022 categories, LDA topics, n-grams (via DLATK)
   - **Demographics**: Age, gender, race, CEL scores

2. **Ensures fair analysis** across demographic groups:
   - Propensity score matching for balanced subgroups
   - Participant-level cross-validation (no data leakage)
   - Subgroup performance evaluation

3. **Generates publication-ready results**:
   - Feature correlation tables with Bonferroni correction
   - ExtraTrees regression predictions
   - Performance metrics by demographic subgroups

### Why This Matters

- **Reproducible Research**: Complete pipeline from raw data to results
- **Fair ML**: Explicit demographic balancing and fairness evaluation
- **Multimodal**: Combines complementary audio and text features
- **Methodologically Sound**: Participant-level CV prevents inflated performance estimates

---

## Key Capabilities

### Comprehensive Feature Extraction
- **4 audio feature types**: Whisper (speech embeddings), OpenSmile (88 acoustic features), Librosa (38 audio features), Trill (embeddings)
- **3 text feature types**: LIWC2022 (psychological categories), LDA topics, n-grams with PMI filtering
- Automated extraction with GPU acceleration support

### Fair Demographic Analysis
- **Propensity score matching**: Creates balanced demographic subgroups using logistic regression
- **Fairness evaluation**: Tests model performance across gender and racial groups
- Ensures equitable predictions across populations

### Rigorous Methodology
- **Participant-level cross-validation**: Splits by person, not message (prevents data leakage)
- **Statistical rigor**: Bonferroni correction for multiple comparisons
- **Complete documentation**: Every analysis step documented in Jupyter notebooks

---

## Quick Start

### Prerequisites

- Python 3.9+
- MySQL database
- CUDA-capable GPU (recommended)
- [DLATK](https://dlatk.github.io/dlatk/install.html) (for linguistic features)

### Installation

```bash
# 1. Clone and navigate to repository
cd Audio_Analysis

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install Python dependencies
pip install -r requirements.txt

# 4. Install DLATK separately
# Follow: https://dlatk.github.io/dlatk/install.html

# 5. Configure environment
cp .env.example .env
nano .env  # Edit paths and database credentials

# 6. Verify installation
python -c "from config import Config; print('✓ Configuration loaded')"
python -c "import torch, librosa, opensmile; print('✓ All modules available')"
```

### Run the Complete Pipeline

```bash
# Start Jupyter and run notebooks 1-5 in sequence
jupyter notebook notebooks/

# Notebooks to run:
# 1. Audio Feature Extraction Pipeline.ipynb
# 2. Propensity Score Matching Analysis.ipynb
# 3. Data Stratification for Cross-Validation.ipynb
# 4. Features Analysis.ipynb
# 5. Predicting Loneliness from Multimodal Features.ipynb
```

Each notebook is self-contained with detailed explanations. See [Analysis Workflow](#analysis-workflow) for details.

---

## Analysis Workflow

The complete research pipeline consists of 5 sequential stages, implemented as Jupyter notebooks.

**Why Notebooks?**
- **Reproducibility**: Each notebook is self-contained with detailed explanations, making it easy to reproduce results
- **Transparency**: All analysis steps are visible and documented inline with code
- **Exploratory Analysis**: Notebooks allow for iterative data exploration and visualization
- **Research Standard**: Common format for sharing computational research methods
- **DLATK Integration**: DLATK commands are best run interactively in notebook cells

The notebooks can be run independently or in sequence:

### 1. Audio Feature Extraction
**Notebook:** `Audio Feature Extraction Pipeline.ipynb`

- Loads and merges data from multiple sources (demographics, transcripts, audio files)
- Extracts acoustic features:
  - **Whisper**: Speech embeddings using OpenAI's model
  - **OpenSmile**: 88 acoustic features (eGeMAPSv02)
  - **Librosa**: 38 audio features (MFCCs, Chroma, Spectral Contrast, Tonnetz)
  - **Trill**: Pre-computed audio embeddings
- Stores features in MySQL database

**Output:** Database tables with audio features for all participants

---

### 2. Propensity Score Matching
**Notebook:** `Propensity Score Matching Analysis.ipynb`

- Calculates propensity scores using logistic regression
- Performs 1:1 nearest neighbor matching
- Creates balanced demographic datasets:
  - `stratified_male` / `stratified_female`
  - `stratified_black` / `stratified_white`
- Evaluates model fairness across subgroups

**Output:** Balanced demographic subgroups for fair model evaluation

**Key Functions:**
- `calculate_propensity_scores()` - Propensity score calculation
- `perform_propensity_matching()` - Matching algorithm
- `evaluate_subgroup_performance()` - Fairness evaluation

---

### 3. Data Stratification & Cross-Validation
**Notebook:** `Data Stratification for Cross-Validation.ipynb`

- Creates participant-level train/test splits (NOT message-level)
- Prevents data leakage: all messages from one person stay in same fold
- Generates stratified folds balancing demographic groups
- Verifies balance across folds

**Output:** Cross-validation fold assignments for proper evaluation

**Critical:** Splits by `klaatch_id` (participant ID), not `message_id`

---

### 4. Linguistic Features & Statistical Analysis
**Notebook:** `Features Analysis.ipynb`

- Extracts linguistic features using [DLATK](https://dlatk.github.io/dlatk/):
  - **LIWC2022**: Psychological language categories
  - **LDA Topics**: Topic modeling on conversation text
  - **N-grams**: 1-3 gram features with PMI ≥ 6.0
- Runs analysis on:
  - Total dataset (all participants)
  - Stratified subgroups (male, female, black, white)
- Computes correlations with Bonferroni correction
- Generates correlation tables (Tables 2, 3, S1-S9)

**Output:** Linguistic feature tables and statistical correlation results

**Note:** This notebook contains DLATK command-line calls. DLATK is run via shell commands within notebook cells:
```bash
dlatkInterface.py -d audio_analysis -t merged_data -c message_id \
    --add_liwc --outcome_table stratified_female --outcomes CEL_Total
```
See [DLATK documentation](https://dlatk.github.io/dlatk/) for command syntax.

---

### 5. Prediction Models
**Notebook:** `Predicting Loneliness from Multimodal Features.ipynb`

- Combines all features (audio + text + demographics)
- Creates feature combinations:
  - **Combined (Text)**: LIWC + LDA + N-grams
  - **Combined (Audio)**: Whisper + OpenSmile + Librosa + Trill
  - **Multimodal**: Text + Audio combined
- Trains ExtraTrees regression with hyperparameter tuning
- Performs participant-level cross-validation
- Evaluates performance by demographic subgroups
- Computes Pearson correlations with confidence intervals

**Output:** Prediction results tables (Table 4), feature importance, performance metrics

---

## Documentation

### Configuration

Edit `.env` to configure paths and database credentials:

```bash
# Database Configuration
DB_HOST=localhost
DB_DATABASE=audio_analysis
DB_USERNAME=your_username
DB_PASSWORD=your_password

# Data Paths
AUDIO_ROOT=/path/to/audio/files
WHISPER_FEATURES_DIR=/path/to/whisper/output
TRILL_FEATURES_DIR=/path/to/trill/embeddings

# Input Files
ID_MAPPING_FILE=Klaatch Sanitized TextCEL 2021 to 2023 (1).xlsx
DEMOGRAPHICS_FILE=Demographics_Klaatch Sanitized TextCEL 2021 to 2023.xlsx
TRANSCRIPTS_FILE=Klaatch_transcripts.csv

# Processing Options
TARGET_SAMPLE_RATE=16000
WHISPER_MODEL=openai/whisper-base
```

### Python Dependencies

Core packages installed via `requirements.txt`:
- **Core**: numpy, pandas, torch, transformers
- **Audio**: librosa, soundfile, pydub, opensmile
- **ML**: scikit-learn
- **Database**: mysql-connector-python
- **Config**: python-dotenv

**Linguistic Features**: [DLATK](https://dlatk.github.io/dlatk/) must be installed separately for notebook 4.

### Notebook Reference

| Notebook | Purpose | Key Outputs |
|----------|---------|-------------|
| 1. Audio Feature Extraction | Extract acoustic features | Database tables with Whisper, OpenSmile, Librosa, Trill features |
| 2. Propensity Score Matching | Create balanced subgroups | `stratified_male`, `stratified_female`, `stratified_black`, `stratified_white` |
| 3. Data Stratification | Setup cross-validation | Participant-level fold assignments |
| 4. Features Analysis | Extract linguistic features | LIWC, LDA, n-gram tables; correlation tables (Tables 2, 3, S1-S9) |
| 5. Prediction Models | Train and evaluate models | Prediction results (Table 4), feature importance |
| 6. Topic Messages Analysis | Exploratory topic modeling | Topic distributions (optional) |

**Note:** Some code may be duplicated across notebooks intentionally for self-contained reproducibility.

### Python CLI Tools (Optional)

For developers who only need audio feature extraction without the full analysis:

```bash
# Load and preprocess data only
python run_pipeline.py --load-only

# Extract specific features
python run_pipeline.py --extract-whisper
python run_pipeline.py --extract-opensmile
python run_pipeline.py --extract-librosa
python run_pipeline.py --load-trill

# Insert features to database
python run_pipeline.py --insert-db

# Extract all audio features
python run_pipeline.py --all
```

**Note:** Python scripts only handle audio features. For linguistic features (LIWC, LDA, n-grams), use notebook 4 with DLATK.

---

## Project Structure

```
Audio_Analysis/
├── 📋 Configuration
│   ├── .env.example              # Environment template
│   ├── requirements.txt          # Python dependencies
│   └── config/
│       └── config.py             # Configuration management
│
├── 🔧 Source Code (src/)
│   ├── data_loader.py            # Data loading and merging
│   ├── text_processor.py         # Text preprocessing
│   ├── audio_utils.py            # Audio processing utilities
│   ├── database.py               # Database operations
│   └── extractors/
│       ├── whisper_extractor.py  # Whisper features
│       ├── opensmile_extractor.py# OpenSmile features
│       ├── librosa_extractor.py  # Librosa features
│       └── trill_extractor.py    # Trill features
│
├── 📓 Jupyter Notebooks (notebooks/)
│   ├── 1. Audio Feature Extraction Pipeline.ipynb
│   ├── 2. Propensity Score Matching Analysis.ipynb
│   ├── 3. Data Stratification for Cross-Validation.ipynb
│   ├── 4. Features Analysis.ipynb (DLATK commands)
│   ├── 5. Predicting Loneliness from Multimodal Features.ipynb
│   └── 6. Topic Messages Analysis.ipynb
│
├── 🚀 Executable Scripts
│   ├── run_pipeline.py           # CLI for audio extraction
│   └── setup.py                  # Setup script
│
├── 📜 Standalone Scripts (scripts/)
│   ├── audio_pattern_silencer.py
│   ├── audio_sentence_splitter.py
│   └── whisper_feature_extractor.py
│
└── 📚 README.md                  # This file
```

**Code Statistics:**
- 1,200+ lines of organized Python code
- 13 Python modules in `src/`
- 6 comprehensive Jupyter notebooks
- 3 standalone utility scripts

---

## Troubleshooting

### Module Not Found Errors

```bash
# Reinstall dependencies
pip install -r requirements.txt

# Verify virtual environment is activated
which python  # Should point to venv/bin/python
```

### Database Connection Issues

```bash
# Test database connection
python -c "from src.database import test_connection; test_connection()"

# Verify MySQL is running
sudo systemctl status mysql

# Check .env credentials
cat .env | grep DB_
```

### CUDA/GPU Issues

```bash
# Check CUDA availability
python -c "import torch; print('CUDA:', torch.cuda.is_available())"

# Use CPU if GPU unavailable (slower)
# Whisper will automatically fall back to CPU
```

### DLATK Command Errors

```bash
# Verify DLATK installation
which dlatkInterface.py

# Check database permissions for DLATK
# DLATK requires specific MySQL permissions

# See: https://dlatk.github.io/dlatk/install.html#database-setup
```

### Audio Files Not Found

```bash
# Verify AUDIO_ROOT path in .env
ls $AUDIO_ROOT/*.mp3

# Check file naming convention: {ID}_{Date}.mp3
# Example: 559_2021-01-22.mp3
```

### Memory Issues

- Use smaller Whisper model: `WHISPER_MODEL=openai/whisper-tiny` in `.env`
- Process files in smaller batches
- Close other applications to free RAM
- Use a machine with more memory for large datasets

---

## Advanced Usage

### Custom Feature Extractors

Create new feature extractors by extending base classes:

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

Process large datasets in batches:

```python
from src.data_loader import load_and_merge_all_data
from src.extractors import WhisperExtractor

df = load_and_merge_all_data()
extractor = WhisperExtractor()

batch_size = 100
for i in range(0, len(df), batch_size):
    batch = df.iloc[i:i+batch_size]
    for idx, row in batch.iterrows():
        if pd.notna(row['Filepath']):
            extractor.process_file(row['Filepath'], row['Filename'])
```

### Database Schema

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
- `feat$whisper_mean_n$merged_data$message_id` - Whisper mean features
- `feat$whisper_median_n$merged_data$message_id` - Whisper median features
- `feat$opensmile_n$merged_data$message_id` - OpenSmile features
- `feat$librosa_n$merged_data$message_id` - Librosa features
- `feat$trill_mstd$merged_data$message_id` - Trill features
- `feat$cat_LIWC2022_lw$merged_data$message_id$1gra` - LIWC features
- `feat$cat_klaatch_senten2_lda_cp_w$merged_data$message_id$1gra` - LDA topics
- `feat$1to3gram$merged_data$message_id$0_05$pmi6_0` - N-gram features

---

## Citation

If you use this pipeline in your research, please cite:

```bibtex
@article{yourpaper2024,
  title={Multimodal Prediction of Emotional Loneliness from Conversational Audio},
  author={Your Name et al.},
  journal={Journal Name},
  year={2024}
}
```

---

## License

[Add license information]

---

## Contributing

This pipeline was developed as part of a research study. For questions, issues, or contributions:

- **Issues**: Open an issue on GitHub
- **Questions**: [Add contact email]
- **Pull Requests**: Welcome for bug fixes and improvements

---

## Acknowledgments

This pipeline uses the following open-source projects:

- **[OpenAI Whisper](https://github.com/openai/whisper)** - Speech recognition and embeddings
- **[OpenSMILE](https://www.audeering.com/opensmile/)** - Audio feature extraction
- **[Librosa](https://librosa.org/)** - Audio analysis library
- **[Google TRILL](https://github.com/google-research/google-research/tree/master/non_semantic_speech_benchmark)** - Audio embeddings
- **[DLATK](https://dlatk.github.io/)** - Differential Language Analysis ToolKit
- **[Hugging Face Transformers](https://huggingface.co/transformers/)** - Model infrastructure

**Data Source:** Klaatch dataset (2021-2023)

---

## Version History

### Version 1.0.0 (2024-01-29)

**Initial Release** - Complete multimodal loneliness prediction pipeline

**Features:**
- 4 audio feature extractors (Whisper, OpenSmile, Librosa, Trill)
- 3 text feature types (LIWC, LDA, N-grams via DLATK)
- Propensity score matching for fair demographic analysis
- Participant-level cross-validation
- Complete Jupyter notebook workflow (5 notebooks)
- Python CLI tools for audio extraction
- MySQL database integration
- Comprehensive documentation

**Methodology:**
- Ensures fair ML through demographic balancing
- Prevents data leakage via participant-level CV
- Statistical rigor with Bonferroni correction
- Generates publication-ready results tables

---

**Status:** ✅ Production-Ready

**Repository:** https://github.com/karthik-strikes/Audio_Analysis

**Last Updated:** 2024-02-09
