# Jupyter Notebooks

This folder contains the original Jupyter notebooks that were used for audio analysis and feature extraction. These notebooks have been preserved for reference, exploration, and ad-hoc analysis.

## Available Notebooks

### 1. Audio Feature Extraction Pipeline.ipynb
**Main notebook** - Complete audio feature extraction pipeline

Contains:
- Data loading and merging from multiple sources
- Text preprocessing and cleaning
- Whisper feature extraction
- OpenSmile acoustic features
- Librosa audio features
- Trill embeddings processing
- Database storage operations
- Statistical analysis

**Note**: This notebook has been converted into the production-ready Python modules in the `src/` directory.

### 2. Features Analysis.ipynb
Feature exploration and analysis

Contains:
- Feature distribution analysis
- Feature correlation studies
- Feature importance evaluation
- Visualization of extracted features

### 3. Data Stratification for Cross-Validation.ipynb
Cross-validation setup and data stratification

Contains:
- Train/test split strategies
- Stratified sampling methods
- Data balance verification
- Cross-validation fold creation

### 4. Predicting Loneliness from Multimodal Features.ipynb
Machine learning models for loneliness prediction

Contains:
- Multimodal feature integration
- Model training and evaluation
- Baseline comparisons
- Performance metrics
- Feature ablation studies

### 5. Propensity Score Matching Analysis.ipynb
Statistical analysis using propensity score matching

Contains:
- Propensity score calculation
- Matching algorithms
- Balance diagnostics
- Treatment effect estimation

### 6. Topic Messages Analysis.ipynb
Topic modeling and message analysis

Contains:
- Text topic extraction
- Topic distribution analysis
- Message clustering
- Thematic analysis

## Usage

### Running Notebooks

1. **Install Jupyter**:
   ```bash
   pip install jupyter notebook
   # or
   pip install jupyterlab
   ```

2. **Start Jupyter**:
   ```bash
   cd /home/karthik9/Audio_Analysis
   jupyter notebook
   # or
   jupyter lab
   ```

3. **Open any notebook** from the `notebooks/` folder

### Dependencies

All notebooks require the dependencies listed in `../requirements.txt`:
```bash
pip install -r ../requirements.txt
```

### Environment Setup

Make sure you have:
- Created and configured your `.env` file (see `../.env.example`)
- Set up database connection (if using database features)
- Access to audio files and data directories