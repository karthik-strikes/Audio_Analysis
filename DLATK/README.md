# Klaatch_Dlatk.ipynb

A comprehensive Jupyter Notebook for performing linguistic and textual analysis using the DLATK (Differential Language Analysis ToolKit). This notebook demonstrates various stages of text processing, feature extraction, and correlation analysis on textual datasets.

## Table of Contents

- [Overview](#overview)
- [Key Functionality](#key-functionality)
  - [N-gram Feature Extraction](#n-gram-feature-extraction)
  - [LIWC Feature Extraction](#liwc-feature-extraction)
  - [LDA Topic Modeling](#lda-topic-modeling)
  - [Correlation Analysis and Word Clouds](#correlation-analysis-and-word-clouds)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Outputs](#outputs)
- [File Structure](#file-structure)

## Overview

The notebook utilizes the `dlatkInterface.py` script from the DLATK library to interact with MySQL databases (`Audio_features` and `alzdem_2025`) and perform various natural language processing tasks. It processes textual data (identified by `message_id`) to extract meaningful linguistic features and correlate them with outcome variables such as:

- `CEL_Total`
- `CELVAL1`
- `CELVAL2`
- `CELVAL3`
- `id`

The analysis pipeline generates CSV files containing correlation matrices and word cloud images for visualizing significant linguistic patterns.

## Key Functionality

### N-gram Feature Extraction

#### 1-gram Extraction

Extracts unigram (single word) features from the `merged_data` corpus:

```bash
!python /home/karthik9/TheDlatk/dlatk/dlatkInterface.py -d Audio_features -t merged_data -g message_id --add_ngrams -n 1
```

#### 1, 2, and 3-gram Extraction with Combination

Extracts and combines unigrams, bigrams, and trigrams into a single feature table (`feat$1to3gram$merged_data$message_id`):

```bash
!python /home/karthik9/TheDlatk/dlatk/dlatkInterface.py -d Audio_features -t merged_data -g message_id --add_ngrams -n 1 2 3 --combine_feat_tables 1to3gram
```

#### N-gram Feature Filtering

Applies occurrence and collocation filters to refine the N-gram feature set:

```bash
# With PMI threshold 6.0
!python /home/karthik9/TheDlatk/dlatk/dlatkInterface.py -d Audio_features -t merged_data -c message_id \
-f 'feat$1to3gram$merged_data$message_id' --feat_occ_filter --feat_colloc_filter --set_p_occ 0.05 --set_pmi_threshold 6.0

# With PMI threshold 3.0
!python /home/karthik9/TheDlatk/dlatk/dlatkInterface.py -d Audio_features -t merged_data -c message_id \
-f 'feat$1to3gram$merged_data$message_id' --feat_occ_filter --feat_colloc_filter --set_p_occ 0.05 --set_pmi_threshold 3.0
```

### LIWC Feature Extraction

#### LIWC2022 Lexicon Application

Applies the LIWC2022 lexicon to extract psychological and linguistic categories:

```bash
! python /home/karthik9/TheDlatk/dlatk/dlatkInterface.py -d Audio_features -t merged_data -c message_id --add_lex_table -l LIWC2022 --liwc_normalization
```

#### LIWC Correlation Analysis

Correlates LIWC features with outcome variables and generates word clouds:

```bash
!python /home/karthik9/TheDlatk/dlatk/dlatkInterface.py -d Audio_features -t merged_data -c message_id \
-f 'feat$cat_LIWC2015_lw$merged_data$message_id$1gra' \
    --outcome_table merged_data  --group_freq_thresh 5 \
    --outcomes CEL_Total CELVAL1 CELVAL2 CELVAL3 --output_name Klaatch_LIWC \
    --tagcloud --make_wordclouds --csv
```

### LDA Topic Modeling

The notebook performs LDA (Latent Dirichlet Allocation) topic modeling at different granularities:

#### 50 Topics (Overall)

Estimates 50 LDA topics from 1-gram features:

```bash
!python /home/karthik9/TheDlatk/dlatk/dlatkInterface.py -d Audio_features -t merged_data -g message_id  \
    -f 'feat$1gram$merged_data$message_id' \
    --estimate_lda_topics \
    --mallet_path /home/karthik9/mallet-2.0.8RC3/bin/mallet \
    --lda_lexicon_name lone_lda_50 \
    --num_lda_threads 20 \
    --save_lda_files /home/karthik9/lone/lone_50 \
    --num_topics 50
```

Extract features using the generated lexicon:

```bash
!python /home/karthik9/TheDlatk/dlatk/dlatkInterface.py -d Audio_features -t merged_data -c message_id  \
    --add_lex_table -l lone_lda_50_cp --weighted_lexicon
```

#### 100 Topics (Sentence-level LDA)

Estimates 100 LDA topics from sentence-tokenized data:

```bash
!python /home/karthik9/TheDlatk/dlatk/dlatkInterface.py -d Audio_features -t merged_data_senten -g message_id  \
    -f 'feat$1gram$merged_data_senten$message_id' \
    --estimate_lda_topics \
    --mallet_path /home/karthik9/mallet-2.0.8RC3/bin/mallet \
    --lda_lexicon_name klaatch_senten2_lda \
    --num_lda_threads 20 \
    --save_lda_files /home/karthik9/meta \
    --num_topics 100
```

Extract features:

```bash
!python /home/karthik9/TheDlatk/dlatk/dlatkInterface.py -d Audio_features -t merged_data -c message_id  \
    --add_lex_table -l klaatch_senten2_lda_cp --weighted_lexicon
```

#### 100 Topics (Normal LDA)

Estimates 100 LDA topics from 1-gram features:

```bash
!python /home/karthik9/TheDlatk/dlatk/dlatkInterface.py -d Audio_features -t merged_data -g message_id  \
    -f 'feat$1gram$merged_data$message_id' \
    --estimate_lda_topics \
    --mallet_path /home/karthik9/mallet-2.0.8RC3/bin/mallet \
    --lda_lexicon_name klattch_normal_lda \
    --num_lda_threads 20 \
    --save_lda_files /home/karthik9/lone/lone_100 \
    --num_topics 100
```

Extract features:

```bash
!python /home/karthik9/TheDlatk/dlatk/dlatkInterface.py -d Audio_features -t merged_data -c message_id  \
    --add_lex_table -l klattch_normal_lda_cp --weighted_lexicon
```

### Correlation Analysis and Word Clouds

#### Overall LDA Topic Correlation

```bash
!python /home/karthik9/TheDlatk/dlatk/dlatkInterface.py -d Audio_features -t merged_data -c message_id \
-f 'feat$cat_klattch_normal_lda_cp_w$merged_data$message_id$1gra' \
    --outcome_table  merged_data  --group_freq_thresh 5 \
    --outcomes CEL_Total  --output_name lda_overall \
    --topic_tagcloud --make_topic_wordcloud --topic_lexicon klattch_normal_lda_freq_t50ll \
    --tagcloud_colorscheme bluered --csv
```

#### Stratified LDA Topic Correlation

Analysis by demographic groups (gender, race):

```bash
# Female stratified analysis
!python /home/karthik9/TheDlatk/dlatk/dlatkInterface.py -d Audio_features -t merged_data -c message_id \
-f 'feat$cat_klaatch_senten2_lda_cp_w$merged_data$message_id$1gra' \
    --outcome_table stratified_female  --group_freq_thresh 5 \
    --outcomes CEL_Total  --output_name final_lda_klaatch_female \
    --topic_tagcloud --make_topic_wordcloud --topic_lexicon klaatch_senten2_lda_freq_t50ll \
    --tagcloud_colorscheme bluered --csv
```

## Requirements

### System Dependencies

- **Python 3.x**: The notebook uses Python 3 compatibility
- **MySQL Server**: Running instance with appropriate databases
- **Java**: Required for Mallet topic modeling toolkit

### Python Libraries

- **DLATK Library**: Main analysis toolkit
- **Pandas**: Data manipulation and analysis
- **Scipy**: Scientific computing and statistical analysis
- **Statsmodels**: Statistical modeling and econometrics
- **NumPy**: Numerical computing support

### External Tools

- **Mallet**: Topic modeling toolkit (path: `/home/karthik9/mallet-2.0.8RC3/bin/mallet`)
- **LIWC2022**: Linguistic Inquiry and Word Count lexicon

### Database Structure

Required MySQL databases and tables:

- **Audio_features** database containing:
  - `merged_data` table
  - `merged_data_senten` table (sentence-tokenized)
  - `stratified_male` table
  - `stratified_female` table
  - `stratified_black` table
  - `stratified_white` table
  - `prec_reddit` table
- **alzdem_2025** database (for some operations)

## Installation

1. **Install DLATK**:

   ```bash
   git clone https://github.com/dlatk/dlatk.git
   cd dlatk
   pip install -e .
   ```

2. **Install Mallet**:

   ```bash
   wget http://mallet.cs.umass.edu/dist/mallet-2.0.8RC3.tar.gz
   tar -xzf mallet-2.0.8RC3.tar.gz
   ```

3. **Install Python dependencies**:

   ```bash
   pip install pandas scipy statsmodels numpy mysql-connector-python
   ```

4. **Setup MySQL databases** with required tables and populate with your text data

5. **Configure LIWC2022 lexicon** in your DLATK installation

## Usage

### Environment Setup

1. Ensure DLATK and dependencies are correctly installed and accessible
2. Verify MySQL server is running with populated databases
3. Update file paths in notebook cells if referencing:
   - `Klatch_librosa.predicted_data.csv`
   - `LIWC_50topics_etc_goo_health.pickle`

### Execution

1. **Run cells sequentially** - each cell performs a specific analysis step
2. **Monitor progress** through console outputs from DLATK commands
3. **Verify outputs** are generated in specified directories

### Customization

- Modify database names (`-d` parameter) to match your setup
- Adjust table names (`-t` parameter) for your data structure
- Change outcome variables (`--outcomes` parameter) as needed
- Update file paths for your system configuration

## Outputs

### DLATK Command Line Output

- Detailed logs from `dlatkInterface.py` commands
- Database operation progress
- Feature extraction statistics
- Runtime information and performance metrics

### CSV Files

Generated correlation results:

- **Klaatch_LIWC.csv**: LIWC features vs outcome variables
- **Klaatch_ngram.csv**: N-gram features vs outcome variables
- **final_lda_klaatch.csv**: Overall LDA topic correlations
- **final_lda_klaatch_female.csv**: Female-stratified LDA correlations
- **final_lda_klaatch_male.csv**: Male-stratified LDA correlations
- **final_lda_klaatch_black.csv**: Black-stratified LDA correlations
- **final_lda_klaatch_white.csv**: White-stratified LDA correlations
- **prec_reddit_50_LDA.csv**: Reddit corpus 50-topic correlations

### Word Cloud Images (.png)

Visual representations in subdirectories:

- `Klaatch_LIWC_tagcloud_wordclouds/`: LIWC feature visualizations
- `final_lda_klaatch_topic_tagcloud_wordclouds/`: LDA topic visualizations
- Images categorized by outcome variable and correlation direction (positive/negative)

### LDA Model Files

Saved in specified directories:

- `/home/karthik9/lone/lone_50/`: 50-topic model files
- `/home/karthik9/meta/`: Sentence-level 100-topic model files
- `/home/karthik9/lone/lone_100/`: Normal 100-topic model files

Model files include:

- `lda.topicGivenWord.csv`: Topic-word distributions
- `lda.loglik.csv`: Log-likelihood values
- `lda.wordGivenTopic.csv`: Word-topic distributions
- `lda.freq.threshed50.loglik.csv`: Thresholded frequency files
- Mallet state files for model persistence

## File Structure

```
project_directory/
├── Klaatch_Dlatk.ipynb                    # Main analysis notebook
├── Klaatch_LIWC.csv                       # LIWC correlation results
├── Klaatch_ngram.csv                      # N-gram correlation results
├── final_lda_klaatch*.csv                 # LDA correlation results
├── prec_reddit_50_LDA.csv                 # Reddit LDA results
├── Klaatch_LIWC_tagcloud_wordclouds/      # LIWC word clouds
├── final_lda_klaatch_topic_tagcloud_wordclouds/  # LDA word clouds
└── model_files/
    ├── lone_50/                           # 50-topic model files
    ├── meta/                              # Sentence-level model files
    └── lone_100/                          # 100-topic model files
```

---

**Note**: This notebook requires significant computational resources and database setup. Ensure adequate system specifications and properly configured MySQL databases before execution. Update all file paths and database configurations to match your local environment.
