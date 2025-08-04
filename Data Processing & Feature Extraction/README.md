# Klaatch Data Processing and Feature Extraction

This Jupyter Notebook documents the process of preparing and enriching the Klaatch dataset for machine learning tasks. It involves several key steps: data loading, cleaning, merging with demographic information, and extracting various audio features using different libraries and models.

## Setup

The notebook requires the following Python libraries. You can install them by uncommenting and running the first cell:

- `torch`, `torchvision`, `torchaudio`
- `transformers`
- `librosa`
- `soundfile`
- `numpy`
- `openpyxl`
- `scikit-learn`
- `pandas`
- `pydub`
- `opensmile`
- `mysql-connector-python`
- `matplotlib`

## Workflow

The notebook follows a structured process to prepare the data:

1.  **Library and Database Connection:**

    - All necessary libraries are imported at the beginning.
    - A function `connect_to_db()` is defined to establish a connection to a MySQL database named `Audio_features`.

2.  **Data Loading and Merging:**

    - The script loads three primary data files:
      - `Klaatch Sanitized TextCEL 2021 to 2023 (1).xlsx`: Used to map old user IDs to new user IDs.
      - `Demographics_Klaatch Sanitized TextCEL 2021 to 2023.xlsx`: Contains demographic information for users.
      - `Klaatch_transcripts.csv`: Contains call transcripts and associated metadata.
    - It performs a series of data cleaning and merging operations:
      - Removes rows with missing `New ID` from the ID mapping file.
      - Cleans and processes the demographics data, ensuring `Gender` and `Race` are not null.
      - Merges the `transcripts_df` with the `demographics_df` first by `New ID` and then by `Old ID` for unmatched entries to create a comprehensive `metadata_df`.
    - It processes audio file names from a specified directory (`/sandata/karthik9/nwa`) to standardize them and create an `df_audio` DataFrame.
    - Finally, it merges the `metadata_df` with `df_audio` on the `Filename` column to create a master `merged_df` containing all relevant information.

3.  **Text Preprocessing:**

    - A dedicated cell handles text cleaning. It removes specific stopwords, speaker labels (`lisa`, `speaker1`, `speaker2`), conversational fillers (`mm hmm`), timestamps, and punctuation from the `Text` column of the `merged_df`.

4.  **Feature Extraction and Storage:**
    - The notebook extracts several types of audio features and stores them in the MySQL database.
    - **Whisper Features:** It uses the `openai/whisper-base` model to extract and save audio features as NumPy files (`.npy`). The mean and median of these features are then calculated and stored in separate database tables: `feat$whisper_mean_n$merged_data$message_id` and `feat$whisper_median_n$merged_data$message_id`.
    - **openSMILE Features:** It uses the `eGeMAPSv02` feature set from the `opensmile` library to extract a comprehensive set of audio features. These features are stored as a JSON object within the `new_audio_features` table in the database.
    - **Librosa Features:** It extracts traditional audio features such as MFCCs, chroma, spectral contrast, and tonnetz using the `librosa` library. The mean of these features is calculated and stored in individual columns within the `new_librosa_features` table.
    - **TRILL Features:** The notebook loads pre-computed TRILL embeddings from `.npy` files, restructures them, and inserts them into the `feat$trill_mstd$merged_data$message_id` table in the database.

## Database Schema

The notebook interacts with a MySQL database. The following tables are used:

- `merged_data`: Stores combined metadata, including file names, IDs, transcripts, and demographic information.
- `feat$whisper_mean_n$merged_data$message_id`: Stores the mean values of Whisper features.
- `feat$whisper_median_n$merged_data$message_id`: Stores the median values of Whisper features.
- `new_audio_features`: Stores openSMILE features as a JSON column.
- `new_librosa_features`: Stores Librosa features in individual columns.
- `feat$trill_mstd$merged_data$message_id`: Stores TRILL embeddings.

## Key Functions

- `connect_to_db()`: Connects to the local MySQL database.
- `insert_feature_table_into_db()`: A generic function to insert a pandas DataFrame into a specified database table.
- `process_and_store_whisper_features()`: Extracts Whisper features, saves them as `.npy` files, and prepares mean/median DataFrames for database insertion.
- `compute_mean_median()`: Helper function for processing Whisper `.npy` files.
- `insert_features_into_db_opensmile()`: Extracts openSMILE features and stores them in a JSON format in the database.
- `retrieve_features_from_opensmile_db()`: Retrieves openSMILE data from the database and converts the JSON back into a DataFrame.
- `process_audio_files()`: Extracts Librosa features and inserts them into a database table.
