# DLTK Prediction Experiments

This repository contains a Jupyter notebook (`Klaatch_predictions.ipynb`) that runs several experiments using the DLTK (Digital Language Toolkit) to predict the `CEL_Total` outcome from various audio and text features. The notebook also includes a section for combining the predictions from multiple models using an ensemble method.

## Code Description

The Jupyter notebook consists of several code cells, each performing a specific task:

1.  **Text Feature Models:**

    - **1-3 grams:** Trains an `ExtraTreesRegressor` model using 1- to 3-gram text features (`feat$1to3gram$merged_data$message_id$0_05$pmi3_0`) to predict `CEL_Total`.
    - **LIWC 2022:** Uses LIWC 2022 features (`feat$cat_LIWC2022_lw$merged_data$message_id$1gra`) with an `ExtraTreesRegressor` model to predict `CEL_Total`.
    - **Sentence LDA:** Employs Sentence LDA topic modeling features (`feat$cat_klaatch_senten2_lda_cp_w$merged_data$message_id$1gra`) with an `ExtraTreesRegressor` model to predict `CEL_Total`.
    - **Combined Text:** Combines the features from the three text models above to train a single `ExtraTreesRegressor` model.

2.  **Audio Feature Models:**

    - **Librosa:** Trains an `ExtraTreesRegressor` model using Librosa audio features (`feat$librosa_n$merged_data$message_id`) to predict `CEL_Total`.
    - **OpenSMILE:** Uses OpenSMILE audio features (`feat$opensmile_n$merged_data$message_id`) with an `ExtraTreesRegressor` model to predict `CEL_Total`.
    - **Whisper Mean:** Employs the mean of Whisper audio features (`feat$whisper_mean_n$merged_data$message_id`) with an `ExtraTreesRegressor` model to predict `CEL_Total`.
    - **Whisper Median:** Uses the median of Whisper audio features (`feat$whisper_median_n$merged_data$message_id`) with an `ExtraTreesRegressor` model to predict `CEL_Total`, as well as `CELVAL1`, `CELVAL2`, and `CELVAL3`.
    - **Trill Mean, Max, and Min:** Experiments are run using `trill_mean`, `trill_max`, and `trill_min` audio features to predict `CEL_Total`, `CELVAL1`, `CELVAL2`, and `CELVAL3`.

3.  **Ensemble Model:**
    - The notebook combines the individual predictions from the different models (referred to as "experts") into a single dataframe.
    - A linear regression model acts as a "gating network" to assign weights to each expert's prediction.
    - It then calculates a "final prediction" for each instance by taking a weighted average of the top-k expert predictions, where k=3.
    - The notebook calculates and prints the Pearson's r, R-squared, Mean Absolute Error (MAE), and p-values for the final ensemble prediction.

## DLTK Command Flag Explanations

The DLTK command used in the notebook, for example, `!python /home/karthik9/TheDlatk/dlatk/dlatkInterface.py -d Audio_features -t merged_data -c message_id -f 'feat$combined_text$merged_data$message_id' --outcome_table merged_data --group_freq_thresh 0 --outcomes CEL_Total --nfold_test_regression --model extratrees --fold_column fold`, uses the following flags:

- `-d Audio_features`: Specifies the database name to be used.
- `-t merged_data`: Indicates the corpus or table containing the data.
- `-c message_id`: Identifies the column that serves as the unique group identifier.
- `-f 'feat$combined_text$merged_data$message_id'`: Specifies the feature table to be used for the model.
- `--outcome_table merged_data`: Indicates the table where the outcome variable is stored.
- `--group_freq_thresh 0`: Sets a minimum frequency threshold for the groups, where a value of `0` means no groups are filtered out based on frequency.
- `--outcomes CEL_Total`: Specifies the outcome variable to be predicted.
- `--nfold_test_regression`: Instructs DLTK to perform n-fold cross-validation for a regression task.
- `--model extratrees`: Selects the machine learning model to be used, in this case, `ExtraTreesRegressor`.
- `--fold_column fold`: Specifies that the `fold` column in the data should be used for the cross-validation folds.

## Instructions to Reproduce

1.  **Prerequisites:**

    - Ensure you have the DLTK environment set up with the necessary dependencies, including Python 3, `scikit-learn`, `statsmodels`, `pandas`, and `mysql-connector`.
    - A MySQL database named `Audio_features` is required, with a table named `merged_data` and feature tables for each modality mentioned above. The notebook assumes a pre-existing database and tables.
    - The feature tables are named in the format `feat$<feature_name>$merged_data$message_id`. For example, `feat$librosa_n$merged_data$message_id`.

2.  **Running the Notebook:**
    - Open the `Klaatch_predictions.ipynb` file in a Jupyter environment.
    - Run the cells in sequential order. Each cell contains DLTK commands or Python code to execute the experiments and process the results.
    - The `!python` commands are DLTK command-line interface calls, so they should be executed within an environment where the DLTK is configured correctly.
    - The Python cells at the end of the notebook handle the combination of predictions and the evaluation of the ensemble model.

## Results Summary

The experiments evaluate the performance of various models in predicting `CEL_Total`. The best-performing model combinations are:

- **Combined Text Features:** The combination of LIWC, Sentence LDA, and 1- to 3-gram features achieved an overall $R^2$ of **0.0757** and a Pearson's $r$ of **0.2778**.
- **Ensemble Model:** The final ensemble model, which combines predictions from the individual models, achieved an overall $R^2$ of **0.0224** and a Pearson's $r$ of **0.2619**.

Below is a table summarizing the results from some of the individual and combined models:

| Model Features                                                                  | Overall R² | Overall Pearson's r |
| ------------------------------------------------------------------------------- | ---------- | ------------------- |
| librosa                                                                         | -0.0051    | 0.1523              |
| opensmile                                                                       | 0.0006     | 0.1796              |
| whisper_mean                                                                    | -0.0060    | 0.1144              |
| whisper_median                                                                  | -0.0196    | 0.0924              |
| librosa, opensmile, whisper_mean                                                | 0.0354     | 0.2137              |
| LIWC2022                                                                        | 0.0657     | 0.2615              |
| ldasenten                                                                       | 0.0218     | 0.1736              |
| 1to3grams                                                                       | 0.0608     | 0.2478              |
| LIWC2022, ldasenten, 1to3grams                                                  | 0.0757     | 0.2778              |
| All Features (librosa, opensmile, whisper_mean, LIWC2022, ldasenten, 1to3grams) | 0.1003     | 0.3222              |
| Ensemble (top-3 weighted average of individual models)                          | 0.0224     | 0.2619              |

_Note: The `trill` models were also evaluated for different outcomes (`CELVAL1`, `CELVAL2`, `CELVAL3`, `CEL_Total`) with varying results. The best-performing `trill` model for `CEL_Total` was `trill_mean`, with an $R^2$ of 0.0339 and a Pearson's $r$ of 0.1996._
