### Stratification & Analysis

This project involves two Python scripts, `Klaatch_stratify.ipynb` and `Klaatch_topmsgs.ipynb`, designed to prepare and analyze audio-related message data from a MySQL database. The scripts perform data stratification for cross-validation and extract top messages associated with specific topics.

#### `Klaatch_stratify.ipynb`

This notebook handles data stratification to create balanced folds for cross-validation.

**Key functionalities:**

- **MySQL Connection:** Connects to the `Audio_features` MySQL database using a specified user and password.
- **Data Loading:** Loads data from the `stratified_female` MySQL table into a pandas DataFrame.
- **Balanced Fold Assignment:** It assigns fold numbers by first extracting a `klaatch_id` from the `message_id` column. The script then sorts these IDs by their record count in descending order and assigns each `klaatch_id` to the fold with the fewest records to ensure a balanced distribution of samples.
- **Validation:** A validation function checks that each `klaatch_id` is assigned to only a single fold.
- **SQL Table Update:** The script updates the original MySQL table with new `klaatch_id` and `fold` columns and their calculated values.

#### `Klaatch_topmsgs.ipynb`

This notebook focuses on identifying and extracting the top messages for different categories or topics.

**Key functionalities:**

- **Database Connection:** Connects to the `Audio_features` MySQL database using `sqlalchemy`.
- **Data Retrieval:** Reads data from two tables: `feat$cat_ZClaatch_100_cp_w$merged_data$message_id$1gra` and `merged_data`.
- **Data Processing:** The script cleans the data by removing rows with `_intercept` as a feature and merges the two tables on `group_id` and `message_id`. It then filters for messages with more than five words.
- **Top Messages Extraction:** The script groups the data by `category` and identifies the top 10 messages for each based on the `group_norm` value.
- **Top Words Merging:** It merges the top messages with a separate CSV file (`50TOPICS.csv`) to include the top 10 words for each topic.
- **Output:** The final result, which includes the `category`, `message_id`, `message`, and `top_10_words`, is saved to a CSV file named `z100.csv`.
- **Dlatk Integration:** The notebook includes a shell command for using the Dlatk interface to perform a regression analysis on the prepared data.
