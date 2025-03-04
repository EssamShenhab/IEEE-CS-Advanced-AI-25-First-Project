# IEEE-CS-Advanced-AI-25-First-Project
---


## 1. Dataset Upload (input_data())

This function allows users to upload a dataset for preprocessing. It supports CSV and Excel files.

### Functionality:

- Users can upload a dataset through the file uploader.
- The app determines the file type and reads the data accordingly.
- The uploaded dataset is stored in session state for further processing.

### Steps:

1. Users upload a CSV or Excel file.
2. The app reads the file into a Pandas DataFrame.
3. The dataset is stored in session state.
4. A "Continue" button allows users to proceed to the next step.

## 2. Data Visualization (visualization())

This function enables users to visualize the dataset by selecting a column for visualization.

### Functionality:

- Users select a column from the dataset.
- The app determines the data type and chooses an appropriate visualization:
    - **Categorical columns**: Displayed using bar charts.
    - **Numerical columns**: Displayed using histograms.

### Steps:

1. Users select a column from the dataset.
2. If the column is categorical, a bar chart is generated.
3. If the column is numerical, a histogram with a KDE plot is generated.
4. The visualization is displayed using Matplotlib and Seaborn.

## 3. Handle Missing Values

This feature allows you to manage missing values in your dataset efficiently by providing options to:

- **Drop Columns**: Remove columns with a high percentage of missing values.
- **Drop Null Values**: Drop rows containing missing values for specific features.
- **Fill Null Values**: Fill missing values using various strategies, including:
  - Zero
  - Mean
  - Median
  - Mode
  - Forward Fill
  - Back Fill
- **Show Dataset**: View the updated dataset after handling nulls.

### How It Works

1. The app calculates and displays null information for each feature, showing:
   - Count of missing values
   - Percentage of missing values
   - Suggested action based on threshold values:
     - **Drop Feature**: When null percentage ≥ 80%
     - **Fill Values**: When 30% ≤ null percentage < 80%
     - **Drop Observations**: When null percentage < 30%
2. Users can interactively choose the preferred action using buttons and selection tools.
3. The dataset is updated in real-time and displayed after each operation.

## 4. Remove Redundant Features

This feature identifies and removes features with high redundancy, such as columns with:

- A single repeated value in the majority of rows.
- Values exceeding a specified redundancy threshold (default is 80%).

### How It Works

1. The app analyzes each feature to determine if a single value dominates the column.
2. Displays a list of redundant features with their redundancy percentage.
3. Allows users to select a redundant feature from a dropdown and remove it with a button click.
4. The updated dataset is displayed after the feature is removed.

## 5. Feature Selection (feature_selection(df))

This function is used to identify the important features in DS, remove the irrelevant or redundant ones through column selection, and visualize correlations.

### Functionality:

- Displays a correlation heatmap: Helps identify relationships between numerical columns.
- Allows users to select columns to drop: The dataset is refined by removing unwanted numerical features, then updated.
- Handles the non-existence of numerical data: Informs the user through a message.

### Steps:

1. Extract numerical columns from dataset.
2. Displays a heatmap that shows correlations among features.
3. A drop-down is provided to select the unneeded columns.
4. Updates and displays the refined dataset.

## 6. Encode Data (encode_data())

This function converts the selected categorical data into numerical variables so that ML models can process them, either through Label Encoding or One-Hot Encoding.

### Functionality:

- Lists categorical columns: Displays the number of unique values in each.
- Provides encoding options:
    - Label Encoding: Converts categorical labels to numerical values.
    - One-Hot Encoding: Creates binary columns for each category in a selected column (avoiding multicollinearity by using `drop_first = True`).
- After encoding, it updates the dataset.

### Steps:

1. Displays categorical columns in a dataset.
2. Displays the count of unique values per column.
3. User selects columns to encode.
4. Two encoding methods are available:
    1. Label Encoding
    2. One-Hot Encoding
5. After encoding, the dataset is updated and displayed.

## Technologies Used
- **Python** (Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn)
- **Streamlit** (for interactive UI)

---

## Contributors
- [Essam Shenhab](https://github.com/EssamShenhab)
- [Salma Swailem](https://github.com/Salma-Swailem)
- [Basmalla Abuhashim](https://github.com/BasmalaAbuhashim)
- [Mina Sameh](https://github.com/MINAsamehj)

---