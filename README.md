# IEEE-CS-Advanced-AI-25-First-Project

---

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
