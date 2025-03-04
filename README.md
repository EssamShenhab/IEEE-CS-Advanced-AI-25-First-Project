# IEEE-CS-Advanced-AI-25-First-Project

## 5.Feature selection (feature_selection(df)):

This function is used to identify the important features in DS, remove the irrelevant or redundant ones through column selection, and visualize correlations.

### Functionality:

- displays a correlation Heatmap: which helps identify relationships between numerical columns.
- Allows users to select columns to drop: The Dataset is refined by removing unwanted numerical features, then updated.
- Handles the non-existence of numerical data: Informs the user through a message

### Steps:

1. Extract numerical columns from dataset
2. displays a heatmap that shows correlations among features
3. A drop-down is provided to select the uneeded columns
4. Updates and displays the refined dataset

## 6-Encode data (encode_data()):

This function converts the selected categorical data into numerical variables so that ML models can process them, Either through Label- encoding or One-Hot encoding

### Functionality:

- Lists categorical columns: displays the number of unique values in each
- Provide encoding options:
    - Label Encoding: which converts categorical labels to numerical values
    - One-Hot Encoding: creates binary columns for each category in a selected column “avoid multicollinearity by using drop_first = True”
- After encoding it updates the Dataset

                  

## Steps:

1. Displays categorical columns in a dataset.
2. Display the count of unique values per column.
3. User selects columns to encode
4. Two encoding methods are available 
    1. Label Encoding
    2. One-Hot Encoding
5. After encoding, the dataset is updated and dispalyed
