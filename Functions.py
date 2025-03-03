import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging





def go_to_next():
    st.session_state.page = "next"


def input_data():
    """
    This function handles the dataset upload process in the Dataset Preprocessor application.
    It displays a file uploader for users to select a CSV or Excel file, reads the file into a DataFrame,
    and logs the successful upload.

    Parameters:
    None

    Returns:
    None
    """
    st.title('Dataset Preprocessor')
    st.text('Welcome to Dataset Preprocessor: Your Gateway to Efficient Data Understanding and Preprocessing!')

    st.header('Upload your Dataset')
    uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx"])

    if uploaded_file is not None:
        # Determine the file type
        file_type = uploaded_file.type

        # Read the file into a DataFrame
        if file_type == "text/csv":
            df = pd.read_csv(uploaded_file)
        elif file_type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
            df = pd.read_excel(uploaded_file)
        logging.info("\"%s\" uploaded successfully", uploaded_file)
        st.session_state.original_df = df.copy()
        st.session_state.df = df
        st.button('Continue', on_click=go_to_next)


def visualization():
    side_bar()
    st.header("Data Visualization")
    st.write("Select a column to visualize:")
    column = st.selectbox("Select a column", st.session_state.df.columns)
    if st.session_state.df[column].dtype == "numerical":
        st.write("Bar Chart")
        plt.figure(figsize=(10, 6))
        sns.countplot(x=column, data=st.session_state.df)
        st.pyplot()
    else:
        st.write("Histogram")
        plt.figure(figsize=(10, 6))
        sns.histplot(st.session_state.df[column], kde=True)
        st.pyplot()


def handle_missing_values():
    side_bar()
    st.header("Handle Missing Values")  

def remove_redundant_features(): 
    side_bar()   
    st.header("Remove Redundant Features")

def feature_selection():
    side_bar()
    st.header("Feature Selection")

def encode_data():
    side_bar()
    st.header("Encode Data")

def feature_scaling():
    side_bar()
    st.header("Feature Scaling")

def automatic_processing():
    side_bar()
    st.header("Automatic Processing")

def download_preprocessed_data():
    side_bar()
    st.header("Download Preprocessed Data")


# "Documentation

def documentation():
    """
    This function displays the documentation page of the Dataset Preprocessor application.
    It provides detailed descriptions of each functionality, including uploading a dataset,
    data visualization, handling missing values, removing redundant features, feature selection,
    encoding data, feature scaling, automatic processing, downloading preprocessed data,
    resetting the DataFrame, working with another dataset, and logging.

    Parameters:
    None

    Returns:
    None
    """
    side_bar()
    st.header("Documentation")
    st.subheader("Upload Your Dataset")
    st.write("You can upload your dataset in CSV or Excel format. The dataset will be read into a DataFrame. The DataFrame is a two-dimensional labeled data structure with columns of potentially different types. It is generally the most commonly used pandas object.")
    st.subheader("Data Visualization")
    st.write("This section provides a detailed statistical summary of the dataset or a specific column. For numerical columns, the description display it as a histogram or any appropriate display type. If this column is categories, display it as a bar chart or Pie chart.")
    st.subheader("Handle Null Values")
    st.write("This section provides various options to handle null values in the dataset. You can view the count of null values in each column, remove columns with null values, drop rows with null values, or fill null values with a specific value (like zero, mean, median, mode, forward fill, or back fill). For categorical columns, only forward fill and back fill are available.")
    st.subheader("Encode Data")
    st.write("This section allows you to encode categorical columns in the dataset using one-hot encoding. The selected column will be replaced with multiple columns (one for each unique value in the column), where each row has a 1 in the column corresponding to the value it had and 0 in all other new columns.")
    st.subheader("Feature Scaling")
    st.write("This section provides options to scale numerical columns in the dataset. You can normalize (MinMax Scaler) or standardize (Standard Scaler) the whole dataset or a specific column. Normalization scales the data between 0 and 1, while standardization scales data to have a mean of 0 and a standard deviation of 1.")
    st.subheader("Feature Selection")
    st.write("This section allows you to select features in the dataset. You can view a correlation matrix of the dataset and drop selected columns. The correlation matrix provides a visual representation of the linear relationships between variables.")
    st.subheader("Download the Dataset")
    st.write("This section allows you to download the preprocessed dataset in CSV or Excel format. You can also download the log file containing the actions performed on the dataset. This can be useful for auditing and debugging purposes.")
    st.subheader("Reset DataFrame")
    st.write("This section allows you to reset the DataFrame. This action reverses all the operations performed on the DataFrame, returning it to its original state when it was first uploaded.")
    st.subheader("Work with Another Dataset")
    st.write("This section allows you to choose to work with another dataset. All the work done on the current dataset will be lost. This is useful when you want to preprocess multiple datasets in one session. Be sure to download the preprocessed dataset before switching to another dataset.")
    st.subheader("Logging")
    st.write("All the operations performed on the dataset are logged. The log file can be downloaded along with the preprocessed dataset. The log file contains information such as the name of the operation, the time it was performed, and any additional details.")



def side_bar():
    if "confirm_reset" not in st.session_state:
        st.session_state.confirm_reset = False
    with st.sidebar:
        st.header('Dataset Preprocessor')
        if st.button('Visualization'):
            st.session_state.page = "Visualization"
        if st.button('Handle Missing Values'):
            st.session_state.page = "Handle Missing Values"
        if st.button('Remove Redundant Features'):
            st.session_state.page = "Remove Redundant Features"
        if st.button('Feature Selection'):
            st.session_state.page = "Feature Selection"
        if st.button('Encode Data'):
            st.session_state.page = "Encode Data"
        if st.button('Feature Scaling'):
            st.session_state.page = "Feature Scaling"
        if st.button('Automatic Processing'):
            st.session_state.page = "Automatic Processing"
        if st.button('Download Preprocessed Data'):
            st.session_state.page = "Download Preprocessed Data"
        if st.button('Documentation'):
<<<<<<< HEAD
            st.session_state.page = "Documentation"
=======
            st.session_state.page = "Documentation"
>>>>>>> Fixing df intialization
