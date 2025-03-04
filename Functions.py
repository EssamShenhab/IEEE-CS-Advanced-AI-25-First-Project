import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging





def go_to_next():
    """
    Sets the current page in the session state to "next".

    Parameters:
    None

    Returns:
    None
    """
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
    
    # User selects a column
    column = st.selectbox("Select X-axis", st.session_state.df.columns)

    # Check if the column is numerical or categorical
    if pd.api.types.is_numeric_dtype(st.session_state.df[column]):
        st.write("### Histogram")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(st.session_state.df[column], kde=True, ax=ax)
        st.pyplot(fig)

    else:
        st.write("### Bar Chart")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.countplot(x=column, data=st.session_state.df, ax=ax)
        plt.xticks(rotation=45)
        st.pyplot(fig)

        st.write("### Pie Chart")
        fig, ax = plt.subplots(figsize=(6, 6))
        st.session_state.df[column].value_counts().plot.pie(autopct="%1.1f%%", ax=ax)
        st.pyplot(fig)


def handle_missing_values():
    side_bar()
    st.header("Handle Missing Values")

    def show_nulls_information(data, threshold_max=80, threshold_min=30):
        try:
            nulls_count_df = data.isnull().sum().reset_index().rename(columns={'index': 'Feature', 0: 'Count'})
            nulls_count_df = nulls_count_df[nulls_count_df['Count'] > 0].sort_values(by='Count', ascending=False)
            nulls_count_df['Percentage'] = (nulls_count_df['Count'] / data.shape[0]) * 100
            nulls_count_df['Category'] = 'Drop Observations'
            nulls_count_df.loc[nulls_count_df['Percentage'] >= threshold_max, 'Category'] = 'Drop Feature'
            nulls_count_df.loc[(nulls_count_df['Percentage'] >= threshold_min) & (nulls_count_df['Percentage'] < threshold_max), 'Category'] = 'Fill Values'

            st.write('Nulls Information:', nulls_count_df)
            return nulls_count_df
        except Exception as e:
            st.error(f"Error displaying nulls information: {e}")
            return pd.DataFrame()

    st.session_state.nulls_info_df = show_nulls_information(st.session_state.df)

    if 'selected_feature' not in st.session_state:
        st.session_state.selected_feature = None
    if 'show_drop_col_section' not in st.session_state:
        st.session_state.show_drop_col_section = False
    if 'show_fill_section' not in st.session_state:
        st.session_state.show_fill_section = False
    if 'show_drop_null_values' not in st.session_state:
        st.session_state.show_drop_null_values = False
    if 'show_dataset' not in st.session_state:
        st.session_state.show_dataset = False

    # Interactive Columns for Actions
    col1, col2, col3, col4 = st.columns(4)

    # Drop Columns Button
    with col1:
        if st.button('Drop Columns'):
            st.session_state.show_drop_col_section = True
            st.session_state.show_drop_null_values = False
            st.session_state.show_fill_section = False
            st.session_state.show_dataset = False

    with col2:
        if st.button('Drop Null Values'):
            st.session_state.show_drop_col_section = False
            st.session_state.show_drop_null_values = True
            st.session_state.show_fill_section = False
            st.session_state.show_dataset = False

    with col3:
        if st.button('Fill Null Values'):
            st.session_state.show_drop_col_section = False
            st.session_state.show_drop_null_values = False
            st.session_state.show_fill_section = True
            st.session_state.show_dataset = False

    with col4:
        if st.button('Show Dataset'):
            st.session_state.show_drop_col_section = False
            st.session_state.show_drop_null_values = False
            st.session_state.show_fill_section = False
            st.session_state.show_dataset = True


    # Drop Columns Section
    if st.session_state.show_drop_col_section:

        st.write("---")
        st.subheader("Drop Columns")
        feature_to_drop = st.selectbox(
            "Select Feature to Drop",
            st.session_state.nulls_info_df[st.session_state.nulls_info_df['Category'] == 'Drop Feature']['Feature'])
        if st.button('Confirm Drop'):
            try:
                if feature_to_drop:
                    st.session_state.df = st.session_state.df.drop(columns=feature_to_drop, axis=1)
                    st.session_state.show_dropdown = False
                    st.write('Column dropped Successfully.', st.session_state.df)
            except Exception as e:
                st.error(f"Error dropping column: {e}")

    #Drop Null Values Section
    if st.session_state.show_drop_null_values :
        st.write("---")
        st.subheader("Drop NULL Values")
        try:
            observations_to_drop = st.session_state.nulls_info_df[st.session_state.nulls_info_df['Category'] == 'Drop Observations']['Feature']
            st.session_state.df = st.session_state.df.dropna(subset=observations_to_drop)
            st.write('Null rows dropped Successfully', st.session_state.df)
        except Exception as e:
            st.error(f"Error dropping null values: {e}")

    # Fill Null Values Section
    if st.session_state.show_fill_section:
        st.write("---")
        st.subheader("Fill NULL Values")
        try:
            columns_to_fill = st.session_state.nulls_info_df[st.session_state.nulls_info_df['Category'] == 'Fill Values']['Feature']
            fill_col = st.selectbox('Select a column to FILL',columns_to_fill)

            if pd.api.types.is_numeric_dtype(st.session_state.df[fill_col]):
                fill_method = st.radio('Choose a method to fill NULL values',
                                       ('Zero', 'Mean', 'Median', 'Mode', 'Forwardfill', 'Backfill'))
            else:
                fill_method = st.radio('Choose a method to fill NULL values',
                                       ('Mode', 'Forwardfill', 'Backfill'))

            if st.button('Submit'):
                if fill_method == 'Zero':
                    st.session_state.df[fill_col].fillna(0, inplace=True)
                elif fill_method == 'Mean':
                    st.session_state.df[fill_col].fillna(st.session_state.df[fill_col].mean(), inplace=True)
                elif fill_method == 'Median':
                    st.session_state.df[fill_col].fillna(st.session_state.df[fill_col].median(), inplace=True)
                elif fill_method == 'Mode':
                    st.session_state.df[fill_col].fillna(st.session_state.df[fill_col].mode()[0], inplace=True)
                elif fill_method == 'Forwardfill':
                    st.session_state.df[fill_col].fillna(method='ffill', inplace=True)
                elif fill_method == 'Backfill':
                    st.session_state.df[fill_col].fillna(method='bfill', inplace=True)
                st.write('Nulls filled successfully.', st.session_state.df)
        except Exception as e:
            st.error(f"Error filling null values: {e}")

    # Show Dataset Section
    if st.session_state.show_dataset:
        st.write("---")
        st.subheader("Drop NULL Values")
        try:
            st.write('Updated Dataset:', st.session_state.df)
        except Exception as e:
            st.error(f"Error displaying dataset: {e}")


def remove_redundant_features():
    side_bar()   
    st.header("Remove Redundant Features")

    # Lets Handle Redundant Features
    def get_redundant_features(data, threshold = 0.8):
        try:
            redundant_features = []

            # Iterate through columns and calculate redundancy
            for col in data.columns:
                count_value = data[col].value_counts().max()
                percentage = (count_value / data.shape[0]) * 100

                if percentage > threshold * 100:
                    redundant_features.append([col, count_value, percentage])

            # Create a DataFrame with the results
            redundant_df = pd.DataFrame(redundant_features, columns=['Feature', 'Redundant Count', 'Percentage'])

            return redundant_df
        except Exception as e:
            st.error(f"Error identifying redundant features: {e}")
            return pd.DataFrame()

    try:
        redundant_df = get_redundant_features(st.session_state.df)
        st.dataframe(redundant_df)

        # Selectbox to choose a feature to drop
        feature_to_drop = st.selectbox("Select a redundant feature to drop:", redundant_df['Feature'])

        # Button to drop the selected feature
        if st.button("Drop Feature"):
            st.session_state.df = st.session_state.df.drop(columns=[feature_to_drop])
            st.write(f"Feature '{feature_to_drop}' has been dropped.")
            st.write("Updated DataFrame:")
            st.dataframe(st.session_state.df)
    except Exception as e:
        st.error(f"Error processing redundant features: {e}")


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


# Documentation

def documentation():
    """
    Displays the documentation page of the Dataset Preprocessor application.
    Provides detailed descriptions of each functionality, including uploading a dataset,
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
    """
    Creates a sidebar in the Streamlit application with navigation buttons for different sections
    of the Dataset Preprocessor application. It also initializes a session state variable for
    confirming reset if not already present.

    Parameters:
    None

    Returns:
    None
    """
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
            st.session_state.page = "Documentation"