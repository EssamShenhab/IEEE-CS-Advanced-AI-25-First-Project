import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import numpy as np
from sklearn.preprocessing import MinMaxScaler

le = LabelEncoder()
encoder = OneHotEncoder()
scaler = MinMaxScaler()



st.title('Automated data preprocessing')
if 'df' in st.session_state:
    df = st.session_state.df

##handle null values



    def clean_null(df):

        null_values = pd.DataFrame(df.isnull().sum(), columns=['null_count'])
        null_values['null_percentage'] = null_values['null_count'] / df.shape[0]
        
        null_features = []
        null_rows = []
        
        for row in null_values.itertuples():
            if row.null_percentage >= 0.8:
                null_features.append(row.Index)
            if 0 < row.null_percentage <= 0.03:
                null_rows.append(row.Index)

        categorical_cols = []
        numerical_cols = []
        
        for col, dtype in df.dtypes.items():
            if dtype in ['int64', 'float64']:
                numerical_cols.append(col)
            else:
                categorical_cols.append(col)

        df.drop(null_features, axis=1, inplace=True)
        df.dropna(subset=null_rows, inplace=True)

        if numerical_cols:
            df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].mean())
        if categorical_cols:
            df[categorical_cols] = df[categorical_cols].fillna(df[categorical_cols].mode().iloc[0])

        return df

    def duplicates(df):
        df.drop_duplicates(inplace=True)
        return(df)


###### redundant

    def most_redundant_value(df):
        redundant_info = {}

        for col in df.columns:
            most_frequent_value = df[col].mode().iloc[0]  # Get the most common value
            count = df[col].value_counts().iloc[0]        # Get the count of that value

            redundant_info[col] = {
                'most_redundant_value': most_frequent_value,
                'repeated_times': count
            }
        redundant_df =pd.DataFrame(redundant_info).transpose()

        redundant_df['red_percentage'] = redundant_df['repeated_times'] /df.shape[0]
        redunduant_features = []
        for row in redundant_df.itertuples():
            if row.red_percentage >= 0.8:
                redunduant_features.append(row.Index)
            
        df.drop(redunduant_features, axis=1, inplace=True)

        
        return df
    



    def encoding(df):
        categorical_cols = df.select_dtypes(include=['object']).columns
        for column in categorical_cols:
            unique = df[column].nunique()

            if unique==2:
                df[column] = le.fit_transform(df[column])

            else:
                encoded = pd.get_dummies(df[column], prefix=column, drop_first=True).astype(int)
                df.drop(column, axis=1, inplace=True)
                df = pd.concat([df, encoded], axis=1)
        return df


    def normalization(df):
        columns =df.columns
        numeric_cols = df.select_dtypes(include=['number']).columns

        normalized_data = scaler.fit_transform(df[numeric_cols])

        return pd.DataFrame(normalized_data , columns= columns) 



    st.subheader("Choose preprocessing options")


    null_values =st.checkbox('Handle null values')
    duplicate = st.checkbox('Remove duplicates')
    redundant_values = st.checkbox('Handle redundant values')

    normalize = st.checkbox('Normalization' )
    encode =  st.checkbox('Encode')

    
    start = st.button('process')
    # if normalize and not encode:
    st.warning("If you want to Normalize you have to choose encode")
    def process(df):
        processed_df =df.copy()
        if null_values: processed_df = clean_null(processed_df)

        if duplicate: processed_df = duplicates(processed_df)

        if redundant_values: processed_df = most_redundant_value(processed_df)

        if encode: processed_df = encoding(processed_df)
        if normalize: processed_df = normalization(processed_df)

        return processed_df
        
    if start:
        processed_df =process(df)


        st.table(processed_df.head())
        
        file_name = st.text_input('write the file name to download it')

        csv_data = processed_df.to_csv(index=False).encode('utf-8')
        st.download_button(
        label="Download data as CSV",
        data=csv_data,
        file_name= file_name +'.csv',
        mime='text/csv')




        




### handle dupliactes


## remove 


else: st.warning('please uplaod your dataset')

