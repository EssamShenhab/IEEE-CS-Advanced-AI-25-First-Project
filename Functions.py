import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
sys.path.append("/Users/macbookpro/streamlit101/")
import Functions as f


import streamlit as st
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

def feature_selection(df):
    side_bar()
    st.header("**Feature_Selection**")

    if "df" not in st.session_state:
        st.error("Please upload a dataset first.")
        return

    num = df.select_dtypes(include=['number'])

    if num.empty:
        st.write("**No numerical columns found :(**")
        return

    plt.figure(figsize=(10, 6))
    sns.heatmap(num.corr(), annot=True, cmap='coolwarm', fmt='.2f')
    st.pyplot(plt)

    drp_col = st.multiselect("Select columns you want to drop", num.columns)
    if drp_col:
        st.session_state.df = df.drop(columns=drp_col)
        st.write("Updated DF:")
        st.write(st.session_state.df.head())

def encode_data(df):
    side_bar()
    st.header("**Encode-Data (caegorical ones) **")

    if "df" not in st.session_state:
        st.error("Please upload a dataset first.")
        return

    cat_df = df.select_dtypes(include=['object'])

    if cat_df.empty:
        st.write("No categorical cols found :(")
        return

    st.write("Unique vals per each categorical col: ")
    cat_info = {col: df[col].nunique() for col in cat_df.columns}
    st.write(cat_info)

    col_to_encode = st.selectbox("Select column to encode", cat_df.columns)
    encoding_type = st.radio("Encoding Type: ", ("Encoding_d_label", "One-Hot-Encoding"))

    if st.button("Encoding"):
        if encoding_type == "Encoding_d_label":
            le = LabelEncoder()
            st.session_state.df[col_to_encode] = le.fit_transform(df[col_to_encode])
        elif encoding_type == "One-Hot-Encoding":
            st.session_state.df = pd.get_dummies(df, columns=[col_to_encode], drop_first=True)

        st.write("Updated Df:")
        st.write(st.session_state.df.head())
