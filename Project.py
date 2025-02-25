import streamlit as st

if "page" not in st.session_state:
    st.session_state.page = "upload"

def go_to_next():
    st.session_state.page = "next"


if st.session_state.page == "upload":
    st.title('Dataset Preprocessor')
    st.text('Welcome to Dataset Preprocessor: Your Gateway to Efficient Data Understanding and Preprocessing!')

    st.header('Upload your Dataset')
    uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx"])
    
    if uploaded_file is not None:
        btn = st.button('Continue', on_click=go_to_next)


elif st.session_state.page == "next":
    st.title('Dataset Preprocessor')
    st.session_state.click = True
    st.success("File uploaded successfully!")






    # else:
    #     st.text('Welcome to Dataset Preprocessor: Your Gateway to Efficient Data Understanding and Preprocessing!')

    #     st.session_state.click = False

# st.button(text)