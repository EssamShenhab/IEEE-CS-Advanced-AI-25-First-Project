def main():
    """
    The main function orchestrates the different functionalities of the dataset preprocessor application.
    It manages the navigation between different pages based on the user's actions.

    Parameters:
    None

    Returns:
    None
    """
    if "page" not in st.session_state:
        st.session_state.page = "upload"

    if st.session_state.page == "upload":
        f.input_data()
    elif st.session_state.page == "next":
        f.side_bar()
    elif st.session_state.page == "Visualization":
        f.visualization()
    elif st.session_state.page == "Handle Missing Values":
        f.handle_missing_values()
    elif st.session_state.page == "Remove Redundant Features":
        f.handle_missing_values()
    elif st.session_state.page == "Feature Selection":
        f.feature_selection()
    elif st.session_state.page == "Encode Data":
        f.encode_data()
    elif st.session_state.page == "Feature Scaling":
        f.feature_scaling()
    elif st.session_state.page == "Automatic Processing":
        f.automatic_processing()
    elif st.session_state.page == "Download Preprocessed Data":
        f.download_preprocessed_data()
    elif st.session_state.page == "Documentation":
        f.documentation()


if __name__ == "__main__":
    main()
    
