import streamlit as st
import time


class GeneralUI:
    """
    This is a class for Genral User Interface with streamlit.
    """
    def progress_bar(self):
        """
        This method creates a progress bar.
        """
        progress_bar = st.progress(0)
        for percent_complete in range(100):
            time.sleep(0.01)
            progress_bar.progress(percent_complete + 1)
        progress_bar.empty()

    def display_error(self, message) -> None:
        """
        Display error message.
        """
        st.error(message)

    def display_success(self, message) -> None:
        """
        Display success message.
        """
        st.success(message)
