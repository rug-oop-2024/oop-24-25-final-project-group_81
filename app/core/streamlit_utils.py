import streamlit as st
import time


class GeneralUI:
    """
    This is a class for Genral User Interface with streamlit.
    """

    def __init__(self):
        self.action: str | None = None
        self._action_list: list[str] | None = None
        self._sidebar_header: str = None

    def render_sidebar(self) -> None:
        """
        Render the sidebar for selecting an action.

        :param action_list: the list of actions
        :type action_list: list[str]
        """
        st.sidebar.header(self._sidebar_header)
        self.action = st.sidebar.selectbox("Choose Action", self._action_list)

    def progress_bar(self) -> None:
        """
        This method creates a progress bar.
        """
        progress_bar = st.progress(0)
        for percent_complete in range(100):
            time.sleep(0.01)
            progress_bar.progress(percent_complete + 1)
        progress_bar.empty()

    def display_error(self, message: str) -> None:
        """
        Display error message.

        :param message: the message you want
        to display
        :type message: str
        """
        st.error(message)

    def display_success(self, message: str) -> None:
        """
        Display success message.

        :param message: the message you want
        to display
        :type message: str
        """
        st.success(message)

    def button(self, message: str, **kwargs) -> bool:
        """
        Creates a button.

        :param message: the message displayed
        on the button
        :type message: str
        :return: whether it was clicked T/F
        :rtype: bool
        """
        return st.button(message, **kwargs)
