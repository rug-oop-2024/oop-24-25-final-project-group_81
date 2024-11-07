import streamlit as st
import pandas as pd
import os
from typing import Any

from app.core.system import AutoMLSystem
from app.core.streamlit_utils import GeneralUI
from app.core.deployment_utils import ControllerWithPipelines


st.set_page_config(page_title="Deployment", page_icon="🚀")

def write_helper_text(text: str):
    st.write(f"<p style=\"color: #888;\">{text}</p>", unsafe_allow_html=True)

st.write("# 🚀 Deployment")

st.write("You can use this page to load and deploy existing Pipelines!")

automl = AutoMLSystem.get_instance()

pipelines = automl.registry.list(type="pipeline")

# your code here

class UserInterfaceDeployment(GeneralUI):
    def __init__(self):
        self.action = None

    def render_sidebar(self) -> None:
        """
        Render the sidebar for selecting an action.
        """
        st.sidebar.header("Actions")
        self.action = st.sidebar.\
            selectbox("Choose Action", ["Upload Dataset", "View Datasets"])


class ControllerDeployment(ControllerWithPipelines):
    def __init__(self):
        super().__init__()
        self.ui_manager = UserInterfaceDeployment()

    def run(self):
        """
        Main loop to run the application.
        """
        self._handle_view_saved_items_logic()
        self._display_item()

if __name__ == "__main__":
    control = ControllerDeployment()
    control.run()