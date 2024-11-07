import streamlit as st
import pickle
import os

from app.core.abstract_server import AbstractServer
from autoop.core.ml.model import get_model

   
class ControllerWithPipelines(AbstractServer):
    """
    This class serves as a parent class for all classes trying to
    view the database for datasets. It inherits from AbstractServer
    which is the generic way of viewing items in the database.
    """
    def __init__(self):
        """
        A way of insantiating the object. Sets the constructor
        of the AbstractServer in such a way as to look for `dataset`
        items in the database.
        """
        super().__init__(item_name = "pipeline")

    def _handle_view_saved_items_logic(self) -> dict:
        artifact_attributes = super()._handle_view_saved_items_logic()
        if artifact_attributes is not None:
            self._pipeline = self._reassemble_pipeline(artifact_attributes)

    def _reassemble_pipeline(self, artifact_attributes):
        pipeline_data = pickle.loads(artifact_attributes["data"])
        model_id = artifact_attributes["metadata"]["model_id"]
        artifact_model = self._automl.registry.get(model_id)
        
        self._input_features = pipeline_data["input_features"]
        self._target_feature = pipeline_data["target_feature"]
        self._split = pipeline_data["split"]

        # Split on the colon and take the second part
        artifact_model_name = artifact_model.name.split(":")[1]

        # Split on space and take the first part, then replace underscores with spaces
        self._model_name = artifact_model_name.split(" ")[0]\
            .replace("_", " ")
        
        self._model = get_model(self._model_name)

    def _display_item(self):
        """
        Display the dataset as a dataframe.

        :param df: the dataframe to be dispalyed
        """
        st.write("# Preview of Pipeline:")
        st.write("# Model:")
        self._model_info(self._model_name)
        st.write("# Input feature/s:")
        st.write(self._input_features)
        st.write("# Target feature:")
        st.write(self._target_feature)
        st.write("# Split:")
        st.write(self._split)

    def _model_info(self, model_name: str):
        st.write(model_name)
        filename = model_name.\
            lower().\
                replace(" ", "_")
        file_path = "assets\\model_descriptions\\" + filename + ".txt"
        working_dir = os.getcwd()
        full_path = os.path.join(working_dir, file_path)
        with open(full_path, "r", encoding="utf-8") as file:
            description = file.read()
        st.write(description)
        