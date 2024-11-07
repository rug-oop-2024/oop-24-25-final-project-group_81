import streamlit as st
import pickle
import os
import pandas as pd
from typing import Any

from app.core.abstract_server import AbstractServer
from autoop.core.ml.model import get_model
from autoop.core.ml.feature import Feature
from autoop.core.ml.pipeline import Pipeline
from autoop.core.ml.dataset import Dataset
from autoop.core.ml.metric import Metric

   
class ControllerWithPipelines(AbstractServer):
    """
    This class serves as a parent class for all classes trying to
    view the database for datasets. It inherits from AbstractServer
    which is the generic way of viewing items in the database.
    """
    def __init__(self) -> None:
        """
        A way of insantiating the object. Sets the constructor
        of the AbstractServer in such a way as to look for `dataset`
        items in the database.
        """
        super().__init__(item_name = "pipeline")

    def _handle_view_saved_items_logic(self):
        """
        Handles the logic of viewing save pipelines in the database.
        """
        artifact_attributes = super()._handle_view_saved_items_logic()
        if artifact_attributes is not None:
            self._pipeline = self._reassemble_pipeline(artifact_attributes)

    def _reassemble_pipeline(
            self,
            artifact_attributes: list[Any]
            ) -> Pipeline:
        """
        Manges the reassembly of the pipeline

        :param artifact_attributes: the artifact attributes of the
        pipeline
        :type artifact_attributes: list[Any]
        :return: an instance of Pipeline
        :rtype: Pipeline
        """
        # Loads the data from the artifact
        pipeline_data = pickle.loads(artifact_attributes["data"])

        # Retrieve the input and output features
        self._retrieve_features(pipeline_data)
        
        # Retrieve split
        self._split: int = pipeline_data["split"]

        # Retrieve dataset
        dataset_id: str = pipeline_data["dataset_id"]
        self._retrieve_dataset(dataset_id)
        
        # Retrieve metrics
        self._metrics: list[Metric] = pipeline_data["metrics"]

        # Retrieve model
        model_id = artifact_attributes["metadata"]["model_id"]
        self._retrieve_model(model_id)

        return Pipeline(
            self._metrics,
            self._dataset,
            self._model,
            self._input_features,
            self._target_feature,
            self._split
            )
    
    def _retrieve_features(self, pipeline_data: dict) -> None:
        """
        Used to retrieve the input and output features of the Pipeline

        :param pipeline_data: the pipeline data as dict
        :type pipeline_data: dict
        """
        self._input_features: list[Feature] = pipeline_data["input_features"]
        self._target_feature: Feature = pipeline_data["target_feature"]

    def _retrieve_dataset(self, dataset_id: str) -> None:
        """
        Used to retrieve the dataset used in the pipeline.

        :param dataset_id: the dataset id
        :type dataset_id: str
        """
        dataset_artifact = self._automl.registry.get(dataset_id)
        dataset_attributes = vars(dataset_artifact)
        self._dataset = Dataset(**dataset_attributes)

    def _retrieve_model(self, model_id: str) -> None:
        """
        Retrieves the model used in the Pipeline

        :param model_id: the model id
        :type model_id: str
        """
        artifact_model = self._automl.registry.get(model_id)

        # Split on the colon and take the second part
        artifact_model_name = artifact_model.name.split(":")[1]

        # Split on space and take the first part,
        # then replace underscores with spaces
        self._model_name = artifact_model_name.split(" ")[0]\
            .replace("_", " ")
        
        self._model = get_model(self._model_name)

    def _display_item(self):
        """
        Display a preview of the pipeline.
        """
        st.write("# Preview of Pipeline:")

        # Display Model
        st.write("# Model:")
        self._model_info(self._model_name)

        # Display Split
        st.write(f"# Split: {float(self._split)}")

        # Create two columns:
        # one for Input Features and one for Target Feature
        col1, col2 = st.columns(2)

        with col1:
            st.write("# Input feature/s:")
            self._display_features(self._input_features)

        with col2:
            st.write("# Target feature:")
            self._display_features([self._target_feature])

    def _display_features(self, features: list[Feature]):
        """
        Display the features in a nice way in the form of dataframes.

        :param features: a list of features
        :type features: list[Feature]
        """
        dict_features = {}
        df = self._dataset.read()
        
        for feature in features:
            key = feature.name
            dict_features[key] = df[key]

        st.dataframe(pd.DataFrame(dict_features), hide_index = True)

    def _model_info(self, model_name: str) -> None:
        """
        Used to display the model info from a file.

        :param model_name: the name of the model
        :type model_name: str
        """
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
        