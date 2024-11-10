import streamlit as st
import os
import numpy as np

from app.core.streamlit_utils import GeneralUI
from app.core.datasets_utils import ControllerWithDatasets
from autoop.core.ml.feature import Feature
from autoop.core.ml.model import REGRESSION_MODELS, CLASSIFICATION_MODELS
from autoop.core.ml.metric import CLASSIFICATION_METRICS, REGRESSION_METRICS
from autoop.core.ml.metric import get_metric
from autoop.functional.feature import detect_feature_types
from autoop.core.ml.model import get_model
from autoop.core.ml.pipeline import Pipeline
from autoop.core.ml.metric import Metric


class UserInterfaceModelling(GeneralUI):
    def __init__(self):
        super().__init__()
        self._action_list = ["View Pipeline", "Execute", "Save"]
        self._sidebar_header = "The Pipeline is ready!"

    def display_model_types(
            self,
            list_of_model_types: list[str]
            ) -> str:
        """
        Display the model types that can be used in the pipeline.

        :return selected_model: the type of model the user selected.
        :type return: str
        """
        st.write("Chose the type of model you want to use.")
        selected_model = st.selectbox(
            "Options",
            list_of_model_types
            )
        return selected_model

    def display_models(
            self,
            models_list: list[str]
            ) -> str:
        """
        Displays the avaliable models to be used in making
        the pipeline.

        :param models_list: the list of avaliable models
        :type models_list: list[str]
        :return: the selected model.
        :rtype: str
        """
        st.subheader("Avaliable models")
        selected_model = st.selectbox("Choose a Model", models_list)
        return selected_model
    
    def display_model_info(
            self,
            model_name: str
            ) -> None:
        """
        Used to display some general info about the selected model by
        reading it from a file stored in "assets\\model_descriptions\\".
        The name of the file should follow the semantics of the model name.
        `Model Name` -> `model_name.txt`.

        :param model_name: the selected model name
        :type model_name: str
        """
        st.subheader(model_name)
        filename = model_name.\
            lower().\
                replace(" ", "_")
        file_path = "assets\\model_descriptions\\" + filename + ".txt"
        working_dir = os.getcwd()
        full_path = os.path.join(working_dir, file_path)
        with open(full_path, "r", encoding="utf-8") as file:
            description = file.read()
        st.write(description)

    def display_metrics(
            self,
            metrics_list: list[str]
            ) -> list[str]:
        """
        A way of displaying the matrics avaliable for the
        specific model type.

        :param metrics_list: a list of metrics as string
        :type metrics_list: list[str]
        :return: the selected metrics
        :rtype: list[str]
        """
        metrics_display_list = [
            metric.\
                replace("_", " ").\
                    title()
                    for metric in metrics_list
            ]
        st.subheader("Chose metric/s to evaluate the model.")
        metrics = st.multiselect("Avaliable metrics:", metrics_display_list)
        metrics_indecies = [
            index
            for index, value in enumerate(metrics_display_list)
            if value in metrics
            ]
        selected_metrics = [metrics_list[i] for i in metrics_indecies]
        return selected_metrics
    
    def display_input_features(
            self,
            features: list[Feature],
            feature_type: str
            ) -> list[str]:
        """
        Displays the avaliable input features and prompts the user
        to chose one or more from a multiselect box. It returns
        a list of the column names of the features selected by the user.

        :param features: a list of features
        :type features: list[Feature]
        :param feature_type: desired features type to filter
        all the features with
        :type feature_type: str
        :return: a list of the chosen features
        :rtype: list[str]
        """
        avaliable_columns = self._get_avaliable_features(
            features,
            feature_type
            )

        st.subheader("Chose input feature/s:")
        features_list = st.multiselect(
            "Avaliable input feature/s:",
            avaliable_columns
            )
        return features_list

    def display_target_features(
            self,
            features: list[Feature],
            input_features: list[Feature],
            feature_type: str
            ) -> Feature:
        """
        Displays the remaining features, by considering the choice
        the user made for input feature/s, and prompts the user
        to chose target features.

        :param features: all the features
        :type features: list[Feature]
        :param input_features: the input features
        :type input_features: list[Feature]
        :param feature_type: the type of features
        :type feature_type: str
        :return: the selected target feature
        :rtype: str
        """
        avaliable_columns = []
        input_features_names = [
            name.get_column_type()[0] for name in input_features
            ]
        for feature in features:
            # Get the name and the type of the feature
            name, type = feature.get_column_type()

            if type == feature_type and name not in input_features_names:
                # Append if it is not already in the input features
                avaliable_columns.append(feature)

        st.subheader("Chose target feature:")
        target_feature = st.selectbox(
            "Avaliable target feature:",
            avaliable_columns
            )
        return target_feature
    
    def show_results(
            self,
            exection: dict,
            decoder_classes: np.ndarray|None = None
            ) -> None:
        """
        Show the result of the metrics.

        :param exection: the return from the execute method
        of the pipeline.
        :type exection: dict
        """
        metrics_result = exection["metrics"]
        st.subheader("Metrics:")
        if decoder_classes is None:
            self._show_numerical_results(metrics_result)
        else:
            self._show_categorical_results(metrics_result, decoder_classes)

    def _show_numerical_results(
            self,
            metrics_result: tuple[Metric,float]
            ) -> None:
        """
        Shows the results from the metrics of a regression model.

        :param metrics_result: the metrics results
        :type metrics_result: tuple[Metric,float]
        """
        for metric_result in metrics_result:
            metric, result = metric_result
            st.write(f"{metric}: {round(result, 2)}")

    def _show_categorical_results(
            self,
            metrics_result: tuple[Metric,np.ndarray],
            decoder_classes: np.ndarray
            ) -> None:
        """
        Shows the results from the metrics of a categorical model.

        :param metrics_result: the results from the metrics
        :type metrics_result: tuple[Metric,np.ndarray]
        :param decoder_classes: the encoded classes
        :type decoder_classes: np.ndarray
        """
        for metric_result in metrics_result:
            metric, result = metric_result
            st.write(f"{metric}:")
            for count, res in enumerate(result):
                st.write(f" - {decoder_classes[count]}: {round(res, 2)}")

    def get_pipeline_saving_info(self) -> tuple[str, str]:
        """
        UI for saving a pipeline. Prompts the user
        to give name and a version to the pipleine.

        :return pipeline_name, version: the chosen name and version
        :type return: tuple[str, str]
        """
        st.subheader("Save your Pipeline")
        pipeline_name = st.text_input(
            "Pipeline Name",
            value="MyVeryNicePipeline"
            )
        version = st.text_input("Version", value="1.0")
        return pipeline_name, version

    def _get_avaliable_features(
            self,
            features: list[Feature],
            feature_type: str
            ) -> list[Feature]:
        """
        Gets the avaliable features from a feature list
        based on a chosen feature type.

        :param features: a list with all the features
        :type features: list[Feature]
        :param feature_type: the type of the desired
        features that need to be taken
        :type feature_type: str
        :return: a list of features that correspond to
        the type specified in feature_type
        :rtype: list[Feature]
        """
        avaliable_columns = []
        for feature in features:
            col, type = feature.get_column_type()
            if type == feature_type:
                avaliable_columns.append(feature)
        return avaliable_columns


class ControllerModelling(ControllerWithDatasets):
    def __init__(self) -> None:
        super().__init__()
        self.ui_manager = UserInterfaceModelling()
        self._reboot()

    @property
    def pipeline(self) -> Pipeline:
        """
        A getter for the pipeline.

        :return: the pipleine
        :rtype: Pipeline
        """
        return self._pipeline

    def run(self) -> None:
        """
        Main loop to run the application.
        """
        st.set_page_config(page_title="Modelling", page_icon="📈")

        st.write("# ⚙ Modelling")

        self._handle_build_pipeline()

        if self._pipeline is not None:
            self.ui_manager.render_sidebar()

        if self.ui_manager.action == "View Pipeline":
            self._handle_view_pipeline()

        elif self.ui_manager.action == "Execute":
            self._handle_execute_pipeline()
            
        elif self.ui_manager.action == "Save":
            self._handle_save_pipeline()

    def _handle_build_pipeline(self) -> None:
        """
        This method is used to handle the logic of building a Pipeline.
        It builds a pipeline by collecting the neccessary components.
        """
        self._handle_view_saved_datasets()
        if self._dataset is not None:
            self._select_model_type()

            if self._model_type != "Select Model Type":
                self._select_input_features()
                if self._input_features:
                    self._select_target_features()

        if self._target_features is not None:
            self._chose_metrics()

        if self._metrics is not None:
            self._select_model()

        if self._model is not None:
            self._chose_split()

        if self._split is not None:
            self._build_pipeline()

    def _handle_view_saved_datasets(self) -> None:
        """
        This method is responsible for handling the logic of viewing
        existing datasets. It gets the dataset and its feature.
        """
        super()._handle_view_saved_datasets()
        if self._dataset is not None:
            self._features = detect_feature_types(self._dataset)

    def _handle_view_pipeline(self) -> None:
        """
        A method to view the pipeline.
        """
        st.write(self._pipeline)

    def _handle_execute_pipeline(self) -> None:
        """
        A method to execute the pipeline.
        """
        self._exection = self._pipeline.execute()
        test_evaluation, train_evaluation = self._exection

        decoder_classes = self._pipeline.decoder_classes

        col1, col2 = st.columns(2)

        with col1:
            st.header("Test Evaluation")
            if decoder_classes is not None:
                self.ui_manager.show_results(test_evaluation, decoder_classes)
            else:
                self.ui_manager.show_results(test_evaluation)

        with col2:
            st.header("Train Evaluation")
            if decoder_classes is not None:
                self.ui_manager.show_results(train_evaluation, decoder_classes)
            else:
                self.ui_manager.show_results(train_evaluation)

    def _handle_save_pipeline(self) -> None:
        """
        A method that handles the saving of a pipeline.
        """
        pipeline_name, version = self.\
            ui_manager.get_pipeline_saving_info()

        if pipeline_name and version:
            # Gets a list of artifacts for saving
            pipeline_artifacts = self._pipeline.\
                artifacts(pipeline_name, version)
            
            if st.button("Save Pipeline"):
                for pipeline_artifact in pipeline_artifacts:
                    # Saves every element in the list of artifacts
                    self._automl.registry.register(pipeline_artifact)
                self.ui_manager.\
                    display_success(
                        f"Pipeline '{pipeline_name}' saved successfully!"
                        )

    def _build_pipeline(self) -> None:
        """
        This method creates an instance of a Pipeline.
        """
        self._pipeline = Pipeline(
            self._metrics,
            self._dataset,
            self._model,
            self._input_features,
            self._target_features,
            self._split)

    def _select_model(self) -> None:
        """
        Handle select specific model logic.
        """
        if self._model_type == "Regression Model":
            self._chose_model(REGRESSION_MODELS, "regression")

        elif self._model_type == "Classification Model":
            self._chose_model(CLASSIFICATION_MODELS, "classification")

    def _select_model_type(self) -> None:
        """
        Used to select a model type.
        """
        # Full list of model types
        list_of_model_types = [
            "Select Model Type", "Regression Model", "Classification Model"
            ]

        # Cheecks the avaliable features in the dataset
        list_of_types = self._check_features_type()

        if "numerical" not in list_of_types:
            list_of_model_types.remove("Regression Model")

        if "categorical" not in list_of_types:
            list_of_model_types.remove("Classification Model")

        # Select the model type
        model_type = self.ui_manager.\
            display_model_types(list_of_model_types)
        
        if model_type == "Select Model Type":
            st.write("Please select a model type to continue.")
            self._feature_type = None
        
        if model_type == "Regression Model":
            self._feature_type = "numerical"

        if model_type == "Classification Model":
            self._feature_type = "categorical"

        self._model_type = model_type

    def _select_input_features(self) -> None:
        """
        Used to select input features.
        """
        self._input_features = self.ui_manager.\
            display_input_features(
                self._features,
                "numerical"
                )

    def _select_target_features(self) -> None:
        """
        Used to select target features.
        """
        self._target_features = self.ui_manager.\
            display_target_features(
                self._features,
                self._input_features,
                self._feature_type
                )

    def _chose_model(
            self,
            list_of_models: list[str],
            type_of_models: str
            ) -> None:
        """
        Used to instantiate a specific model from a list of models.
        It utises the `get_model` function from the model package in
        this app.

        :param list_of_models: the list of avaliable models
        :type list_of_models: list[str]
        :param type_of_models: the specific type of models
        :type type_of_models: str
        """
        self.ui_manager.progress_bar()
        self.ui_manager.\
            display_success(f"You chose to use a {type_of_models} model.")
        
        selected_model = self.ui_manager.display_models(list_of_models)
        self.ui_manager.display_model_info(selected_model)
        self._model = get_model(selected_model)
        
    def _chose_metrics(self) -> None:
        """
        Used to chose metrics based on the type of model.
        """
        selected_metrics = None

        if self._model_type == "Classification Model":
            selected_metrics = self.ui_manager.\
                display_metrics(CLASSIFICATION_METRICS)

        if self._model_type == "Regression Model":
            selected_metrics = self.ui_manager.\
                display_metrics(REGRESSION_METRICS)

        if selected_metrics is not None:
            self._get_metrics(selected_metrics)

    def _get_metrics(
            self,
            selected_metrics: list[str]
            ) -> None:
        """
        Used to instantiate a metric by utilising the `get_metric`
        funciton in the metrics.py.

        :param selected_metrics: the selected metrics
        :type selected_metrics: list[str]
        """
        metrics_list = []
        for metric in selected_metrics:
            metric_instance = get_metric(metric)
            metrics_list.append(metric_instance)

        self._metrics = metrics_list

    def _check_features_type(self) -> list[str]:
        """
        Checks the type of features in the dataset.

        :return list_of_types: a list consting of the types of features
        :type return: list[str]
        """
        list_of_types = []
        for feature in self._features:
            list_of_types\
                .append(
                    feature.type
                    )
        
        # In case the user has selected a dataset with 1 feature
        if len(list_of_types) == 1:
            self.ui_manager.display_error(
                "No avaliable model because of invalid dataset."
                "You can not do any analysis on a single column!"
                "You need to chose a different dataset."
                )
            self.ui_manager.progress_bar()
            self._reboot()
            return self.run()
        
        return list_of_types
    
    def _chose_split(self) -> None:
        """
        Used to chose a split for the pipeline.
        """
        self._split = st.slider("Select a dataset split", 0.01, 0.99)
    
    def _reboot(self) -> None:
        """
        Used to reboot the pipeline creation procedure.
        """
        self._model_type = None
        self._model = None
        self._dataset = None
        self._metrics = None
        self._features = None
        self._feature_type = None
        self._input_features = None
        self._target_features = None
        self._split = None
        self._pipeline = None
        self._exection = None

if __name__ == "__main__":
    control = ControllerModelling()
    control.run()
