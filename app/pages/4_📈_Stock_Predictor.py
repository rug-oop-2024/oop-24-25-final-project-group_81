import streamlit as st
from datetime import date, timedelta
import os

from app.core.streamlit_utils import GeneralUI
from pricepredictor.forcast.forecastFactory_initializer import ForcastFactoryInitializer
from pricepredictor.forcast.forcastFactory import ForcastFactory


class UserInterfaceStockPredictions(GeneralUI):
    """
    UI for Stock Predictions.
    """
    def __init__(self) -> None:
        """
        A way of instantiating an UI.
        """
        super().__init__()
        self._action_list = ["Information", "Setup the model"]
        self._sidebar_header = "Actions"

    def display_info(
        self,
    ) -> None:
        """
        Displays useful information to the user by loading it from filepath.
        """
        st.subheader("General Information")
        file_path = "assets\\model_descriptions\\pricepredictor.txt"
        working_dir = os.getcwd()
        full_path = os.path.join(working_dir, file_path)
        with open(full_path, "r", encoding="utf-8") as file:
            description = file.read()
        st.write(description)

    def show_avaliable_stocks(self) -> str:
        """
        Shows the avaliable stocks and prompts the user to chose one.

        :return: the chosen stock.
        :rtype: str
        """
        st.write("# Chose a financial assset")
        selected_stocks = st.selectbox("Options", ["AAPL", "MSFT"])
        return selected_stocks

    def avaliable_dates_for_predictions(self) -> date:
        """
        Shows the avaliable dates for predictions and promopt the
        user to chose.

        :return: the user's choice.
        :rtype: date
        """
        st.write("# Select a date to do the prediction/s from")
        # Define the date range
        min_date = date(2022, 1, 1)
        default_date = date(2024, 10, 1)

        selected_date = st.date_input(
            "Avaliable dates",
            value=default_date,  # Default to today’s date
            min_value=min_date,  # Set the minimum selectable date
            max_value=date.today(),  # Set the maximum selectable date
        )
        return selected_date

    def display_num_of_predictions(self) -> int:
        """
        Displays the avaliable number of predictions and prompts the
        user to chose a number.

        :return: the user's chosen number of predictions.
        :rtype: int
        """
        st.write("# Chose number of predictions")
        num_of_predictions = st.slider(
            "Options", min_value=1, max_value=5, step=1, value=5
        )
        return num_of_predictions

    def avaliable_num_neurons(self) -> int:
        """
        Prompts the user to select number of neurons for the first and second
        hidden layer in the MLP used to predict residuals.

        :return: the user's choice of neuron configuration.
        :rtype: int
        """
        st.write(
            "# Chose number of neurons in the first " +
            "and second hidden layer of the MLP"
        )
        first_layer_number = st.number_input(
            "Select nummber of neurons in the first layer:",
            min_value=4,
            max_value=16,
            value=8,
            step=1,
        )

        second_layer_number = st.number_input(
            "Select nummber of neurons in the second layer:",
            min_value=0,
            max_value=16,
            value=0,
            step=1,
        )

        return first_layer_number, second_layer_number

    def avaliable_learning_rates(self) -> float:
        """
        Prompts the user to chose a learning rate.

        :return: the chosen learning rate.
        :rtype: float
        """
        st.write("# Chose a learning rate")
        learning_rate = st.number_input(
            "Select a learning rate:",
            min_value=0.0005,
            max_value=0.0100,
            value=0.0035,
            step=0.0001,
            format="%.4f",
        )
        return learning_rate

    def avaliable_batch_size(self) -> int:
        """
        Prompts the user to chose a batch size.

        :return: the chosen batch size.
        :rtype: int
        """
        st.write("# Chose a batch size")
        batch_size = st.slider("Select a batch size", 1, 5, step=1, value=3)
        return batch_size


class ControllerStockPredictions:
    """
    Controller for Stock Predictions.
    """
    def __init__(self) -> None:
        """
        A way of instantiating ControllerStockPredictions.
        """
        super().__init__()
        self.ui_manager = UserInterfaceStockPredictions()

    def run(self) -> None:
        """
        Main loop to run the application.
        """
        self.ui_manager.\
            render_sidebar()

        if self.ui_manager.action == "Information":
            self._handle_information()

        if self.ui_manager.action == "Setup the model":
            self._handle_model_setup()

            if self._forcast_factory is not None:
                self._chose_button()

    def _handle_get_forcast(self) -> None:
        """
        Handles the logic behind getting a forcast.
        """
        fig = self._forcast_factory.plot_predictions()
        st.pyplot(fig)

    def _handle_comparison(self) -> None:
        """
        Handles the logic behind getting a comparison.
        """
        # Compare the predictions with observations
        mse = self._forcast_factory.\
            compare_predictions_with_observations()
        fig = self._forcast_factory.\
            plot_comparison()
        st.pyplot(fig)
        st.write("# Mean Squared Error")
        st.write(
            "The MSE of the predictons compared to the actual"
            f" closing prices is: {round(mse, 2)} $"
        )

    def _handle_model_setup(self) -> None:
        """
        Handles the logic behind setting up the model.
        """
        self._chose_stock()

        if self._stock is not None:
            self._chose_date()

        if self._date is not None:
            self._chose_num_predictions()

        if self._num_predictions is not None:
            self._chose_architecture()

        if self._architecture is not None:
            self._chose_learning_rate()

        if self._learning_rate is not None:
            self._chose_batch_size()

        if self._batch_size is not None:
            self._build_forcast_factory()

    def _handle_information(self) -> None:
        """
        Handles the logioc behind displaying the information.
        """
        self.ui_manager.\
            display_info()

    def _chose_button(self) -> None:
        """
        Gives the user a choice between prediction or comparison.
        """
        st.write("# Chose an Action")

        # Get two columns
        col1, col2 = st.columns(2)
        comparison_button = False

        with col1:
            forcast_button = self.ui_manager.\
                button("Get a forcast")
        end_date_of_prediction = (
            self._date + timedelta(days=self._num_predictions + 1)
        )
        if end_date_of_prediction < date.today():
            with col2:
                comparison_button = self.ui_manager.\
                    button("Get a comparison")

        if forcast_button:
            self._handle_get_forcast()

        if comparison_button:
            self._handle_comparison()

    def _build_forcast_factory(self) -> None:
        """
        Builds a forcast factory.
        """
        initializer = ForcastFactoryInitializer()

        model_parameters = initializer.generate_model_parameters(
            self._architecture,
            self._learning_rate,
            batch_size = self._batch_size
        )

        # Generate datafacotry parameters
        datafactory_parameters = initializer.\
            generate_datafactory_parameters()

        self._forcast_factory = ForcastFactory(
            self._stock,
            model_parameters,
            datafactory_parameters
        )

        self._forcast_factory.\
            predict(
                self._num_predictions,
                end_date=self._date_str
            )

    def _chose_stock(self) -> None:
        """
        Choses a stock.
        """
        self._stock = self.ui_manager.\
            show_avaliable_stocks()

    def _chose_date(self) -> None:
        """
        Choses avaliable date and saves it as an attribute.
        It saves both the date as datetime and as string.
        """
        self._date = self.ui_manager.\
            avaliable_dates_for_predictions()
        self._date_str = self._date.\
            strftime("%Y-%m-%d")

    def _chose_num_predictions(self) -> None:
        """
        Choses the number of predictions.
        """
        self._num_predictions = self.ui_manager.\
            display_num_of_predictions()

    def _chose_architecture(self) -> None:
        """
        Constructs the architecture. If the number of neurons in the
        second layer is left to 0 it creates only 1 hidden layer.
        This is prefered as the model performs best.
        """
        first_layer, second_layer = self.ui_manager.\
            avaliable_num_neurons()
        if second_layer == 0:
            self._architecture = [first_layer]
        else:
            self._architecture = [first_layer, second_layer]

    def _chose_learning_rate(self) -> None:
        """
        Choses the learning rate.
        """
        self._learning_rate = self.ui_manager.\
            avaliable_learning_rates()

    def _chose_batch_size(self) -> None:
        """
        Choses the batch size.
        """
        self._batch_size = self.ui_manager.\
            avaliable_batch_size()


if __name__ == "__main__":
    controller = ControllerStockPredictions()
    controller.run()
