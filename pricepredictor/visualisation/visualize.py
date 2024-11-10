import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import pandas as pd
import matplotlib.dates as mdates
from pandas.core.series import Series


class PlotStocks:
    """
    A class for visualizing stock data, including candlestick charts,
    simple moving averages (SMA), residuals, and predicted closing prices.
    """

    def __init__(
        self,
        stockData: list[tuple[Series, Series, Series, Series, Series]],
        sma: list[float] = None,
        extrapolated_sma: list[float] = None,
        residuals: list[float] = None,
        predicted_closing_prices: list[float] = None,
        predicted_residuals: list[float] = None,
        sma_length: int = 3,
    ) -> None:
        """
        Initializes the PlotStocks class with stock data and
        optional parameters for additional visualizations.

        :param stockData: List of stock data tuples
        (date, open, high, low, close).
        :type stockData: list[tuple[
        pandas.Series,
        pandas.Series,
        pandas.Series,
        pandas.Series,
        pandas.Series]
        ]
        :param sma: Simple moving average values.
        :type sma: list[float], optional
        :param extrapolated_sma: Extrapolated SMA values for future dates.
        :type extrapolated_sma: list[float], optional
        :param residuals: The residuals
        :type residuals: list[float], optional
        :param predicted_closing_prices: Predicted closing prices.
        :type predicted_closing_prices: list[float], optional
        :param predicted_residuals: Predicted residual values.
        :type predicted_residuals: list[float], optional
        :param sma_length: Lookback period for calculating SMA.
        :type sma_length: int
        """
        self._dates, self._open, self._high, self._low, self._close = list(
            zip(*stockData)
        )
        self._sma = sma
        self._sma_length = sma_length
        self._extrapolated_sma = extrapolated_sma
        self._residuals = residuals
        self._predicted_closing_prices = predicted_closing_prices
        self._predicted_residuals = predicted_residuals

    def masterPlot(self) -> Figure:
        """
        Creates a comprehensive plot displaying candlestick data and residuals
        data.
        """
        self.masterPlot_on = True
        fig, (ax, ax1) = plt.subplots(
            2, 1, figsize=(8, 6), gridspec_kw={"height_ratios": [3, 1]}
        )

        ax.grid(color="#D3D3D3", linestyle="-", linewidth=0.5)
        ax1.grid(color="#D3D3D3", linestyle="-", linewidth=0.5)

        # Set range on the x - axis
        num_predictions = len(self._predicted_closing_prices)
        num_datum = len(self._close)
        num_days = num_datum - num_predictions
        x_min = self._dates[num_days]

        # Create the Plots
        self._plot_candlestick(ax)
        self._plot_residuals(ax1)

        # Adjust the spacing between subplots (increase hspace)
        plt.subplots_adjust(hspace=0.5)

        ax.set_xlim(left=x_min)
        ax1.set_xlim(left=x_min)

        fig.patch.set_facecolor(
            "lightgray"
        )  # Set a light background color for the entire figure
        ax.set_facecolor("whitesmoke")  # Set a lighter background for the first axis
        ax1.set_facecolor("whitesmoke")  # Set the same for the second axis

        self.masterPlot_on = False
        return fig

    def _plot_candlestick(self, ax: Axes) -> None:
        """
        Helper method to plot the candlestick data on a provided axis.

        :param ax: Axis on which to plot the candlestick data.
        :type ax: matplotlib.axes.Axes
        :param simpleMovingAverage: If True, includes SMA overlay.
        :type simpleMovingAverage: bool
        :param predictedClosingPrices: If True, includes predicted
        closing prices overlay.
        :type predictedClosingPrices: bool
        :param years_on_x_axis: Put only the years on the x axis.
        :type years_on_x_axis: bool
        """
        # Number of days
        num_data = len(self._high)

        # Create an array of index values to represent days
        dates = self._dates
        dates_numeric = mdates.date2num(dates)

        # Candlestick plotting
        for i in range(num_data):
            # Plot the line between high and low (the wick)
            ax.plot(
                [dates[i],
                 dates[i]],
                 [self._high[i], self._low[i]],
                 color="black"
                 )

            # Determine the color based on the open and close prices
            color = "green" if self._close[i] > self._open[i] else "red"
            # Plot the rectangle (the body) between open and close
            ax.add_patch(
                plt.Rectangle(
                    (dates_numeric[i] - 0.2, self._open[i]),
                    0.4,
                    abs(self._close[i] - self._open[i]),
                    color=color,
                )
            )

        self._plot_sma(ax, dates)

        self._plot_predicted_closing_prices(ax)

        # Formatting
        ax.xaxis.set_major_locator(mdates.DayLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        ax.tick_params(axis="x", labelrotation=90)

        ax.set_title("Stock Data")
        ax.set_ylabel("Price ($)")

        ax.legend(loc="lower right", bbox_to_anchor=(1, 0))

    def _plot_sma(self, ax: Axes, dates) -> None:
        """
        Helper method to plot the SMA on the
        specified axis, with optional extrapolated SMA.

        :param ax: Axis on which to plot the SMA.
        :type ax: matplotlib.axes.Axes
        :param dates: Dates for plotting the SMA.
        :type dates: list
        """
        # Delete the first n elements
        x_val_SMA = dates[self._sma_length - 1 :]
        ax.plot(x_val_SMA, self._sma, color="red", label="SMA")
        if self._extrapolated_sma is not None:
            # Getting the x value for the days that the extrapolation happend
            nr_days_extrapolated = len(self._extrapolated_sma)
            last_day = dates[-1]

            extrapolated_dates = pd.bdate_range(
                start=last_day, periods=nr_days_extrapolated
            ).tolist()

            ax.plot(
                extrapolated_dates,
                self._extrapolated_sma,
                color="yellow",
                label=f"Extrapolated SMA ({nr_days_extrapolated} days)",
            )

    def plot_residuals(self) -> None:
        """
        This method plots the residuals.

        :param ax1: used only when called
        """
        _, ax = plt.subplots()

        self._plot_residuals(ax)

        plt.show()

    def _plot_residuals(self, ax: Axes) -> None:
        """
        Helper method to plot residuals on the specified axis,
        with an option to include predicted residuals.

        :param ax: Axis on which to plot the residuals.
        :type ax: matplotlib.axes.Axes
        :param predictedResiduals: If True, includes
        predicted residuals in the plot.
        :type predictedResiduals: bool
        """
        # Number of data points
        num_data = len(self._residuals)

        # Create an array of index values to represent days
        dates = self._dates[-num_data:]  # range(num_data)

        ax.plot(dates, self._residuals, color="purple", label="Residuals")

        self._plot_predicted_residuals(ax)

        # Formatting
        ax.set_title("Plot of the residuals")
        ax.set_ylabel("Value")

        ax.legend()

    def _plot_predicted_residuals(self, ax: Axes) -> None:
        """
        Helper method to plot predicted residuals on the specified axis.

        :param ax: Axis on which to plot the predicted residuals.
        :type ax: matplotlib.axes.Axes
        """
        nr_days_extrapolated = len(self._predicted_residuals)
        last_day = self._dates[-1]

        future_dates = pd.bdate_range(
            start=last_day, periods=nr_days_extrapolated
        ).tolist()

        ax.plot(
            future_dates,
            self._predicted_residuals,
            color="orange",
            label="Predicted Residuals",
        )

    def plot_predicted_closing_prices(self) -> None:
        """
        Plots predicted closing prices on a new figure,
        showing future price predictions.
        """
        _, ax = plt.subplots()

        self._plot_predicted_closing_prices(ax)

        plt.show()

    def _plot_predicted_closing_prices(self, ax: Axes) -> None:
        """
        Helper method to plot predicted closing
        prices on the specified axis.

        :param ax: Axis on which to plot the
        predicted closing prices.
        :type ax: matplotlib.axes.Axes
        """
        nr_days_extrapolated = len(self._predicted_closing_prices)
        last_day = self._dates[-1]

        future_dates = pd.bdate_range(
            start=last_day, periods=nr_days_extrapolated + 1
        ).tolist()

        future_dates_numeric = mdates.date2num(future_dates[1:])

        for count, predicted_closing_price in enumerate(
            self._predicted_closing_prices
            ):
            rounded_closing_price = round(predicted_closing_price, 2)
            ax.hlines(
                rounded_closing_price,
                future_dates_numeric[count] - 0.2,
                future_dates_numeric[count] + 0.2,
                color="green",
                label="Predicted Closing Prices" if count == 0 else None,
            )


class PlotForcastComparison(PlotStocks):
    """
    A class for plotting stock data comparisons, including observed and
    predicted closing prices, simple moving averages (SMA), and residuals.

    Inherits from:
    PlotStocks: Provides the core plotting functionality and data handling.
    """

    def _plot_candlestick(self, ax: Axes) -> None:
        """
        Plots observed candlestick data on the provided axis.

        :param ax: The axis on which to plot the candlestick chart.
        :type ax: matplotlib.axes.Axes
        """
        # Number of days
        num_data = len(self._high)

        # Create an array of index values to represent days
        dates = self._dates
        dates_numeric = mdates.date2num(dates)

        # Candlestick plotting
        for i in range(num_data):
            ax.hlines(
                self._close[i],
                dates_numeric[i] - 0.2,
                dates_numeric[i] + 0.2,
                color="darkgreen",
                label="Observed Closing Prices" if i == 0 else None,
            )

        self._plot_sma(ax, dates)

        self._plot_predicted_closing_prices(ax)

        # Formatting
        ax.set_title("Stock Data")
        ax.set_ylabel("Price ($)")

        ax.legend()

    def _plot_residuals(self, ax: Axes) -> None:
        """
        Plots residuals of the observed data and predicted residuals.

        :param ax: The axis on which to plot residuals.
        :type ax: matplotlib.axes.Axes
        """
        self._residuals = self._residuals[-len(self._dates) :]
        super()._plot_residuals(ax)

    def _plot_sma(self, ax: Axes, dates: list[str]) -> None:
        """
        Plots the simple moving average (SMA) on the specified axis and
        the extrapolated SMA if available.

        :param ax: The axis on which to plot the SMA.
        :type ax: matplotlib.axes.Axes
        :param dates: Dates for plotting the SMA line.
        :type dates: list
        """
        # Delete the first n elements
        x_val_SMA = dates
        y_val_SMA = self._sma[-len(dates) :]
        ax.plot(x_val_SMA, y_val_SMA, color="gold", label="SMA")
        if self._extrapolated_sma is not None:
            # Getting the x value for the days that the extrapolation happend
            nr_days_extrapolated = len(self._extrapolated_sma)

            ax.plot(
                x_val_SMA,
                self._extrapolated_sma,
                color="darkorange",
                label=f"Extrapolated SMA ({nr_days_extrapolated} days)",
            )

    def _plot_predicted_closing_prices(self, ax: Axes) -> None:
        """
        Plots the simple moving average (SMA) on the specified axis and
        the extrapolated SMA if available.

        :param ax: The axis on which to plot the SMA.
        :type ax: matplotlib.axes.Axes
        :param dates: Dates for plotting the SMA line.
        :type dates: list
        """
        nr_days_extrapolated = len(self._predicted_closing_prices)
        first_day = self._dates[0]

        future_dates = pd.bdate_range(
            start=first_day, periods=nr_days_extrapolated
        ).tolist()

        future_dates_numeric = mdates.date2num(future_dates)

        for count, predicted_closing_price in enumerate(
            self._predicted_closing_prices
            ):
            rounded_closing_price = round(predicted_closing_price, 2)
            ax.hlines(
                rounded_closing_price,
                future_dates_numeric[count] - 0.2,
                future_dates_numeric[count] + 0.2,
                color="dodgerblue",
                label="Predicted Closing Prices" if count == 0 else None,
            )

    def _plot_predicted_residuals(self, ax: Axes) -> None:
        """
        Plots predicted residuals on the specified axis.

        :param ax: The axis on which to plot predicted residuals.
        :type ax: matplotlib.axes.Axes
        """
        nr_days_extrapolated = len(self._predicted_residuals)
        last_day = self._dates[0]

        future_dates = pd.bdate_range(
            start=last_day, periods=nr_days_extrapolated
        ).tolist()

        ax.plot(
            future_dates,
            self._predicted_residuals,
            color="green",
            label="Predicted Residuals",
        )
