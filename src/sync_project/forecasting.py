"""This file contains classes useful for forecasting."""

from collections.abc import Callable

import numpy
import pandas
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing

# pylint: disable=abstract-method

# Forecasting method class wrappers:

# Statistical methods:

# Add ARIMA


class SingleExponentialSmoothing(
    SimpleExpSmoothing,
):
    """Wrapper class to hold Single Exponential
    Smoothing forecasting methods and information."""

    def __init__(self, time_data: pandas.Series, value_data: pandas.Series):
        """Initialization method for class.

        Args:
            time_data (pandas.Series):
                Series of times to be used in
                fitting Single Exponential
                Smoothing forecasting method.
            value_data (pandas.Series):
                Series of data values to be used in
                fitting single Exponential
                Smoothing forecasting method.
        """
        # Check for data types.
        self.time_data = time_data
        self.value_data = value_data
        self.data_series = pandas.Series(
            self.value_data.to_numpy(),
            self.time_data.to_numpy(),
        )
        super().__init__(self.data_series, initialization_method="estimated")
        self.forecasting_method = "Single Exponential Smoothing"


class TripleExponentialSmoothing(
    ExponentialSmoothing,
):
    """Wrapper class to hold Triple Exponential
    Smoothing forecasting methods and information."""

    def __init__(self, time_data: pandas.Series, value_data: pandas.Series):
        """Initialization method for class.

        Args:
            time_data (pandas.Series):
                Series of times to be used in
                fitting Triple Exponential
                Smoothing forecasting method.
            value_data (pandas.Series):
                Series of data values to be used in
                fitting Triple Exponential
                Smoothing forecasting method.
        """
        # Check for data types.
        self.time_data = time_data
        self.value_data = value_data
        self.data_series = pandas.Series(
            self.value_data.to_numpy(),
            self.time_data.to_numpy(),
        )
        super().__init__(self.data_series, initialization_method="estimated")
        self.forecasting_method = "Triple Exponential Smoothing"


# Machine Learning methods:

# Add scikit-learn methods

# Add XGBoost method

# Add LLM method?


# Class to find and use best forecasting method:
class Forecast:  # pylint: disable=too-many-instance-attributes
    """Class to find the best forecasting method
    and parameters, of the given forecasting
    methods."""

    def __init__(  # pylint: disable=too-many-arguments
        self,
        training_time_data: pandas.Series,
        training_value_data: pandas.Series,
        validation_time_data: pandas.Series,
        validation_value_data: pandas.Series,
        methods: list | None = None,
        metric: Callable = mean_squared_error,
    ):
        """Initialization method for class.

        Args:
            training_time_data (pandas.Series):
                Series of times to be used in
                fitting forecasting methods.
            training_value_data (pandas.Series):
                Series of data values to be used in
                fitting forecasting methods.
            validation_time_data (pandas.Series):
                Series of times to be used in
                validating forecasting methods.
            validation_value_data (pandas.Series):
                Series of data values to be used in
                validating forecasting methods.
            methods (List[Class] | None): List of
                classes of forecasting methods. If
                None, uses a pre-defined list of
                classes. Defaults to None.
            metric (Callable): Metric to use to
                determine best forecast method.
                Defaults to 'mean_squared_error'.
        """
        self.training_time_data = training_time_data
        self.training_value_data = training_value_data
        self.training_data = pandas.Series(
            self.training_value_data.to_numpy(),
            self.training_time_data.to_numpy(),
        )
        self.validation_time_data = validation_time_data
        self.validation_value_data = validation_value_data
        self.validation_data = pandas.Series(
            self.validation_value_data.to_numpy(),
            self.validation_time_data.to_numpy(),
        )
        self.methods = (
            methods
            if methods
            else [
                SingleExponentialSmoothing,
                TripleExponentialSmoothing,
            ]
        )
        self.metric_values = [None for _ in self.methods]
        self.metric = metric
        self.best_forecast_method = None
        self.best_forecast_fit = None
        # Figure out how many forecasting data
        # points should be included to compare with
        # validation data.
        # First, determine if the time deltas between
        # data points areidentical.
        training_time_deltas = (
            self.training_time_data[1:].array
            - self.training_time_data[:-1].array
        )
        validation_time_deltas = (
            self.validation_time_data[1:].array
            - self.validation_time_data[:-1].array
        )
        if len(numpy.unique(training_time_deltas)) == 1:
            if (
                len(numpy.unique(validation_time_deltas)) == 1
                and training_time_deltas[0] == validation_time_deltas[0]
            ):
                # If all the time deltas are
                # identical, we can simply use
                # the length of the validation data.
                self.forecast_length = len(self.validation_time_data)
            else:
                # If the time deltas for training
                # are uniform but for validation
                # are not (or are not the same as
                # for training), calculate the
                # forecast length based on the
                # training time delta.
                self.forecast_length = 1
                current_time = (
                    self.training_time_data.array[-1] + training_time_deltas[0]
                )
                while (  # pylint: disable=while-used
                    self.validation_time_data.array[-1] > current_time
                ):
                    self.forecast_length += 1
                    current_time += training_time_deltas[0]
        else:
            # If the training time deltas are not
            # uniform, just set the forecast length
            # to that of the validation data.
            # I suspect this will cause issues when
            # calculating metrics.
            self.forecast_length = len(self.validation_time_data)

    def fit(self, comparison_method: Callable = min):
        """Method to find the best forecast method
        of the given options.

        Args:
            comparison_method (Callable): Method to
                compare calculated metric values,
                to determine the best value.
                Defaults to 'min'.
        """
        for ind, method in enumerate(self.methods):
            fit = method(
                self.training_time_data,
                self.training_value_data,
            ).fit()
            self.metric_values[ind] = self.metric(
                self.validation_data,
                fit.forecast(self.forecast_length),
            )
        self.best_forecast_method = self.methods[
            self.metric_values.index(comparison_method(self.metric_values))
        ](
            pandas.concat(
                [
                    self.training_time_data,
                    self.validation_time_data,
                ],
            ),
            pandas.concat(
                [
                    self.training_value_data,
                    self.validation_value_data,
                ],
            ),
        )
        assert self.best_forecast_method is not None
        self.best_forecast_fit = self.best_forecast_method.fit()
        # Get forecast model params with:
        # self.best_forecast_fit.model.params
        return self.best_forecast_fit

    def forecast(self, forecast_length: int) -> pandas.Series:
        """Calls the forecast method of the
        previously fit best_forecast_method.

        Args:
            forecast_length (int): Number of
                forecasts to predict.

        Returns:
            pandas.Series: Series of forecasts.
        """
        if not self.best_forecast_method:
            raise RuntimeError(
                "The 'forecast' method can only be"
                " called after the 'fit' method."
            )
        return self.best_forecast_fit.forecast(
            forecast_length,
        )
