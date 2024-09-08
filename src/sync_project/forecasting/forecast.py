"""This file contains classes useful for forecasting and
fitting the best forecasting method."""

from collections.abc import Callable

import numpy
import pandas
from sklearn.metrics import mean_squared_error

from sync_project.forecasting.statistics.single_exponential_smoothing import (
    SingleExponentialSmoothing,
)
from sync_project.forecasting.statistics.triple_exponential_smoothing import (
    TripleExponentialSmoothing,
)


class Forecast:  # pylint: disable=too-many-instance-attributes
    """Class to find the best forecasting method
    and parameters, of the given forecasting
    methods."""

    def __init__(  # pylint: disable=too-many-arguments
        self,
        training_time_data: numpy.ndarray | pandas.Series,
        training_value_data: numpy.ndarray | pandas.Series,
        validation_time_data: numpy.ndarray | pandas.Series,
        validation_value_data: numpy.ndarray | pandas.Series,
        methods: list | None = None,
        metric: Callable = mean_squared_error,
        comparison_method: Callable = min,
    ):
        """Initialization method for class.

        Args:
            training_time_data (numpy.ndarray | pandas.Series):
                Series of times to be used in
                fitting forecasting methods.
            training_value_data (numpy.ndarray | pandas.Series):
                Series of data values to be used in
                fitting forecasting methods.
            validation_time_data (numpy.ndarray | pandas.Series):
                Series of times to be used in
                validating forecasting methods.
            validation_value_data (numpy.ndarray | pandas.Series):
                Series of data values to be used in
                validating forecasting methods.
            methods (List[Class] | None): List of
                classes of forecasting methods. If
                None, uses a pre-defined list of
                classes. Defaults to None.
            metric (Callable): Metric to use to
                determine best forecast method.
                Defaults to 'mean_squared_error'.
            comparison_method (Callable): Method to
                compare calculated metric values,
                to determine the best value.
                Defaults to 'min'.
        """
        self.training_time_data = training_time_data
        if isinstance(self.training_time_data, pandas.Series):
            self.training_time_data = self.training_time_data.to_numpy()
        self.training_value_data = training_value_data
        if isinstance(self.training_value_data, pandas.Series):
            self.training_value_data = self.training_value_data.to_numpy()
        self.training_data = pandas.Series(
            self.training_value_data,
            self.training_time_data,
        )
        self.validation_time_data = validation_time_data
        if isinstance(self.validation_time_data, pandas.Series):
            self.validation_time_data = self.validation_time_data.to_numpy()
        self.validation_value_data = validation_value_data
        if isinstance(self.validation_value_data, pandas.Series):
            self.validation_value_data = self.validation_value_data.to_numpy()
        self.validation_data = pandas.Series(
            self.validation_value_data,
            self.validation_time_data,
        )
        self.methods = (
            methods
            if methods
            else [
                SingleExponentialSmoothing,
                TripleExponentialSmoothing,
            ]
        )
        self.forecast_methods: dict[str, dict] = {}
        self.metric_values = [None for _ in self.methods]
        self.metric = metric
        self.comparison_method = comparison_method
        self.best_forecast_method = None
        self.best_forecast_fit = None
        # Figure out how many forecasting data
        # points should be included to compare with
        # validation data.
        # First, determine if the time deltas between
        # data points areidentical.
        training_time_deltas = (
            self.training_time_data[1:] - self.training_time_data[:-1]
        )
        validation_time_deltas = (
            self.validation_time_data[1:] - self.validation_time_data[:-1]
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
                    self.training_time_data[-1] + training_time_deltas[0]
                )
                # There is probably a better/more pythonic/more efficient way
                # to do this, but it's not coming to me at the moment.
                while (  # pylint: disable=while-used
                    self.validation_time_data[-1] > current_time
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

    def fit(self):
        """Method to find the best forecast method
        of the given options.
        """
        for ind, method in enumerate(self.methods):
            method_class = method(
                self.training_time_data,
                self.training_value_data,
            )
            fit = method_class.fit()  # I assume the likelihood estimate is
            # better than what a grid search would find?
            forecast = fit.forecast(self.forecast_length)
            metric = self.metric(
                self.validation_data,
                forecast,
            )
            self.forecast_methods[method_class.forecasting_method] = {
                "class": method_class,
                "fit": fit,
                "params": fit.model.params,
                self.metric.__name__: metric,
                "forecast": forecast,
            }
            self.metric_values[ind] = metric
        self.best_forecast_method = self.methods[
            self.metric_values.index(
                self.comparison_method(self.metric_values)
            )
        ](
            numpy.concatenate(
                (
                    self.training_time_data,
                    self.validation_time_data,
                ),
            ),
            numpy.concatenate(
                (
                    self.training_value_data,
                    self.validation_value_data,
                ),
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
