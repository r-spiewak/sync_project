"""This file contains classes useful for forecasting and
fitting the best forecasting method."""

import warnings
from collections.abc import Callable
from types import NoneType

import matplotlib.pyplot
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
        methods: dict | None = None,
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
            methods (dict | None): Dictionary of
                classes and parameters of forecasting
                methods. If None, uses a pre-defined
                dictionary. Defaults to None.
            metric (Callable): Metric to use to
                determine best forecast method.
                Defaults to 'mean_squared_error'.
            comparison_method (Callable): Method to
                compare calculated metric values,
                to determine the best value.
                Defaults to 'min'.
        """
        # Add type validation.
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
            else {
                "Single Exponential Smoothing": {
                    "class": SingleExponentialSmoothing,
                    "param_grid": {
                        2: "d",
                    },
                },
                "Triple Exponential Smoothing": {
                    "class": TripleExponentialSmoothing,
                    "param_grid": {
                        2: "c",
                    },
                },
            }
        )
        # self.forecast_methods: dict[str, dict] = {}
        self.metric_values = [None for _ in self.methods]
        self.metric = metric
        self.comparison_method = comparison_method
        self.best_forecast_method = None
        self.best_forecast_fit_result = None
        self.best_forecast_fit = None
        self.best_forecast_forecast = None
        # Figure out how many forecasting data
        # points should be included to compare with
        # validation data.
        # First, determine if the time deltas between
        # data points are identical.
        # self.uniform_time_deltas = False
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
                self.forecast_frequency = training_time_deltas[0]
                self.forecast_times = self.validation_time_data.copy()
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
                self.forecast_times = [current_time]
                # There is probably a better/more pythonic/more efficient way
                # to do this, but it's not coming to me at the moment.
                while (  # pylint: disable=while-used
                    self.validation_time_data[-1] > current_time
                ):
                    self.forecast_length += 1
                    current_time += training_time_deltas[0]
                    self.forecast_times.append(current_time)
                self.forecast_frequency = training_time_deltas[0]
            # self.forecast_times = numpy.array(
            #     [
            #         self.validation_time_data[0] + i*self.forecast_frequency
            #         for i in range(self.forecast_length)
            #     ]
            # )
        else:
            # If the training time deltas are not
            # uniform, one idea is to set the forecast
            # length to that of the validation data
            # (based on the (end-start)/freq, and the
            # forecast frequency to that of the average
            # training time delta).
            # I suspect this will cause issues when
            # calculating metrics. (It will, since the
            # lengths of the arrays won't be identical.)
            # Two options: Either figure out a way to
            # calculate the metrics incorporating the
            # time index info as well, or just use the
            # times and length of the validation set.
            # Since the second option is more
            # straightforward, we will use that here
            # (at least for now).
            self.forecast_frequency = numpy.mean(training_time_deltas)
            # self.forecast_length = int(
            #     (
            #         self.validation_time_data[-1]
            #         - self.validation_time_data[0]
            #     ) / self.forecast_frequency
            # )
            self.forecast_length = len(self.validation_time_data)
            # This is especially important in this case, since
            # the forecast itself will not have attached time
            # data and this needs to be estimated, since the time
            # data from the training_data does not have a uniform
            # frequency.
            self.forecast_times = self.validation_time_data.copy()
        # For the final forecast with the best fit model:
        self.best_forecast_times = None

    def fit(self):
        """Method to find the best forecast method
        of the given options.
        """
        for ind, method in enumerate(self.methods.keys()):
            method_class = self.methods[method]["class"](
                self.training_time_data,
                self.training_value_data,
            )
            fit = method_class.fit()  # I assume the likelihood estimate is
            # better than what a grid search would find?
            forecast = fit.forecast(self.forecast_length)
            # An alternative to the above method is below.
            # However, note that in each case, if there is no
            # regular frequency to the index (i.e., if the time
            # deltas are not uniform), it cannot actually
            # generate values at specific timestamps, so
            # an alternative way of estimating timestamps
            # must be used. This means the "length" of the
            # 'forecast' above does not directly correspond to
            # timestamps, and the 'predict' below cannot actually
            # be used at all (since th start and end times won't
            # correspond to actual values interpolated from the data).
            # forecast = fit.predict(
            #     start=self.forecast_times[0],
            #     end=self.forecast_times[-1],
            # )
            metric = self.metric(
                self.validation_data,
                forecast,
            )
            self.methods[method].update(
                {
                    "result_object": fit,
                    "fit": fit.fittedvalues,
                    "params": fit.model.params,
                    self.metric.__name__: metric,
                    "forecast": forecast,
                }
            )
            self.metric_values[ind] = metric
        # self.best_forecast_method = self.methods[
        #     self.metric_values.index(
        #         self.comparison_method(self.metric_values)
        #     )
        self.best_forecast_method = self.methods[
            self.comparison_method(
                self.methods,
                key=lambda v: self.methods[v][self.metric.__name__],
            )
        ]["class"](
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
        self.best_forecast_fit_result = self.best_forecast_method.fit()
        # Get forecast model params with:
        # self.best_forecast_fit_result.model.params
        self.best_forecast_fit = self.best_forecast_fit_result.fittedvalues
        return self

    def forecast(self, forecast_length: int) -> pandas.Series:
        """Calls the forecast method of the
        previously fit best_forecast_method.

        Args:
            forecast_length (int): Number of
                forecasts to predict.

        Returns:
            pandas.Series: Series of forecasts.

        Raises:
            RuntimeError: If the 'fit' method has not been called.
        """
        if not self.best_forecast_method:
            raise RuntimeError(
                "The 'forecast' method can only be"
                " called after the 'fit' method."
            )
        # Maybe I should change this to predict also, so as to get the time values too?
        self.best_forecast_forecast = self.best_forecast_fit_result.forecast(
            forecast_length,
        )
        # Detect whether the forecast comes with a
        # compatible index. If not, construct one
        # like those constructed in the init method:
        if (
            self.best_forecast_forecast.index.to_numpy().dtype
            == self.training_time_data.dtype
        ):
            self.best_forecast_times = (
                self.best_forecast_forecast.index.to_numpy()
            )
        else:
            self.best_forecast_times = numpy.array(
                [
                    self.validation_time_data[-1] + i * self.forecast_frequency
                    for i in range(forecast_length)
                ]
            )
        return self

    def plot_all(self):
        """Plots all the fits and forecasts."""
        matplotlib.pyplot.plot(
            self.training_time_data,
            self.training_value_data,
            "ko",
            label="data",
        )
        matplotlib.pyplot.plot(
            self.validation_time_data, self.validation_value_data, "ko"
        )
        colors_list = ("b", "g", "r", "c", "m", "y")
        linestyles_list = ("-", "-.")
        linestyles_list_pred = ("--", ":")
        if (
            "fit"  # pylint: disable=magic-value-comparison
            not in self.methods[
                list(
                    self.methods.keys()  # pylint: disable=consider-iterating-dictionary
                )[0]
            ]
        ):
            warnings.warn(
                UserWarning(
                    "'fit' method has not yet been called; "
                    "only input data will be included on the plot."
                )
            )
        for ind, (method, vals) in enumerate(self.methods.items()):
            if (
                "fit"  # pylint: disable=magic-value-comparison
                not in vals.keys()
            ):
                continue
            color = colors_list[int(ind % len(colors_list))]
            linestyle_ind = int(
                (ind / len(colors_list)) % len(linestyles_list)
            )
            linestyle = linestyles_list[linestyle_ind]
            linestyle_pred = linestyles_list_pred[linestyle_ind]
            matplotlib.pyplot.plot(
                vals["fit"],
                color=color,
                linestyle=linestyle,
                label=method,
            )
            matplotlib.pyplot.plot(
                self.forecast_times,
                vals["forecast"],
                color=color,
                linestyle=linestyle_pred,
            )
        matplotlib.pyplot.legend()
        matplotlib.pyplot.show()

    def plot_best(self):
        """Plots all the fits and forecasts.

        Raises:
            RuntimeError: If the 'forecast' method has not
                already been called.
        """
        if isinstance(self.best_forecast_forecast, NoneType):
            raise RuntimeError(
                "The 'plot_best' method can only be"
                " called after the 'forecast' method."
            )
        matplotlib.pyplot.plot(
            self.training_time_data,
            self.training_value_data,
            "ko",
            label="data",
        )
        matplotlib.pyplot.plot(
            self.validation_time_data, self.validation_value_data, "ko"
        )
        matplotlib.pyplot.plot(
            self.best_forecast_fit,
            "b-",
            label=self.best_forecast_method,
        )
        matplotlib.pyplot.plot(
            self.best_forecast_times,
            self.best_forecast_forecast,
            "b-",
        )
        matplotlib.pyplot.legend()
        matplotlib.pyplot.show()
