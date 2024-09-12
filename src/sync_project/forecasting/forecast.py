"""This file contains classes useful for forecasting and
fitting the best forecasting method."""

import warnings
from collections.abc import Callable
from types import NoneType

import matplotlib.pyplot
import numpy
import pandas
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

from sync_project.constants import TimePointLabels
from sync_project.data_operations.data_labeling import timeseries_to_labels
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
        n_splits: int = 5,
        verbose: int = 0,
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
            n_splits (int): Number of splits to be made
                in training data for the cross-validation.
                Defaults to 5.
            verbose (int): Level of verbosity in output
                of GridSearchCV calls. Higher values
                yield more output. Defaults to 0.
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
                        "smoothing_level": numpy.arange(0, 1, 0.1),
                    },
                    "past_points": [0],
                },
                "Triple Exponential Smoothing": {
                    "class": TripleExponentialSmoothing,
                    "param_grid": {
                        "trend": [None, "add", "mul"],
                        "seasonal": [None, "add", "mul"],
                        "seasonal_periods": [2, 3, 6, 12, 24],
                        "smoothing_level": numpy.arange(0, 1, 0.1),
                        "smoothing_trend": numpy.arange(
                            0,
                            1,
                            0.1,
                        ),
                        "smoothing_seasonal": numpy.arange(0, 1, 0.1),
                    },
                    "past_points": [0],
                },
            }
        )
        # self.forecast_methods: dict[str, dict] = {}
        self.metric_values = [None for _ in self.methods]
        self.metric = metric
        self.comparison_method = comparison_method
        self.best_forecast_method = None
        self.best_method_name = None
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
        self.full_training_data = None
        self.n_splits = n_splits
        self.verbose = verbose

    def fit(self):  # pylint: disable=too-many-locals
        # Clearly this monstrosity of a method needs to
        # be broken down into smaller sub-methods...
        """Method to find the best forecast method
        of the given options.
        """
        for ind, method in enumerate(self.methods.keys()):
            # method_class = self.methods[method]["class"](
            #     self.training_time_data,
            #     self.training_value_data,
            # )
            # forecast = fit.forecast(self.forecast_length)
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
            method_class = self.methods[method]["class"]()
            param_grid = self.methods[method]["param_grid"]
            past_points_scores = []
            past_points_preds = []
            past_points_models = []
            for past_points in self.methods[method]["past_points"]:
                this_training_data = timeseries_to_labels(
                    self.training_data,
                    p=past_points,
                    n=1,
                ).dropna()
                train_cols = [
                    col
                    for col in this_training_data.columns
                    if (
                        TimePointLabels.PAST.value in col
                        or TimePointLabels.PRESENT.value in col
                    )
                ]
                pred_col = [
                    col
                    for col in this_training_data.columns
                    if TimePointLabels.FUTURE.value in col
                ]
                this_val_data = timeseries_to_labels(
                    self.validation_data,
                    p=past_points,
                    n=1,
                ).dropna()
                # I could call the following method once outside
                # the loop, if I store the results in a list (since
                # otherwise the generator will not loop results again).
                cv = TimeSeriesSplit(
                    n_splits=self.n_splits,
                ).split(this_training_data)
                grid_search = GridSearchCV(
                    method_class,
                    param_grid,
                    scoring=self.metric,
                    cv=cv,
                    verbose=self.verbose,
                )
                # We need to supply training data to fit the model.
                # We supply it in the form of features for each
                # For ML methods, we must supply the full X df of features
                # (for forecasting with timeseries, we can supply
                # features of the previous n values). We also need to
                # supply y labels, which can be the next p timesteps'
                # values.
                # For statsmodels, we supply a single feature.
                # The wrapper can probably take care of the splitting
                # of the df to just use the one column of the current
                # timestep.
                # fit = method_class.fit(self.training_value_data)
                grid_search.fit(
                    this_training_data[train_cols],
                    this_training_data[pred_col],
                )
                # To use this with the wrapper class and with sklearn estimators,
                # we need to supply a features DataFrame.
                # The conditions of which df to apply are the same as above.
                # For statsmodels, we can supply times (or, really, anything
                # that has the length of the number of predictions we want).
                # For the ML models, we must supply features, as previously.
                # forecast = method_class.predict(X=self.validation_time_data)
                preds = grid_search.predict(X=this_val_data[train_cols])
                past_points_preds.append(preds)
                past_points_scores.append(
                    self.metric(preds, this_val_data[pred_col])
                )
                past_points_models.append(grid_search)
            best_past_points_ind = past_points_scores.index(
                self.comparison_method(past_points_scores)
            )
            best_past_points = self.methods[method]["past_points"][
                best_past_points_ind
            ]
            fit = past_points_models[best_past_points_ind]
            forecast = past_points_preds[best_past_points_ind]
            metric = self.metric(
                self.validation_data[:-1],
                forecast,
            )
            self.methods[method].update(
                {
                    "past_points_fits": {
                        "scores": past_points_scores,
                        "models": past_points_models,
                        "predictions": past_points_preds,
                    },
                    "best_past_points": best_past_points,
                    "result_object": fit,
                    # These two only make sense when fit
                    # itself is a statsmodels object.
                    # "fit": fit.fittedvalues,
                    # "params": fit.model.params,
                    "params": fit.best_params_,
                    self.metric.__name__: metric,
                    "forecast": forecast,
                    "forecast_times": self.validation_time_data[:-1],
                }
            )
            self.metric_values[ind] = metric
        # self.best_forecast_method = self.methods[
        #     self.metric_values.index(
        #         self.comparison_method(self.metric_values)
        #     )
        self.best_method_name = self.comparison_method(
            self.methods,
            key=lambda v: self.methods[v][self.metric.__name__],
        )
        # Retrain best model on training and validation data:
        # Combine datasets and use best_past_points:
        full_training_data = timeseries_to_labels(
            pandas.concat(
                [
                    self.training_data,
                    self.validation_data,
                ]
            ),
            p=self.methods[self.best_method_name]["best_past_points"],
            n=1,
        ).dropna()
        train_cols = [
            col
            for col in full_training_data.columns
            if (
                TimePointLabels.PAST.value in col
                or TimePointLabels.PRESENT.value in col
            )
        ]
        pred_col = [
            col
            for col in full_training_data.columns
            if TimePointLabels.FUTURE.value in col
        ]
        self.best_forecast_method = self.methods[self.best_method_name][
            "class"
        ]()
        #     numpy.concatenate(
        #         (
        #             self.training_time_data,
        #             self.validation_time_data,
        #         ),
        #     ),
        #     numpy.concatenate(
        #         (
        #             self.training_value_data,
        #             self.validation_value_data,
        #         ),
        #     ),
        # )
        assert self.best_forecast_method is not None
        self.best_forecast_fit_result = self.best_forecast_method.fit(
            full_training_data[train_cols], full_training_data[pred_col]
        )
        self.full_training_data = full_training_data[train_cols]
        # These next lines only apply to statsmodels objects.
        # Get forecast model params with:
        # self.best_forecast_fit_result.model.params
        # self.best_forecast_fit = self.best_forecast_fit_result.fittedvalues
        return self

    def forecast(
        self,
        forecast_length: int,
    ) -> pandas.Series | pandas.DataFrame:
        """Calls the forecast method of the
        previously fit best_forecast_method.

        Args:
            forecast_length (int): Number of future predictions
                to forecast.

                Features for forecast predictions. There will be
                one prediction for each set of features
                (i.e., rows in the Series or DataFrame).

        Returns:
            pandas.Series | pandas.DataFrame: Series
                or DataFrame of forecasts.

        Raises:
            RuntimeError: If the 'fit' method has not been called.
        """
        if not self.best_forecast_method:
            raise RuntimeError(
                "The 'forecast' method can only be"
                " called after the 'fit' method."
            )
        # Maybe I should change this to predict also, so as to get the time values too?
        # self.best_forecast_forecast = self.best_forecast_fit_result.forecast(
        #     forecast_length,
        # )
        # forecast_features = self.full_training_data.iloc[-1]
        # Here it's important to have no future points
        # in the DataFrame so that we don't fill in the
        # last row with NaN and drop it.
        forecast_features = (
            timeseries_to_labels(
                pandas.concat(
                    [
                        self.training_data,
                        self.validation_data,
                    ]
                ),
                p=self.methods[self.best_method_name]["best_past_points"],
                n=0,
            )
            .dropna()
            .tail(1)
        )
        feature_cols = [
            col
            for col in forecast_features.columns
            if (
                TimePointLabels.PAST.value in col
                or TimePointLabels.PRESENT.value in col
            )
        ]
        current_cols = [
            col
            for col in forecast_features.columns
            if TimePointLabels.PRESENT.value in col
        ]
        forecast = []
        for _ in range(forecast_length):
            forecast.append(
                self.best_forecast_fit_result.predict(
                    X=forecast_features[feature_cols]
                )
            )
            forecast_features = forecast_features.shift(
                -len(forecast[-1]), axis=1
            )
            forecast_features[current_cols] = forecast[-1]
        self.best_forecast_forecast = forecast
        # Detect whether the forecast comes with a
        # compatible index. If not, construct one
        # like those constructed in the init method:
        # if (
        #     self.best_forecast_forecast.index.to_numpy().dtype
        #     == self.training_time_data.dtype
        # ):
        #     self.best_forecast_times = (
        #         self.best_forecast_forecast.index.to_numpy()
        #     )
        # else:
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
        # if (
        #     "fit"  # pylint: disable=magic-value-comparison
        #     not in self.methods[
        #         list(
        #             self.methods.keys()  # pylint: disable=consider-iterating-dictionary
        #         )[0]
        #     ]
        # ):
        if not self.best_forecast_method:
            warnings.warn(
                UserWarning(
                    "'fit' method has not yet been called; "
                    "only input data will be included on the plot."
                )
            )
            return
        for ind, (method, vals) in enumerate(self.methods.items()):
            color = colors_list[int(ind % len(colors_list))]
            linestyle_ind = int(
                (ind / len(colors_list)) % len(linestyles_list)
            )
            linestyle = linestyles_list[linestyle_ind]
            linestyle_pred = linestyles_list_pred[linestyle_ind]
            if (
                "fit"  # pylint: disable=magic-value-comparison
                not in vals.keys()
            ):
                matplotlib.pyplot.plot(
                    vals["forecast_times"],
                    vals["forecast"],
                    color=color,
                    linestyle=linestyle_pred,
                    label=method,
                )
            else:
                matplotlib.pyplot.plot(
                    vals["fit"],
                    color=color,
                    linestyle=linestyle,
                    label=method,
                )
                matplotlib.pyplot.plot(
                    vals["forecast_times"],
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
        # This only is meaningful for statsmodels methods:
        # matplotlib.pyplot.plot(
        #     self.best_forecast_fit,
        #     "b-",
        #     label=self.best_forecast_method,
        # )
        matplotlib.pyplot.plot(
            self.best_forecast_times,
            self.best_forecast_forecast,
            "b--",
            label=self.best_forecast_method,
        )
        matplotlib.pyplot.legend()
        matplotlib.pyplot.show()
