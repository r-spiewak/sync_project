"""This file contains a wrapper class for the statistical
forecasting method Triple Exponential Smoothing."""

import pandas
from statsmodels.tsa.api import ExponentialSmoothing

# pylint: disable=abstract-method,duplicate-code


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
        # Really should check for data types here.
        self.time_data = time_data
        self.value_data = value_data
        self.data_series = pandas.Series(
            self.value_data.to_numpy(),
            self.time_data.to_numpy(),
        )
        super().__init__(self.data_series, initialization_method="estimated")
        self.forecasting_method = "Triple Exponential Smoothing"
