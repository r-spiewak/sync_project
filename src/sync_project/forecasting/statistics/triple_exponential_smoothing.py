"""This file contains a wrapper class for the statistical
forecasting method Triple Exponential Smoothing."""

import numpy
import pandas
from statsmodels.tsa.api import ExponentialSmoothing

# pylint: disable=abstract-method,duplicate-code


class TripleExponentialSmoothing(
    ExponentialSmoothing,
):
    """Wrapper class to hold Triple Exponential
    Smoothing forecasting methods and information."""

    def __init__(self, time_data: numpy.ndarray, value_data: numpy.ndarray):
        """Initialization method for class.

        Args:
            time_data (numpy.ndarray):
                Series of times to be used in
                fitting Triple Exponential
                Smoothing forecasting method.
            value_data (numpy.ndarray):
                Series of data values to be used in
                fitting Triple Exponential
                Smoothing forecasting method.
        """
        # Really should check for data types here.
        self.time_data = time_data
        self.value_data = value_data
        self.data_series = pandas.Series(
            self.value_data,
            self.time_data,
        )
        super().__init__(self.data_series, initialization_method="estimated")
        self.forecasting_method = "Triple Exponential Smoothing"
