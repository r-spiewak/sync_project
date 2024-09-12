"""This file contains a wrapper class for the statistical
forecasting method Triple Exponential Smoothing."""

from statsmodels.tsa.api import ExponentialSmoothing

from .statsmodels_wrapper import StatsModelsWrapper

# pylint: disable=abstract-method,duplicate-code


class TripleExponentialSmoothing(StatsModelsWrapper):
    """Wrapper class to hold Triple Exponential
    Smoothing forecasting method from statsmodels."""

    def _model_class(self):
        """Return the ExponentialSmoothing class."""
        return ExponentialSmoothing

    # def _instantiate_model(self, X, y=None):
    #     # Instantiate ExponentialSmoothing with the given model_params.
    #     return ExponentialSmoothing(X, **self.model_params)


# class TripleExponentialSmoothing(
#     ExponentialSmoothing,
# ):
#     """Wrapper class to hold Triple Exponential
#     Smoothing forecasting methods and information."""

#     def __init__(self, time_data: numpy.ndarray, value_data: numpy.ndarray):
#         """Initialization method for class.

#         Args:
#             time_data (numpy.ndarray):
#                 Series of times to be used in
#                 fitting Triple Exponential
#                 Smoothing forecasting method.
#             value_data (numpy.ndarray):
#                 Series of data values to be used in
#                 fitting Triple Exponential
#                 Smoothing forecasting method.
#         """
#         # Really should check for data types here.
#         self.time_data = time_data
#         self.value_data = value_data
#         self.data_series = pandas.Series(
#             self.value_data,
#             self.time_data,
#         )
#         super().__init__(self.data_series, initialization_method="estimated")
#         self.forecasting_method = "Triple Exponential Smoothing"
