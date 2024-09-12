"""This file contains a wrapper class for the statistical
forecasting method Single Exponential Smoothing."""

from statsmodels.tsa.api import SimpleExpSmoothing

from .statsmodels_wrapper import StatsModelsWrapper

# pylint: disable=abstract-method,duplicate-code


class SingleExponentialSmoothing(StatsModelsWrapper):
    """Wrapper class for SimpleExpSmoothing model from statsmodels."""

    def _model_class(self):
        """Return the SimpleExpSmoothing class."""
        return SimpleExpSmoothing

    # def _instantiate_model(self, X, y=None):
    #     # Instantiate SimpleExpSmoothing with the given model_params.
    #     return SimpleExpSmoothing(X, **self.model_params)


# Attempt using the statsmodels wrapper from SO:
# class SingleExponentialSmoothing():
#     """Wrapper class to hold Single Exponential
#     Smoothing forecasting method estimator."""
#     def __init__(self, *args, **kwargs):
#         estimator = StatsmodelsWrapper(
#             SimpleExpSmoothing(*args, **kwargs)
#         )
#         return estimator(*args, **kwargs)


# Initial attempt, without realizing the need for GridSearchCV.
# class SingleExponentialSmoothing(
#     SimpleExpSmoothing,
# ):
#     """Wrapper class to hold Single Exponential
#     Smoothing forecasting methods and information."""

#     def __init__(self, time_data: numpy.ndarray, value_data: numpy.ndarray):
#         """Initialization method for class.

#         Args:
#             time_data (numpy.ndarray):
#                 Series of times to be used in
#                 fitting Single Exponential
#                 Smoothing forecasting method.
#             value_data (numpy.ndarray):
#                 Series of data values to be used in
#                 fitting Single Exponential
#                 Smoothing forecasting method.
#         """
#         # Really should check for data types.
#         self.time_data = time_data
#         self.value_data = value_data
#         self.data_series = pandas.Series(
#             self.value_data,
#             self.time_data,
#         )
#         super().__init__(self.data_series, initialization_method="estimated")
#         self.forecasting_method = "Single Exponential Smoothing"
