"""This file contains a wrapper class for the statistical
forecasting method Triple Exponential Smoothing."""

from statsmodels.tsa.api import ExponentialSmoothing

from .statsmodels_wrapper import StatsModelsWrapper


class TripleExponentialSmoothing(StatsModelsWrapper):
    """Wrapper class to hold Triple Exponential
    Smoothing forecasting method from statsmodels."""

    def _model_class(self):
        """Return the ExponentialSmoothing class."""
        return ExponentialSmoothing
