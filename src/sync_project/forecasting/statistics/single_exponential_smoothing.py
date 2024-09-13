"""This file contains a wrapper class for the statistical
forecasting method Single Exponential Smoothing."""

from statsmodels.tsa.api import SimpleExpSmoothing

from .statsmodels_wrapper import StatsModelsWrapper


class SingleExponentialSmoothing(StatsModelsWrapper):
    """Wrapper class for SimpleExpSmoothing model from statsmodels."""

    def _model_class(self):
        """Return the SimpleExpSmoothing class."""
        return SimpleExpSmoothing
