"""This filecontains a wrapper class for the statistical
forecasting method ARIMA."""

from statsmodels.tsa.statespace.sarimax import SARIMAX as sarimax

from .statsmodels_wrapper import StatsModelsWrapper


class SARIMAX(StatsModelsWrapper):
    """Wrapper class for SARIMAX model from statsmodels."""

    def _model_class(self):
        """Return the SARIMAX class."""
        return sarimax
