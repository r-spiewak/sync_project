"""This filecontains a wrapper class for the statistical
forecasting method ARIMA."""

from statsmodels.tsa.arima.model import ARIMA as arima

from .statsmodels_wrapper import StatsModelsWrapper


class ARIMA(StatsModelsWrapper):
    """Wrapper class for ARIMA model from statsmodels to make it scikit-learn compatible."""

    def _model_class(self):
        """Return the ARIMA class."""
        return arima
