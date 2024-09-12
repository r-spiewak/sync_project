"""This filecontains a wrapper class for the statistical
forecasting method ARIMA."""

from statsmodels.tsa.arima.model import ARIMA as arima

from .statsmodels_wrapper import StatsModelsWrapper


class ARIMA(StatsModelsWrapper):
    """Wrapper class for ARIMA model from statsmodels to make it scikit-learn compatible."""

    # def __init__(self, order=(1, 0, 0), **model_params):
    #     """Instantiate class."""
    #     self.order = order
    #     super().__init__(**model_params)

    def _model_class(self):
        """Return the ARIMA class."""
        return arima

    # def _instantiate_model(self, X, y=None):
    #     """Instantiate the ARIMA model with the specified order and model_params.
    #     """
    #     return arima(X, **self.model_init_params)
