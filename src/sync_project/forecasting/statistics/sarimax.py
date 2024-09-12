"""This filecontains a wrapper class for the statistical
forecasting method ARIMA."""

from statsmodels.tsa.statespace.sarimax import SARIMAX as sarimax

from .statsmodels_wrapper import StatsModelsWrapper


class SARIMAX(StatsModelsWrapper):
    """Wrapper class for SARIMAX model from statsmodels."""

    def _model_class(self):
        """Return the SARIMAX class."""
        return sarimax

    def _instantiate_model(self, X, y=None):
        """Instantiate the SARIMAX model with the parameters passed in the constructor.
        SARIMAX can take exogenous variables as well, so `y` could be exog data.
        """
        return sarimax(X, exog=y, **self.model_init_params)
