"""This filecontains a wrapper class for the statistical
forecasting method Markov Switching Autoregression."""

from statsmodels.tsa.regime_switching.markov_autoregression import (
    MarkovAutoregression,
)

from .statsmodels_wrapper import StatsModelsWrapper


class MarkovSwitchingAutoregression(StatsModelsWrapper):
    """Wrapper class for MarkovAutoregression from statsmodels."""

    def _model_class(self):
        """Return the MarkovAutoregression class."""
        return MarkovAutoregression

    # def _instantiate_model(self, X, y=None):
    #     """Instantiate the MarkovAutoregression model with the parameters
    #     passed in the constructor.
    #     """
    #     return MarkovAutoregression(X, **self.model_init_params)
