"""This filecontains a wrapper class for the statistical
forecasting method Markov Switching Regression."""

from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

from .statsmodels_wrapper import StatsModelsWrapper


class MarkovSwitchingRegression(StatsModelsWrapper):
    """Wrapper class for MarkovRegression from statsmodels."""

    def _model_class(self):
        """Return the MarkovRegression class."""
        return MarkovRegression

    # def _instantiate_model(self, X, y=None):
    #     """Instantiate the MarkovRegression model with the parameters
    #     passed in the constructor.
    #     """
    #     return MarkovRegression(X, **self.model_init_params)
