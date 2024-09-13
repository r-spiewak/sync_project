"""This filecontains a wrapper class for the statistical
forecasting method Markov Switching Regression."""

from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

from .statsmodels_wrapper import StatsModelsWrapper


class MarkovSwitchingRegression(StatsModelsWrapper):
    """Wrapper class for MarkovRegression from statsmodels."""

    def _model_class(self):
        """Return the MarkovRegression class."""
        return MarkovRegression
