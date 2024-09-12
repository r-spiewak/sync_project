"""This filecontains a wrapper class for the statistical
forecasting method ARIMA."""

from statsmodels.tsa.statespace.sarimax import SARIMAX as sarimax

from .statsmodels_wrapper import StatsModelsWrapper

# There's an issue with using the predict function here,
# since this class supplies the y data as the exog
# variable which then must also be supplied as exog in
# the predict method. That is not done, at present.
# It works if I remove the instantiate method from here
# and just rely on the one in the base class though.


class SARIMAX(StatsModelsWrapper):
    """Wrapper class for SARIMAX model from statsmodels."""

    def _model_class(self):
        """Return the SARIMAX class."""
        return sarimax

    # def _instantiate_model(self, X, y=None):
    #     """Instantiate the SARIMAX model with the parameters passed in the constructor.
    #     SARIMAX can take exogenous variables as well, so `y` could be exog data.
    #     """
    #     return sarimax(X, exog=y, **self.model_init_params)
