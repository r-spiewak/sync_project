"""This file holds the exponential function and its
class wrapper, for use in sklearn's GridSearchCV."""

import numpy
import pandas

from .curve_fit_wrapper import CurveFitWrapper


# Exponential model function:
def exponential(
    x: float | numpy.ndarray | pandas.Series, a: float, b: float, c: float
) -> float | numpy.ndarray | pandas.Series:
    """Function to return the exponential of x.

    Args:
        x (float | numpy.ndarray | pandas.Series):
            Independent variable.
        a (float): Rate of change of the function
            at each point.
        b (float): Change of base of exponent.
        c (float): y-intercept plus 1.

    Returns:
        float | numpy.ndarray | pandas.Series:
            Result from taking the gaussian.
    """
    return a * numpy.exp(b * x) + c


class Exponential(CurveFitWrapper):
    """Wrapper class for fitting an Exponential curve using curve_fit."""

    def __init__(self, **fit_params):
        """Initialize the CurveFitWrapper with
        the exponential function."""
        # Need 'bounds'?
        super().__init__(exponential, **fit_params)

    # def get_params(self, deep=True):
    #     return {'p0': self.p0, 'bounds': self.bounds, **self.fit_params}

    # def set_params(self, **params):
    #     for param, value in params.items():
    #         setattr(self, param, value)
    #     return self
