"""This file holds the logistic function and its
class wrapper, for use in sklearn's GridSearchCV."""

import numpy
import pandas

from .curve_fit_wrapper import CurveFitWrapper


# Logistic model function:
def logistic(
    x: float | numpy.ndarray | pandas.Series,
    a: float,
    b: float,
    c: float,
) -> float | numpy.ndarray | pandas.Series:
    """Function to return the sine of x.

    Args:
        x (float | numpy.ndarray | pandas.Series):
            Independent variable.
        a (float): Carrying capacity.
        b (float): logistic growth rate.
        c (float): x-value at funtion's midpoint.

    Returns:
        float | numpy.ndarray | pandas.Series:
            Result from taking the logistic.
    """
    return a / (1 + numpy.exp(-b * (x - c)))


class Logistic(CurveFitWrapper):
    """Wrapper class for fitting an Logistic curve using curve_fit."""

    def __init__(self, **fit_params):
        """Initialize the CurveFitWrapper with
        the logistic function."""
        # Need 'bounds'?
        super().__init__(logistic, **fit_params)
