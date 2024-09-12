"""This file holds the gaussian function and its
class wrapper, for use in sklearn's GridSearchCV."""

import numpy
import pandas

from .curve_fit_wrapper import CurveFitWrapper


# Gaussian model function:
def gaussian(
    x: numpy.ndarray | pandas.Series, a: float, b: float, c: float
) -> numpy.ndarray | pandas.Series:
    """Function to return the gaussian of x.

    Args:
        x (numpy.ndarray | pandas.Series):
            Independent variable.
        a (float): Height of the curve's peak.
        b (float): Position of center of peak.
        c (float): Standard deviation, width of "bell".

    Returns:
        numpy.ndarray | pandas.Series:
            Result from taking the gaussian.
    """
    return a * numpy.exp(-((x - b) ** 2) / (2 * c**2))


class Gaussian(CurveFitWrapper):
    """Wrapper function for curve_fit with gaussian
    function, for use in GridSearchCV."""

    def __init__(self, **fit_params):
        """Initialize the CurveFitWrapper with
        the gaussian function."""
        # Need 'bounds'?
        super().__init__(gaussian, **fit_params)
