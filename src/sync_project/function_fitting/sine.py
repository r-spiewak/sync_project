"""This file holds the sine function and its
class wrapper, for use in sklearn's GridSearchCV."""

import numpy
import pandas

from .curve_fit_wrapper import CurveFitWrapper


# Sine model function:
def sine(
    x: float | numpy.ndarray | pandas.Series,
    a: float,
    b: float,
    c: float,
    d: float,
) -> float | numpy.ndarray | pandas.Series:
    """Function to return the sine of x.

    Args:
        x (float | numpy.ndarray | pandas.Series):
            Independent variable.
        a (float): Amplitude.
        b (float): Frequency.
        c (float): Phase shift.
        d (float): y-intercept.

    Returns:
        float | numpy.ndarray | pandas.Series:
            Result from taking the sine.
    """
    return a * numpy.sin(b * x + c) + d


class Exponential(CurveFitWrapper):
    """Wrapper class for fitting an Sine curve using curve_fit."""

    def __init__(self, **fit_params):
        """Initialize the CurveFitWrapper with
        the sine function."""
        # Need 'bounds'?
        super().__init__(sine, **fit_params)
