"""This file holds the tangent function and its
class wrapper, for use in sklearn's GridSearchCV."""

import numpy
import pandas

from .curve_fit_wrapper import CurveFitWrapper


# Tangent model function:
def tangent(
    x: float | numpy.ndarray | pandas.Series,
    a: float,
    b: float,
    c: float,
    d: float,
) -> float | numpy.ndarray | pandas.Series:
    """Function to return the tangent of x.

    Args:
        x (float | numpy.ndarray | pandas.Series):
            Independent variable.
        a (float): Amplitude.
        b (float): Frequency.
        c (float): Phase shift.
        d (float): y-intercept.

    Returns:
        float | numpy.ndarray | pandas.Series:
            Result from taking the tangent.
    """
    return a * numpy.tan(b * x + c) + d


class Tangent(CurveFitWrapper):
    """Wrapper class for fitting an Tangent curve using curve_fit."""

    def __init__(self, **fit_params):
        """Initialize the CurveFitWrapper with
        the tangent function."""
        # Need 'bounds'?
        super().__init__(tangent, **fit_params)
