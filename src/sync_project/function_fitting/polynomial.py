"""This file holds the polynomial function and its
class wrapper, for use in sklearn's GridSearchCV."""

from typing import Any

import numpy
import pandas
from scipy.optimize import curve_fit

from .curve_fit_wrapper import CurveFitWrapper


# Polynomial function:
def polynomial(
    x: numpy.ndarray | pandas.Series, *coeffs: list[float]
) -> numpy.ndarray | pandas.Series:
    """Generic polynomial function.

    Function form:
    $f(x) = a_nx^n + a_{n-1}x^{n-1}+...+a_1x+a_0$

    Args:
        x (float | numpy.ndarray | pandas.Series):
            Independent variable.
        coeffs (list[float]): Coefficients for the
            different polynomial terms.
            coeffs[0]=a_n, coeffs[-1]=a_0, etc.
            The degree of the polynomial is determined
            by len(coeffs).

    Returns:
        numpy.ndarray | pandas.Series:
            Polynomial evaluated at x.
    """
    return sum(c * x**i for i, c in enumerate(reversed(coeffs)))


class Polynomial(CurveFitWrapper):
    """Wrapper class for fitting an polynomial curve using curve_fit."""

    def __init__(
        self,
        **fit_params: Any,
    ):
        """Initialize class.

        Args:
            fit_params (Any):
                Initial guess for the coefficients or other curve_fit options.
        """
        degree = -1
        for (
            i
        ) in fit_params.keys():  # pylint:disable=consider-iterating-dictionary
            if "coef_" in i:  # pylint: disable=magic-value-comparison
                degree += 1
        self.degree = degree
        # Generate initial coefficients, defaulting to 1 if not provided in fit_params
        initial_params = {
            f"coef_{i}": fit_params.get(f"coef_{i}", 1)
            for i in range(degree + 1)
        }
        super().__init__(polynomial, **initial_params)

    def fit(
        self,
        X: float | numpy.ndarray | pandas.Series,
        y: float | numpy.ndarray | pandas.Series | None = None,
    ):
        """Fit the polynomial to the data.

        Args:
            X (float | numpy.ndarray | pandas.Series): (array-like)
                The independent variable where the data is measured.
            y (float | numpy.ndarray | pandas.Series | None): (array-like)
                The dependent data, a 1D array of observed values.
        """
        X = numpy.asarray(X).flatten()
        y = numpy.asarray(y).flatten()

        # Prepare initial guess for curve_fit
        p0 = [
            self.fit_params.get(f"coef_{i}", 1) for i in range(self.degree + 1)
        ]

        # pylint: disable=duplicate-code
        # Extract only valid curve_fit parameters (e.g., bounds, maxfev, etc.)
        valid_curve_fit_params = [
            "bounds",
            "method",
            "sigma",
            "absolute_sigma",
            "check_finite",
            "jac",
        ]
        curve_fit_params = {
            key: self.fit_params[key]
            for key in valid_curve_fit_params
            if key in self.fit_params
        }

        # Fit using curve_fit:
        self.opt_params_, _ = (  # pylint:disable=unbalanced-tuple-unpacking
            curve_fit(polynomial, X, y, p0=p0, **curve_fit_params)
        )
        # pylint: enable=duplicate-code

        return self

    def get_params(self, deep=True):
        """Get the current polynomial coefficients."""
        params = {
            f"coef_{i}": self.fit_params.get(f"coef_{i}", 1)
            for i in range(self.degree + 1)
        }
        return params

    def set_params(self, **params):
        """Set the coefficients for the polynomial."""
        for param, value in params.items():
            if param.startswith("coef_"):
                self.fit_params[param] = value
        return self

    def predict(
        self, X: float | numpy.ndarray | pandas.Series
    ) -> float | numpy.ndarray | pandas.Series:
        """Predict using the optimized polynomial coefficients.

        Args:
            X (float | numpy.ndarray | pandas.Series): (array-like)
                The independent variable values to predict.

        Returns:
            predictions (float | numpy.ndarray | pandas.Series): (array-like)
                The predicted values using the fitted polynomial.
        """
        if self.opt_params_ is None:
            raise ValueError(
                "This PolynomialWrapper instance is not"
                " fitted yet. Call 'fit' with appropriate arguments first."
            )

        return polynomial(X, *self.opt_params_)
