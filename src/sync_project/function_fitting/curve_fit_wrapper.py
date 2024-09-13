"""This file contains a base wrapper class for scipy's
curve_fit function, to allow use in sklearn's GridSearchCV."""

import inspect
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas
from scipy.optimize import curve_fit
from sklearn.base import BaseEstimator, RegressorMixin


class CurveFitWrapper(BaseEstimator, RegressorMixin):
    """A base wrapper for scipy.optimize.curve_fit to make it compatible with scikit-learn."""

    def __init__(
        self,
        func: Callable,
        **fit_params: Any,
    ):
        """Initialize the class.

        Args:
            func (Callable): The model function,
                f(x, ...), that will be fit to the data.
            fit_params (Any):
                Additional keyword arguments passed to curve_fit.
        """
        self.func = func
        self.p0: list = []
        self.bounds = (-np.inf, np.inf)
        self.fit_params = fit_params
        self.opt_params_: list = []
        # To store optimized parameters after fitting.

        # Get the parameter names from the function signature:
        # Skip 'x' as it’s the independent variable.
        self.param_names = list(
            inspect.signature(self.func).parameters.keys()
        )[1:]

    def fit(
        self,
        X: np.ndarray | pandas.Series,
        y: np.ndarray | pandas.Series | None = None,
    ):
        """Fit the curve to the data using scipy.optimize.curve_fit.

        Args:
            X (np.ndarray | pandas.Series): (array-like)
                The independent variable where the data is measured.
            y (np.ndarray | pandas.Series | None): (array-like)
                The dependent data, a 1D array of observed values.

        Returns:
            self (object): Fitted estimator.
        """
        # Pull out the correct 'feature'(s) to use.
        X = np.asarray(X).flatten()
        y = np.asarray(y).flatten()

        # Extract the initial guesses (p0) for each parameter from self.fit_params
        self.p0 = [
            self.fit_params.get(param_name, 1)
            for param_name in self.param_names
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

        # Use curve_fit to fit the model:
        self.opt_params_, _ = (  # pylint:disable=unbalanced-tuple-unpacking
            curve_fit(
                self.func,
                X,
                y,
                p0=self.p0,
                bounds=self.bounds,
                **curve_fit_params,
            )
        )
        # pylint: enable=duplicate-code
        return self

    def predict(
        self, X: np.ndarray | pandas.Series
    ) -> np.ndarray | pandas.Series:
        """Predict using the optimized parameters.

        Args:
            X (np.ndarray | pandas.Series): (array-like)
                The independent variable values to predict.

        Returns:
            predictions (np.ndarray | pandas.Series):
                (array-like) The predicted values.
        """
        if self.opt_params_ is None:
            raise ValueError(
                "This CurveFitWrapper instance is not fitted yet."
                " Call 'fit' with appropriate arguments first."
            )

        return self.func(X, *self.opt_params_)

    def get_params(self, deep=True):
        """Get parameters for the model.
        Required for GridSearchCV.
        This dynamically fetches parameters based on
        the underlying function signature.

        Returns:
            params (dict):
                Parameter names mapped to their values.
        """

        params = {
            param_name: self.fit_params.get(param_name, 1)
            for param_name in self.param_names
        }
        return params

    def set_params(self, **params):
        """Set parameters for the model.
        Required for GridSearchCV.

        Args:
            params (dict):
                Parameter names mapped to their values.

        Returns:
            self (object)
        """
        for param, value in params.items():
            if param in self.param_names:
                self.fit_params[param] = value

        return self
