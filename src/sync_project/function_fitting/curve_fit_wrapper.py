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
        # p0: np.ndarray | pandas.Series | None = None,
        # bounds: tuple[int | float, int | float] = (-np.inf, np.inf),
        **fit_params: Any,
    ):
        """Initialize the class.

        Args:
            func (Callable): The model function,
                f(x, ...), that will be fit to the data.
        #p0 (np.ndarray | pandas.Series | None, optional):
            Initial guess for the parameters.
        #bounds (tuple[int | float, int | float], optional):
            2-tuple of array-like. Lower and upper bounds
            on parameters.
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
        X = np.asarray(X).flatten()
        y = np.asarray(y).flatten()

        # Extract the initial guesses (p0) for each parameter from self.fit_params
        self.p0 = [
            self.fit_params.get(param_name, 1)
            for param_name in self.param_names
        ]

        # Use curve_fit to fit the model
        self.opt_params_, _, _, _, _ = curve_fit(
            self.func, X, y, p0=self.p0, bounds=self.bounds, **self.fit_params
        )
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
        params.update(
            {"p0": self.p0, "bounds": self.bounds, **self.fit_params}
        )
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
