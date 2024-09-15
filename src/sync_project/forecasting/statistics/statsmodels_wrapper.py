"""This file contains a wrapper class for using statsmodels
objects in the sklearn framework."""

import inspect

import numpy as np
import pandas
from sklearn.base import BaseEstimator, RegressorMixin

from sync_project.constants import TimePointLabels


class StatsModelsWrapper(BaseEstimator, RegressorMixin):
    """Generic wrapper class for statsmodels models to make them scikit-learn compatible.
    To use this class, inherit it in specific wrapper classes for different statsmodels models.
    This class automatically determines which parameters belong to the model's __init__ and fit
    methods.
    """

    def __init__(self, **model_params):
        """Initialize class."""
        self.model_params = model_params
        self.model_init_params = {}
        self.model_fit_params = {}  # Store fit-specific params here.
        self.model_ = None

    @staticmethod
    def _get_method_params(method):
        """Get the parameter names for a given method using the inspect module."""
        sig = inspect.signature(method)
        return [
            param
            for param in sig.parameters
            if param != "self"  # pylint: disable=magic-value-comparison
        ]

    def _split_params(self):
        """Automatically split parameters between __init__ and fit methods by
        inspecting their signatures.
        """
        # Get the class of the statsmodels model we're wrapping.
        model_class = self._model_class()

        # Get the parameter names for the model's __init__ and fit methods.
        init_params = self._get_method_params(model_class.__init__)
        fit_params = self._get_method_params(model_class.fit)

        # Split parameters into init and fit based on the method signatures.
        self.model_init_params = {
            k: v for k, v in self.model_params.items() if k in init_params
        }
        self.model_fit_params = {
            k: v for k, v in self.model_params.items() if k in fit_params
        }

    def fit(
        self,
        X: pandas.Series | pandas.DataFrame,
        y: pandas.Series | pandas.DataFrame | None = None,
    ):
        """Fit the statsmodels model to the data.

        Args:
            X (pandas.Series | pandas.DataFrame):
                (array-like, shape (n_samples, n_features)) but not np.ndarray
                Training data. This would typically be the time series data.
            y (pandas.Series | pandas.DataFrame | None):
                (array-like, shape (n_samples,)):
                Target values. Optional if the model does not require a target.

        Returns
            self (object)
        """
        # Pull out the correct 'feature'(s) to use.
        cols = [i for i in X.columns if TimePointLabels.PRESENT.value in i]
        # Flatten the input if necessary, as statsmodels expect 1D arrays for time series.
        X = np.asarray(X[cols]).flatten()
        y = np.asarray(y).flatten()

        # Automatically split params between __init__ and fit.
        self._split_params()

        # Initialize the statsmodels model using model_params.
        self.model_ = self._instantiate_model(X, y)

        # Fit the model to the data.
        # Issue here with y-data not being 1D, though it was
        # transformed earlier, in self._predict?
        self.model_ = self.model_.fit(**self.model_fit_params)

        return self

    def predict(self, X):
        """Predict using the fitted statsmodels model.

        Args:
            X (array-like, shape (n_samples, n_features)):
                Test data.

        Returns:
            predictions (array, shape (n_samples,)):
                Predicted values.
        """
        # Pull out the correct 'feature'(s) to use.
        cols = [i for i in X.columns if TimePointLabels.PRESENT.value in i]
        # Ensure X is flattened as statsmodels might expect 1D arrays for time series.
        X = np.asarray(X[cols]).flatten()

        return self.model_.forecast(len(X))

    def score(self, X, y, sample_weight=None):
        """Returns the score for the model on the given data.
        For regression tasks, this could be R^2, or you can
        implement a custom scoring method.

        Args:
            X (array-like, shape (n_samples, n_features)):
                Test data.
            y (array-like, shape (n_samples,)):
                True target values.

        Returns:
            score (float):
                Model score.
        """
        predictions = self.predict(X)
        # TODO: Update this to use an input method.  # pylint: disable=fixme
        return np.mean(
            (predictions - y) ** 2
        )  # Mean Squared Error as an example

    def get_params(self, deep=True):
        """Get parameters for the model.
        Required for GridSearchCV.

        Returns:
            params (dict):
                Parameter names mapped to their values.
        """
        return self.model_params

    def set_params(self, **params):
        """Set parameters for the model.
        Required for GridSearchCV.

        Args:
            params (dict):
                Parameter names mapped to their values.

        Returns:
            self (object)
        """
        self.model_params.update(params)
        return self

    def _model_class(self):
        """Return the specific statsmodels class that this wrapper is
        supposed to wrap.

        Subclasses should override this method to return the correct class.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    def _instantiate_model(self, X, y):  # pylint: disable=unused-argument
        """Instantiate the specific statsmodels model w ith parameters
        intended for the constructor (__init__).

        Args:
            X: Features.
            y: True labels.
        """
        model_class = self._model_class()  # Get the statsmodels class.
        return model_class(
            X, **self.model_init_params
        )  # Pass only the init params.
