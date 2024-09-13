"""This file contains a class useful for
fitting functions to data."""

import numpy
from sklearn.model_selection import GridSearchCV

from sync_project.forecasting.forecast import Forecast


class FunctionFitter(Forecast):
    """This class is helpful in fitting functions to data."""

    def fit(self):  # pylint: disable=too-many-locals
        # Clearly this monstrosity of a method needs to
        # be broken down into smaller sub-methods...
        """Method to find the function of the given options
        that best fits the data.
        """
        if self.cv:
            return self.fit_with_gs()
        return self.fit_without_gs()

    def fit_with_gs(self):
        """Method to find the function of the given options
        that best fits the data, using GridSearchCV."""
        for _, method in enumerate(self.methods.keys()):
            method_class = self.methods[method]["class"]()
            param_grid = self.methods[method]["param_grid"]
            # We could do this outside of the loop, but then
            # we would have to make special arrangements if
            # we pass a generator so it will loop over itself.
            # Instead, we just keep it here.
            cv = self.cv or zip(
                [numpy.arange(0, len(self.training_time_data))],
                [numpy.arange(0, len(self.training_time_data))],
            )
            # pylint: disable=duplicate-code
            grid_search = GridSearchCV(
                method_class,
                param_grid,
                scoring=self.metric,
                cv=cv,
                verbose=self.verbose,
            )
            grid_search.fit(
                self.training_time_data,
                self.training_value_data,
            )
            # pylint: enable=duplicate-code
            preds = grid_search.predict(X=self.validation_time_data)
            score = self.metric(preds, self.validation_value_data)
            self.methods[method].update(
                {
                    "fit": preds,
                    "fit_times": self.validation_time_data,
                    "result_object": grid_search,
                    "params": grid_search.best_params_,
                    self.metric.__name__: score,
                    # These don't make sense if we're
                    # not making forecasts:
                    # "forecast": forecast,
                    # "forecast_times": self.validation_time_data[:-1],
                }
            )
        self.best_method_name = self.comparison_method(
            self.methods,
            key=lambda v: self.methods[v][self.metric.__name__],
        )
        # Record best model fits:
        self.best_forecast_method = self.methods[self.best_method_name][
            "class"
        ]()
        self.best_forecast_fit_result = self.methods[self.best_method_name][
            "fit"
        ]

        return self

    def fit_without_gs(self):  # pylint: disable=too-many-locals
        # Clearly this monstrosity of a method needs to
        # be broken down into smaller sub-methods...
        """Method to find the function of the given options
        that best fits the data, without using GridSearchCV.
        """
        for _, method in enumerate(self.methods.keys()):
            method_class = self.methods[method]["class"]
            param_grid = self.methods[method]["param_grid"]
            opt_obj = method_class(**param_grid)
            opt_obj.fit(
                self.training_time_data,
                self.training_value_data,
            )
            preds = opt_obj.predict(X=self.validation_time_data)
            score = (
                numpy.inf
                if any(numpy.isinf(preds))
                else self.metric(self.validation_value_data, preds)
            )
            self.methods[method].update(
                {
                    "fit": preds,
                    "fit_times": self.validation_time_data,
                    "result_object": opt_obj,
                    "params": opt_obj.opt_params_,
                    self.metric.__name__: score,
                    # These don't make sense if we're
                    # not making forecasts:
                    # "forecast": forecast,
                    # "forecast_times": self.validation_time_data[:-1],
                }
            )
        self.best_method_name = self.comparison_method(
            self.methods,
            key=lambda v: self.methods[v][self.metric.__name__],
        )
        # Record best model fits:
        self.best_forecast_method = self.methods[self.best_method_name][
            "class"
        ]()
        self.best_forecast_fit_result = self.methods[self.best_method_name][
            "fit"
        ]

        return self
