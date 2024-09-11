"""This file contains functions to label datasets."""

import pandas

from sync_project.constants import DEFAULT_TIMESERIES_LABELS_COL_NAME


def timeseries_to_labels(
    data: pandas.Series | pandas.DataFrame,
    cols: list[str] | None = None,
    p: int = 1,
    n: int = 1,
) -> pandas.DataFrame:
    """This function turns a uni- or multi-variate
    data time series into a "supervised" time series,
    using a lag window of p for the number of past
    features, and a negative lag window of n for the
    number of future labels.

    Args:
        data (pandas.Series | pandas.DataFrame):
            Series or DataFrame with the data.
        cols (list[str] | None): List of column names
            to be used as features and predictions.
            May be None if data is a pandas.Series or
            pandas.DataFrame with only one column. If
            None, the single column in data is used,
            and the column name in the output will
            match that in data (if present) or default
            to 'var'. Defaults to None.
        p (int): Number of past datapoints to use as
            features.
        n (int): Number of future datapoints to use as
            labels.

    Returns:
        pandas.DataFrame: DataFrame containing the new
            features and predictions.

    Raises:
        TypeError: If data is not a pandas.Series or
            pandas.DataFrame.
        ValueError: When data is a pandas.DataFrame with
            multiple columns and cols is None.
        TypeError: If cols is not a list.
    """
    if not isinstance(data, (pandas.Series, pandas.DataFrame)):
        raise TypeError(
            "'data' must be a pandas.Series or pandas.DataFrame."
            f" A {type(data)} was given."
        )
    default_col_name = DEFAULT_TIMESERIES_LABELS_COL_NAME
    if not cols:
        if isinstance(data, pandas.Series):
            cols = [data.name] or [default_col_name]
        elif data.shape[1] == 1:
            cols = [data.columns[0]] or [default_col_name]
        else:
            raise ValueError(
                "'cols' must be supplied when `data` has"
                " more than one column."
            )
    if not isinstance(cols, list):
        raise TypeError(f"'cols' must be a list; a {type(cols)} was given.")
    if isinstance(data, pandas.Series):
        data = data.to_frame(name=cols[0])
    for col in cols:
        for i in range(p, 0, -1):
            data[col + f" (t-{i})"] = data[col].shift(i)
        data[col + " (t)"] = data[col].copy()
        for i in range(1, n):
            data[col + f" (t+{i})"] = data[col].shift(-i)
    return data


def unsorted_timeseries_to_labels(
    data: pandas.Series | pandas.DataFrame,
    *args,
    sort_col: str | None = None,
    **kwargs,
) -> pandas.DataFrame:
    """This function sorts, and then passes along to
    timeseries_to_labels to turn into a "supervised"
    time series, a uni- or multi-variate data time
    series.

    Args:
        data (pandas.Series | pandas.DataFrame):
            Series or DataFrame with the data.
        sort_col (str | None): Column name to be used
            to sort data. May be None if data is a
            pandas.Series or pandas.DataFrame with only
            one column. If None, the single column's
            index in data is used.
        *args: Extra positional arguments to pass to
            timeseries_to_labels.
        **kwargs: Extra keyword arguments to pass to
            timeseries_to_labels.

    Returns:
        pandas.DataFrame: DataFrame containing the new
            features and predictions.

    Raises:
        ValueError: When data is a pandas.DataFrame with
            multiple columns and sort_col is None.
    """
    if not sort_col:
        if isinstance(data, pandas.Series) or data.shape[1] == 1:
            sort_col = data.name or data.index
        else:
            raise ValueError(
                "'sort_col' must be supplied when `data` has"
                " more than one column."
            )
    data = data.sort_values(by=sort_col, axis=0)
    return timeseries_to_labels(data, *args, **kwargs)


# test types, test col names, test sorting
