"""This file contains tests for the data_labeling operations."""

import pytest

from sync_project.data_operations.data_labeling import (
    timeseries_to_labels,
    unsorted_timeseries_to_labels,
)

# pylint: disable=fixme,unused-variable


def test_timeseries_to_labels(test_df):
    """This function tests the timeseries_to_labels function."""
    train_df, _ = test_df
    # Test data types for data:
    with pytest.raises(TypeError):
        timeseries_to_labels(1)
    with pytest.raises(ValueError):
        timeseries_to_labels(train_df)
    with pytest.raises(TypeError):
        timeseries_to_labels(
            train_df,
            cols="Time",
        )
    # Test Series success:
    this_test_df = train_df["Time"].copy()
    labels_df = timeseries_to_labels(
        this_test_df,
    )
    # TODO: test for expected labels and col names
    # Test single column df success:
    this_test_df = train_df["Time"].to_frame()
    labels_df = timeseries_to_labels(
        this_test_df,
    )
    # TODO: test for expected labels and col names
    # Test multi-col df success:
    labels_df = timeseries_to_labels(
        train_df,
        cols=["Value"],
    )
    # TODO: test for expected labels and col names
    labels_df = timeseries_to_labels(
        train_df,
        cols=["Time", "Value"],
    )
    # TODO: test for expected labels and col names
    labels_df = timeseries_to_labels(
        train_df,
        cols=["Time", "Value"],
        p=3,
        n=2,
    )
    # TODO: test for expected labels and col names


def test_unsorted_timeseries_to_labels(test_df):
    """This function tests the
    unsorted_timeseries_to_labels function."""
    train_df, _ = test_df
    # TODO: test ValueError when no sort_col
    # TODO: test series, single column df
    # Test multi-column df:
    this_test_df = train_df.reindex([3, 1, 4, 2, 0])
    labels_df = unsorted_timeseries_to_labels(
        this_test_df,
        sort_col="Time",
        cols=["Time", "Value"],
    )
    # pytest.set_trace()
