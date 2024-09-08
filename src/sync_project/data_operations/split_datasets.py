"""This file contains functions to split datasets
into training and validation sets."""

import pandas


def split_by_fraction(
    dataframe: pandas.DataFrame,
    fractions: list[float],
) -> list[pandas.DataFrame]:
    """Splits the dataframe into chunks based on
    the given fractions. The resulting number of
    chunks is one more than the length of the
    fractions list. Any required sorting should be
    done prior to calling this function.

    Args:
        dataframe (pandas.DataFrame): Dataframe to
            split into chunks.
        fractions (list[float]): List of fractions
            of the initial dataframe's size to
            split the dataframe into. The sum of
            fractions must be less than 1. The last
            chunk will consist of the remainder of
            the dataframe.

    Returns:
        list[pandas.DataFrame]: List of chunks of
            dataframe after being split. The length
            of this list is one more than the length
            of the fractions list.

    Raises:
        ValueError: If the sum of fractions is greater
            than or equal to 1.
    """
    if sum(fractions) >= 1:
        raise ValueError("The sum of 'fractions' must be less than 1.")
    chunk_sizes = [int(len(dataframe) * frac) for frac in fractions]
    chunk_split_locations = [
        sum(chunk_sizes[: i + 1]) for i in range(len(chunk_sizes))
    ]
    chunks = []
    chunks.append(dataframe[0 : chunk_split_locations[0]])
    for i in range(1, len(chunk_split_locations)):
        chunks.append(
            dataframe[chunk_split_locations[i - 1] : chunk_split_locations[i]]
        )
    chunks.append(dataframe[chunk_split_locations[-1] :])
    return chunks
