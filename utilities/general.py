"""general.py"""

import pandas as pd
import numpy as np
from pathlib import Path
from itertools import zip_longest


def get_project_root() -> Path:
    return Path(__file__).parent.parent


def _grouper(iterable, n, fill_value=None):
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fill_value)


def list_grouper(iterable: {list, tuple}, n: int, fill_value=None):
    """
    Returns a list of lists where each sub-list is of size 'n'
    :param iterable: list or tuple
    :param n: length of each sub-list
    :param fill_value: value to be populated as an element into a sub-list that does not have 'n' elements
    :return:
    """
    g = list(_grouper(iterable, n, fill_value))
    try:
        g[-1] = [e for e in g[-1] if e is not None]
        return [list(tup) for tup in g]
    except IndexError:
        return [[]]


def filter_df_by_column_str_values(df: pd.DataFrame, column_value_map: dict, case_insensitive: bool = True) -> pd.DataFrame:
    """
    Returns a filtered DataFrame after filtering for column values containing str
    :param df: DataFrame
    :param column_value_map: dict
        keys: column header
        values: str or list of str
    :param case_insensitive: bool if true both column headers and column values are case insensitive
    :return: DataFrame
    """

    col_copy = df.columns  # store the original column before converting to lower case column headers (if applicable)
    if case_insensitive:
        df.columns = col_copy.str.lower()

    # loop thorough each relevant column and filter the rows
    for col_name, col_value in column_value_map.items():
        if case_insensitive:
            col_value = col_value.lower()
            col_name = col_name.lower()
            col = df[col_name].str.lower()
        else:
            col = df[col_name]

        # column value to be filtered by can either be a str or list of str
        if isinstance(col_value, str):
            df = df[col == col_value]
        else:
            df = df[col.isin(col_value)]

    if case_insensitive:
        df.columns = col_copy
    return df


def two_stage_data_iterable(initial_stage_data: {float, tuple, list, np.array}, terminal_stage_data: {float, tuple, list, np.array},
                            terminal_stage_start: int):
    """
    Returns an array of floats. Takes two sets of data and connect them using interpolated values.
    :param initial_stage_data: float, tuple, list, np.array
    :param terminal_stage_data: float, tuple, list, np.array
    :param terminal_stage_start: int
    :return: np.array
    """
    # handle inputs (convert to np.array if value is not iterable)
    initial_stage_data = return_iterable(initial_stage_data)
    terminal_stage_data = return_iterable(terminal_stage_data)

    # check inputs (terminal stage can't start before the initial stage)
    if terminal_stage_start <= len(initial_stage_data):
        raise ValueError(f"Terminal stage can't start before the initial stage. 'terminal_stage_start' is "
                         f"{terminal_stage_start} while length of 'initial_stage_data' is {len(initial_stage_data)}")

    # interpolated data and combine the initial array with the interpolated data and terminal array
    inter_data = np.linspace(initial_stage_data[-1], terminal_stage_data[0], terminal_stage_start - len(initial_stage_data) + 1)
    return np.concatenate((initial_stage_data, inter_data[1: -1], terminal_stage_data))


def return_iterable(var):
    if not isinstance(var, (list, np.ndarray, tuple)):
        return np.array([var])
    else:
        return var



