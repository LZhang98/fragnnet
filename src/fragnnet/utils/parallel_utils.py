"""Parallel and sequential map helpers over iterables, Series and DataFrame rows, on joblib
(split out of ``utils/data_utils.py``). ``FRAGNNET_JOBLIB_N_JOBS`` sets the worker count.
"""

import logging
import os
import warnings
from collections.abc import Callable, Iterable, Iterator
from pprint import pformat
from typing import Any

import joblib
import pandas as pd
from tqdm import tqdm

JOBLIB_BACKEND = "loky"


FRAGNNET_JOBLIB_N_JOBS = os.environ.get("FRAGNNET_JOBLIB_N_JOBS", None)


JOBLIB_N_JOBS = (
    joblib.cpu_count() if FRAGNNET_JOBLIB_N_JOBS is None else int(FRAGNNET_JOBLIB_N_JOBS)
)


JOBLIB_TIMEOUT = 10800  # default is 300 seconds for "loky"

if FRAGNNET_JOBLIB_N_JOBS is not None:
    logging.info("Using FRAGNNET_JOBLIB_N_JOBS=%s for joblib n_jobs", FRAGNNET_JOBLIB_N_JOBS)


def seq_apply(iterator: Iterable, func: Callable, need_unpack: bool = False) -> list:
    result = []
    for i in iterator:
        if not need_unpack:
            result.append(func(i))
        else:
            result.append(func(*i))
    return result


def par_apply(
    iterator: Iterable,
    func: Callable,
    need_unpack: bool = False,
    return_as_generator: bool = False,
    n_jobs: int = JOBLIB_N_JOBS,
    unordered: bool = False,
) -> list | Iterator:
    """Apply a function to each item in an iterator in parallel.

    Args:
            iterator (_type_): _description_
            func (_type_): _description_
            need_unpack (bool, optional): _description_. Defaults to False.
            return_as_generator (bool, optional): _description_. Defaults to False.
            n_jobs (_type_, optional): _description_. Defaults to JOBLIB_N_JOBS.
            unordered (bool, optional): Yield generator results as workers finish.
                Requires ``return_as_generator=True``. Defaults to False.

    Returns:
            _type_: _description_
    """
    if unordered and not return_as_generator:
        raise ValueError("unordered=True requires return_as_generator=True")

    par_func = joblib.delayed(func)
    if unordered:
        return_as = "generator_unordered"
    elif return_as_generator:
        return_as = "generator"
    else:
        return_as = "list"
    parallel_kwargs = {
        "backend": JOBLIB_BACKEND,
        "n_jobs": n_jobs,
        "timeout": JOBLIB_TIMEOUT,
        "return_as": return_as,
    }
    try:
        parallel = joblib.Parallel(**parallel_kwargs)
    except ValueError:
        if not unordered:
            raise
        warnings.warn(
            "Installed joblib/backend does not support unordered generators; falling back "
            "to ordered generator results.",
            RuntimeWarning,
            stacklevel=2,
        )
        parallel_kwargs["return_as"] = "generator"
        parallel = joblib.Parallel(**parallel_kwargs)
    if not need_unpack:
        result = parallel(par_func(i) for i in iterator)
    else:
        result = parallel(par_func(*i) for i in iterator)
    return result


def par_apply_series(
    series: pd.Series, func: Callable, tqdm_leave: bool = True, use_tqdm: bool = True
) -> pd.Series:
    """Apply a function to each item in a pandas Series in parallel.

    Args:
            series (pd.Series): The Series to process.
            func (callable): The function to apply to each item.
            tqdm_leave (bool, optional): Whether to leave the tqdm progress bar after completion. Defaults to True.

    Returns:
            _type_: _description_
    """
    series_iter = (
        tqdm(series.items(), desc=pformat(func), total=series.shape[0], leave=tqdm_leave)
        if use_tqdm
        else series.items()
    )

    def series_func(tup: tuple) -> Any:
        return func(tup[1])

    result_list = par_apply(series_iter, series_func)
    result_series = pd.Series(result_list, index=series.index)
    return result_series


def par_apply_df_rows(df: pd.DataFrame, func: Callable, use_tqdm: bool = True) -> pd.Series:
    """Apply a function to each row of a DataFrame in parallel.

    Args:
            df (pd.DataFrame): The DataFrame to process.
            func (callable): The function to apply to each row.

    Returns:
            pd.Series: The results of applying the function to each row.
    """

    df_iter = (
        tqdm(df.iterrows(), desc=pformat(func), total=df.shape[0]) if use_tqdm else df.iterrows()
    )

    def df_func(tup: tuple) -> Any:
        return func(tup[1])

    result_list = par_apply(df_iter, df_func)
    if isinstance(result_list[0], tuple):
        raise NotImplementedError("Use par_apply_df_rows_t for multiple return values")
    result_series = pd.Series(result_list, index=df.index)

    return result_series


def par_apply_df_rows_t(
    df: pd.DataFrame, func: Callable, use_tqdm: bool = True
) -> tuple[pd.Series, ...]:
    """Apply a function to each row of a DataFrame in parallel.

    Args:
            df (pd.DataFrame): The DataFrame to process.
            func (callable): The function to apply to each row.

    Returns:
            pd.Series: The results of applying the function to each row.
    """
    df_iter = (
        tqdm(df.iterrows(), desc=pformat(func), total=df.shape[0]) if use_tqdm else df.iterrows()
    )

    def df_func(tup: tuple) -> Any:
        return func(tup[1])

    result_list = par_apply(df_iter, df_func)
    result_series = tuple([pd.Series(rl, index=df.index) for rl in zip(*result_list)])
    return result_series


def seq_apply_df_rows_t(
    df: pd.DataFrame, func: Callable, use_tqdm: bool = True
) -> tuple[pd.Series, ...]:
    """Apply a function to each row of a DataFrame in sequence, returning multiple Series."""
    df_iter = (
        tqdm(df.iterrows(), desc=pformat(func), total=df.shape[0]) if use_tqdm else df.iterrows()
    )

    def df_func(tup: tuple) -> Any:
        return func(tup[1])

    result_list = seq_apply(df_iter, df_func)
    result_series = tuple([pd.Series(rl, index=df.index) for rl in zip(*result_list)])

    return result_series
