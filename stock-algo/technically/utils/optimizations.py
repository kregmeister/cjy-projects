#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 24 13:20:24 2025

@author: cjymain
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    MaxAbsScaler,
    RobustScaler,
    QuantileTransformer,
    PowerTransformer
)


def npshift(arr, periods=1):
    result = np.empty_like(arr, dtype=float)
    if periods > 0:
        result[:periods] = np.nan
        result[periods:] = arr[:-periods]
    elif periods < 0:
        result[periods:] = np.nan
        result[:periods] = arr[-periods:]
    else:
        return arr.copy()
    return result

def nprolling(arr, period: int, calc_type: str = "mean", func_dict = {"func": None}):
    if isinstance(arr, np.ndarray):
        # Keep track of NaN positions
        null_mask = np.isnan(arr)

        # Get indices of non-NaN values
        valid_indices = np.where(~null_mask)[0]

        # Extract valid values
        valid_values = arr[valid_indices]

        # If we don't have enough non-NaN values to form even one window, return array of NaNs
        if len(valid_values) < period:
            return np.full_like(arr, np.nan)

        # Perform calculation on valid values
        if calc_type != "func":
            windows = np.lib.stride_tricks.sliding_window_view(valid_values, period)

        if calc_type == "mean":
            result_values = np.mean(windows, axis=1)
        elif calc_type == "sum":
            result_values = np.sum(windows, axis=1)
        elif calc_type == "std":
            result_values = np.std(windows, axis=1)
        elif calc_type == "min":
            result_values = np.min(windows, axis=1)
        elif calc_type == "max":
            result_values = np.max(windows, axis=1)
        elif calc_type == "func":
            func = func_dict.get("func")
            func_params = {k: v for k, v in func_dict.items() if k != "func"}

            output_len = arr.shape[0] - period + 1
            result_values = np.full(arr.shape[0], False, dtype=bool)
            for i in range(output_len):
                window = arr[i:i+period]
                result_values[i+period-1] = func(window, **func_params)
            return result_values
        else:
            raise ValueError("Invalid type. Must be 'mean', 'std', 'min', 'max', or 'func'.")
    else:
        raise TypeError("Input must be a NumPy array.")

    # Create result array with same shape as input, filled with NaNs
    result = np.full_like(arr, np.nan)

    # Place results at correct positions, accounting for window size
    # The result indices need to be offset by (period-1) to align correctly
    result_positions = valid_indices[period - 1:]

    # Place results
    result[result_positions] = result_values

    return result

def weighted_mean(subdf: pd.Series or np.ndarray, weight_col_name: str, group_by_name: str):
    excluded_cols = [weight_col_name, group_by_name, "assetType"]
    for col in subdf.columns:
        if col in excluded_cols:
            continue
        subdf[subdf[group_by_name] == "Unknown"][col] = np.average(subdf[col], weights=subdf[weight_col_name])
    return subdf

def bucketizer(subdf: pd.DataFrame, buckets: list = ["1997-01-01", "2003-01-01", "2009-01-01", "2017-01-01", "2020-01-01"]):
    """
    :param subdf: Must include date column and columns to split into buckets
    :param buckets: Defaults at a reasonable set of eras
    :return: The bucketed dataframes
    """
    bucketer = subdf["date"]
    bucketed_dfs = []
    for i in range(len(buckets) + 1):
        if i == 0:
            condition = bucketer < buckets[i]
        elif i != 0 and i != len(buckets):
            condition = (bucketer < buckets[i]) & (bucketer >= buckets[i - 1])
        else:
            condition = bucketer >= buckets[-1]
        bucket = subdf[condition]
        # For newer tickers to be properly bucketed
        if bucket.empty:
            continue
        bucketed_dfs.append(bucket)
    return bucketed_dfs

def scaler(scaler: str, subdf: pd.DataFrame or np.ndarray, bucketed=False, return_as="numpy"):
    """
    scaler: Pass the scaler to use: options are Standard, MinMax, MaxAbs, Robust, QuantileTransformer, PowerTransformer
    subdf: Pass a dataframe containing columns to standardize
    bucketed: Set to True to scale each self.bucket dataframe independently; set to False to scale full dataframe
    """
    if type(subdf) == pd.Series:
        subdf = pd.DataFrame(subdf)

    if scaler == "Standard":
        scaler = StandardScaler()
    elif scaler == "MinMax":
        scaler = MinMaxScaler()
    elif scaler == "MaxAbs":
        scaler = MaxAbsScaler()
    elif scaler == "Robust":
        scaler = RobustScaler()
    elif scaler == "QuantileTransformer":
        scaler = QuantileTransformer()
    elif scaler == "PowerTransformer":
        scaler = PowerTransformer()

    if bucketed:
        buckets = bucketizer(subdf)
        scaled_buckets = []
        for bucket in buckets:
            scaler.fit(bucket)
            scaled_buckets.append(scaler.transform(bucket))
        scaled_df = np.concatenate(scaled_buckets)
    else:
        scaler.fit(subdf)
        scaled_df = scaler.transform(subdf)
    if return_as == "numpy":
        return scaled_df
    elif return_as == "pandas":
        return pd.DataFrame(scaled_df, columns=subdf.columns)

def rolling_zscore(data: pd.Series or np.ndarray, window=250, min_len=120):
    if len(data) <= window:
        window = min_len

    if type(data) == np.ndarray:
        data = pd.Series(data)
    return (data - data.rolling(window).mean()) / data.rolling(window).std()

def percentile(data: pd.Series or np.ndarray, percent: int or str):
    # If percent is formatted as "1st" or "50th"
    if isinstance(percent, str):
        percent = int(percent[:-2])

    if type(data) == pd.Series:
        data = data.dropna()
        A = data.values
    if type(data) == np.ndarray:
        A = data[~np.isnan(data)]

    return np.percentile(A, percent)

def lineBestFit(Y: pd.Series or np.ndarray, period: int, return_as="list"):
    if isinstance(Y, pd.Series):
        Y = Y.values
    X = np.asarray(range(len(Y)))

    # Precompute sliding window sums
    X_slices = np.lib.stride_tricks.sliding_window_view(X, period)
    Y_slices = np.lib.stride_tricks.sliding_window_view(Y, period)

    n_windows = X_slices.shape[0]
    sum_x = np.sum(X_slices, axis=1)
    sum_y = np.sum(Y_slices, axis=1)
    sum_xy = np.sum(X_slices * Y_slices, axis=1)
    sum_xx = np.sum(X_slices ** 2, axis=1)

    # Compute covariance for each window
    denominator = (period * sum_xx) - (sum_x ** 2)
    numerator = (period * sum_xy) - (sum_x * sum_y)

    covariances = np.full(n_windows, np.nan)

    mask = denominator != 0
    covariances[mask] = numerator[mask] / denominator[mask]

    # Fill in None for oldest values
    covariance_result = [np.nan] * (period - 1) + covariances.tolist()

    if return_as == "numpy":
        return np.array(covariance_result)
    else:
        return covariance_result

def failureSwings(Y_window, type: str, threshold: int or float):
    """
    :param Y_window: A window of values (from Y.rolling(window)): pd.Series
    :param type: either 'bottom' or 'top'
    :param threshold: indicator threshold (I.e. -100 for bullish CCI)
    :param min_thresh: Minimum number of periods between swing points
    :return:
    """
    Y = Y_window.tolist()
    if type == "bottom":
        low = min(Y)
        if low > threshold:
            return False
        min_thresh = Y.index(low) + 3
        if min_thresh > len(Y)+1:
            return False
        Y_eval = Y[min_thresh:]
        try:
            lowest_valley = min(
                [v for i, v in zip(range(len(Y_eval)-1), Y_eval[:-1])
                 if Y_eval[i] < Y_eval[i+1] and Y_eval[i] < Y_eval[i-1]]
            )
        except ValueError:
            return False
        bullish_swing = (Y[-2] == lowest_valley)
        return bullish_swing

    if type == "top":
        high = max(Y)
        if high < threshold:
            return False
        max_thresh = Y.index(high) + 3
        if max_thresh > len(Y)+1:
            return False
        Y_eval = Y[max_thresh:]
        try:
            highest_peak = max(
                [v for i, v in zip(range(len(Y_eval)-1), Y_eval[:-1])
                 if Y_eval[i] > Y_eval[i+1] and Y_eval[i] > Y_eval[i-1]]
            )
        except ValueError:
            return False
        bearish_swing = (Y[-2] == highest_peak)
        return bearish_swing