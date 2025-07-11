#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 13:03:47 2025

@author: cjymain
"""

# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 14:31:58 2025

@author: cjymain
"""

import pandas as pd
import importlib_resources as rs

import technically.utils.optimizations as opts

class ModelPreparation:
    """
    Prepares individual ticker data for core models.
    """

    def __init__(self, df: pd.DataFrame, mode: str):
        self.df = df
        self.mode = mode

    def get_config(self):
        feature_engineering_config = pd.read_json(
            rs.open_text("technically", "conf/ml_config.json")
        )[self.mode]

        self.scaling_type = feature_engineering_config["cols_to_scale"]["type"]
        self.scaling_cols = feature_engineering_config["cols_to_scale"]["features"]

        self.rolling_periods = feature_engineering_config["cols_to_roll"]["windows"]
        self.rolling_cols = feature_engineering_config["cols_to_roll"]["features"]

        if self.mode == "technicals":  # See 'ml_config.json'
            self.lagging_periods = feature_engineering_config["cols_to_lag"]["lag_periods"]
            self.lagging_cols = feature_engineering_config["cols_to_lag"]["features"]

    def execute(self):
        self.get_config()

        self.scale(self.scaling_cols, self.scaling_type)
        self.roll(self.rolling_cols, self.rolling_periods)
        if self.mode == "technicals":
            self.lag(self.lagging_cols, self.lagging_periods)


        return self.df

    def scale(self, features_to_scale: list, scaling_type: str):
        """
        Scales features according to scaling_type

        :param features_to_scale: List of features to scale
        :param scaling_type: Can equal Standard, MinMax, MaxAbs, Robust, QuantileTransformer, PowerTransformer
        """
        subdf = self.df[features_to_scale].copy()

        # Scales
        scaled_df = opts.scaler(
            scaling_type,
            subdf,
            return_as="pandas"
        )
        # Applies changes
        self.df[features_to_scale] = scaled_df
        return

    def roll(self, features_to_roll, windows):
        """
        Create rolling statistics for specified features.

        :param features_to_roll: List of feature names to create rolling stats for
        :param windows: List of window sizes for rolling calculations
        """

        subdf = self.df[features_to_roll].copy()

        for feature in features_to_roll:
            for window_size in windows:
                arr = subdf[feature].values

                self.df[f"{feature}_mean_{window_size}"] = opts.nprolling(
                    arr,
                    window_size,
                    calc_type="mean"
                )

                self.df[f"{feature}_std_{window_size}"] = opts.nprolling(
                    arr,
                    window_size,
                    calc_type="std"
                )

                self.df[f"{feature}_min_{window_size}"] = opts.nprolling(
                    arr,
                    window_size,
                    calc_type="min"
                )

                self.df[f"{feature}_max_{window_size}"] = opts.nprolling(
                    arr,
                    window_size,
                    calc_type="max"
                )

        return

    def lag(self, features_to_lag, lag_periods):
        """
        Create lagged versions of specified features.

        df: Spark DataFrame
        features_to_lag: List of feature names to lag
        lag_periods: List of lag periods to create

        :return: Spark DataFrame with lagged features
        """

        subdf = self.df[features_to_lag].copy()

        for feature in features_to_lag:
            for lag_period in lag_periods:
                self.df[f"{feature}_lagged_{lag_period}"] = subdf[feature].shift(lag_period)
        return
