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

from pyspark.sql import Window
import pyspark.sql.functions as F
from pyspark.ml.feature import RobustScaler, VectorAssembler

from technically.utils.log import get_logger


class ModelPreparation:
    "Prepares individual ticker data for core models."

    def __init__(self, df: pd.DataFrame, mode: str):
        """

        :param df: Ticker's DataFrame
        :param mode: "technicals" or "fundamentals"; specifies which json configs to open
        """
        self.df = df
        if mode == "technicals":
            None


    def execute(self, scale_cols, lag_cols, rolling_cols):
        self.df = self.scale(scale_cols)
        self.df = self.create_lagging_features()
        self.df = self.create_rolling_features()


        return self.df

    def scale(self, features_to_scale):
        # Assemble columns to be scaled
        assembler = VectorAssembler(inputCols=features_to_scale, outputCol="features")
        df = assembler.transform(self.df)

        # Separate each ticker's data to be scaled individually
        tickers = df.select("ticker").distinct().collect()

        def _scale_ticker(ticker, input_df):
            # Filter out data not associated with ticker
            ticker_df = input_df.filter(F.col("ticker") == ticker)

            # Initialize scaler
            scaler = RobustScaler(
                inputCol="features",
                outputCol="scaled_features",
            )

            # Fit and transform
            scaler_model = scaler.fit(ticker_df)
            scaled_df = scaler_model.transform(ticker_df)

            return scaled_df

        scaled_dfs = [_scale_ticker(ticker, df) for ticker in tickers]

        final_df = self.session.unionAll(scaled_dfs)
        return final_df

        # while True:
        #    try:
        #        if len(cols_to_scale) == 0:
        #            return
        #        subdf = df[cols_to_scale]
        #        break
        #    except KeyError as e:
        #        col = str(e).split("'")[1]
        #        cols_to_scale.remove(col)
        #       continue
        return scaled_df

    def create_lagging_features(self, features_to_lag, lag_periods=(1, 2, 3, 5)):
        """
        Create lagged versions of specified features.

        df: Spark DataFrame
        features_to_lag: List of feature names to lag
        lag_periods: List of lag periods to create

        :return: Spark DataFrame with lagged features
        """

        df = self.df

        window_spec = Window.partitionBy("ticker").orderBy("date")

        for feature in features_to_lag:
            for lag_period in lag_periods:
                lagged_feature = f"{feature}_lag_{lag_period}"
                df = df.withColumn(
                    lagged_feature,
                    F.lag(F.col(feature), lag_period).over(window_spec)
                )
        return df

    def create_rolling_features(self, features_to_roll, windows=(5, 10, 20)):
        """
        Create rolling statistics for specified features.

        :param df: Spark DataFrame
        :param features_to_roll: List of feature names to create rolling stats for
        :param windows: List of window sizes for rolling calculations

        :return: Spark DataFrame with rolling statistical features
        """

        df = self.df

        for feature in features_to_roll:
            for window_size in windows:
                window_spec = (Window.partitionBy("ticker")
                               .orderBy("date")
                               .rowsBetween(-window_size + 1, 0))

                # Rolling statistics
                df = df.withColumn(
                    f"{feature}_rolling_mean_{window_size}",
                    F.avg(F.col(feature)).over(window_spec)
                )

                df = df.withColumn(
                    f"{feature}_rolling_std_{window_size}",
                    F.stddev(F.col(feature)).over(window_spec)
                )

                df = df.withColumn(
                    f"{feature}_rolling_min_{window_size}",
                    F.min(F.col(feature)).over(window_spec)
                )

                df = df.withColumn(
                    f"{feature}_rolling_max_{window_size}",
                    F.max(F.col(feature)).over(window_spec)
                )

        return df
