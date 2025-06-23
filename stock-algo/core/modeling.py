#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  3 11:07:35 2025

@author: cjymain
"""

from technically.utils.handlers.spark import PySparkSession
from technically.utils.log import get_logger
from technically.spark.ML.core.extract import DatasetAggregation
from technically.spark.ML.core.prep import ModelPreparation
from technically.spark.ML.core.train import LogisticRegressionTC

import matplotlib.pyplot as plt
import os
import traceback

class PySparkModelingCoordinator:

    def __init__(self, date, base_path, group_by):
        self.date = date
        self.base_path = base_path
        self.group_by = group_by
        self.cols_to_scale = [
            'open', 'high', 'low', 'close', 'volume',
            'marketCap', 'enterpriseVal', 'EMA5',
            'EMA10', 'EMA20', 'EMA50', 'EMA100', 'EMA250',
            'KAMA20', 'twoWeekTrendLine', 'monthTrendLine',
            'threeMonthTrendLine', 'sixMonthTrendLine', 'yearTrendLine',
            'threeYearTrendLine', 'prevailingTrendLine', 'twoWeekTrend',
            'monthTrend', 'threeMonthTrend', 'sixMonthTrend', 'yearTrend',
            'threeYearTrend', 'prevailingTrend', 'priceChange', 'priceCeiling',
            'priceFloor', 'kalmanClose', 'percentDiffMax', 'percentDiffMin',
            'STDevPercent20', 'bollingerUpper20', 'bollingerSMA20', 'bollingerLower20',
            'bollingerRange20', 'parabolicSAR', 'OBV', 'advances', 'declines',
            'unchanged', 'adv_volume', 'decl_volume', 'unchanged_volume',
            'new_lows', 'new_highs', 'breadthOBV', 'cashAndEq', 'debt', 'equity',
            'retainedEarnings', 'totalAssets', 'totalLiabilities', 'capex',
            'consolidatedIncome', 'costRev', 'ebit', 'ebitda', 'ebt', 'grossProfit',
            'netinc', 'opex', 'opinc', 'revenue', 'rnd', 'sga', 'taxExp'
        ]
        self.cols_to_lag = ["close", "kalmanClose"]
        self.cols_to_roll = ["volume", "OBV"]

    def execute(self):
        group_paths = self.define_groups()
        with PySparkSession() as spark:
            for group_path in group_paths:
                ### TEMP ###
                if group_path != '/exchange=*/cap=*/sector=Utilities':
                    continue
                # PERMANENT
                if group_path == '/exchange=*/cap=*/sector=Unknown':
                    continue

                dataset = DatasetAggregation(spark.session, self.date, self.base_path, group_path).execute()

                train, val, test = ModelPreparation(spark.session, dataset).execute(self.cols_to_scale)
                print(train.columns)
                return

                train.coalesce(1).write.parquet(self.base_path + "/pyspark_test/training.parquet")
                train.repartition(1).write.parquet(self.base_path + "/pyspark_test/training.parquet", mode='overwrite')

                val.coalesce(1).write.parquet(self.base_path + "/pyspark_test/validation.parquet")
                val.repartition(1).write.parquet(self.base_path + "/pyspark_test/validation.parquet", mode='overwrite')

                test.coalesce(1).write.parquet(self.base_path + "/pyspark_test/testing.parquet")
                test.repartition(1).write.parquet(self.base_path + "/pyspark_test/testing.parquet", mode='overwrite')

                return
                LogisticRegressionTC()

                df.show()
                df.printSchema()
                return

    def define_groups(self):
        group_dirs = []
        for root, dirs, files in os.walk(self.base_path + "/parquet"):
            for dir in dirs:
                if self.group_by == "exchange":
                    if dir.startswith("exchange="):
                        group_dirs.append(f"/{dir}/cap=*/sector=*")
                elif self.group_by == "cap":
                    if dir.startswith("cap="):
                        group_dirs.append(f"/exchange=*/{dir}/sector=*")
                elif self.group_by == "sector":
                    if dir.startswith("sector="):
                        group_dirs.append(f"/exchange=*/cap=*/{dir}")
        return set(group_dirs)


    def visualize(self, df):
        fig, ax = plt.subplots(nrows=2, ncols=1)

        closes = ax[0].plot(df["date"], df["close"], color="black")
        scaled_closes = ax[1].plot(df["date"], df["close_scaled"], color="purple")
        # acceleration = ax[1][0].plot(df["date"], df["adxTrendValidation"], color="red")
        # dmiplus = ax[0][1].plot(df["date"], df["dmiPlus"], color="green")
        # dmiminus = ax[0][1].plot(df["date"], df["dmiMinus"], color="red")
        # trendstrength = ax[1][0].plot(df["date"], df["trend_strength"])

        for row in ax:
            for col in row:
                col.tick_params("y", rotation=45)
                col.tick_params("x", rotation=90)
                col.set_xticks(df["date"].iloc[::10])
                col.set_xlabel("Dates")
                col.legend()
        plt.show()
        plt.close()

        # test_plotting()
