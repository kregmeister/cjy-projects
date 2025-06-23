#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 21:23:24 2025

@author: cjymain
"""

import pandas as pd
import numpy as np
from scipy.special import expit
import pyarrow.parquet as pq
import pyarrow as pa
import traceback

from technically.computation.formulas import TechnicalFormulas
from technically.computation.signals import TechnicalIndicatorSignals as Signals
from technically.core.cleaning import PriceAdjustments
from technically.computation.backtesting import IndicatorSuccessRates
from technically.utils.log import get_logger


class TechnicalsExecutor:
    "Facilitates the recusrive calculation, scoring, and storing of candlesticks, technical indicators, EMAs, etc."

    def __init__(self, ticker: str, exchange: str, cap: str, sector: str, indicator_success_rates, base_path: str, db):
        self.ticker = ticker
        self.exchange = exchange
        self.cap = cap
        self.sector = sector
        self.indicator_success_rates = indicator_success_rates[indicator_success_rates["sector"] == sector]
        self.base_path = base_path
        self.db = db

    def execute(self):
        try:
            self.extract()
            integrity = self.check()

            if integrity:
                self.calculate()
                self.score()
                self.load()

                get_logger().info(f"Technicals calculated and saved to parquet {self.exchange}, partition {self.ticker}.")
        except Exception as e:
            get_logger().error(f"Unexpected error calculating technical indicators for {self.ticker}: {traceback.format_exc()}")
            return

    def extract(self):
        # Acquires price data for ticker from prices.duck
        self.df = self.db.sql(f'''
            SELECT
                '{self.ticker}' AS ticker,
                '{self.exchange}' AS exchange,
                '{self.cap}' AS cap,
                '{self.sector}' AS sector,
                * 
            FROM 
                "{self.ticker}" 
            ORDER BY 
                date;
            '''
        ).df()

    def check(self):
        # Verifies ticker has consistent price activity
        return (PriceAdjustments(self.df).
                priceActivityCheck(self.ticker, self.db))

    def calculate(self):
        # Initialize class that handles indicator calculations
        tc = TechnicalFormulas(self.df.copy())

        # Average True Range
        tc.ATR()
        # 5 period EMA
        tc.EMA(5, "close")
        # 10 period EMA
        tc.EMA(10, "close")
        # 20 period EMA
        tc.EMA(20, "close")
        # 50 period EMA
        tc.EMA(50, "close")
        # 100 period EMA
        tc.EMA(100, "close")
        # 250 period EMA
        tc.EMA(250, "close")
        # Kaufman's Adaptive Moving Average
        tc.KAMA(self.df["close"].copy(), 20)
        # Trend determination algorithm
        tc.currentTrend()
        # Finds range between last two established up/down trends
        #tc.currentRange()
        # Uses ROC percentiles to determine arbitrary daily ceilings/floors
        tc.priceCeilingsFloors()
        # A form of smoothing
        tc.kalmanFilterSingle()
        # Percent diff from 250 period max
        tc.percentFromExtreme()
        tc.percentFromExtreme(type="min")
        # Demand Index
        tc.demandIndex()
        # Aroon Oscillator
        tc.aroon()
        # Commodity Channel Index with smoothing line
        tc.CCI()
        # Relative Strength Index
        tc.RSI()
        # The percentage up or down of close a standard deviaiton is
        tc.STDevAsPercentOfSMA()
        # Short-term Know Sure Thing
        tc.shortKST()
        # Creates bands 2 standard deviations above and below close
        tc.bollingerBand()
        # Moving Average Convergence Divergence
        tc.MACD()
        # Williams' Percent R
        tc.williamsR()
        # Stochastic Oscillator with smoothing line
        tc.stochastics()
        # Parabolic SAR
        tc.parabolicSAR()
        # Average Directional Index
        tc.ADX()
        # On Balance Volume
        tc.OBV()

        tc.trend_strength_score()
        tc.trend_swing_confirmation()

        self.df = tc.persist()
        return self.df

    def score(self):
        # Initialize class that handles indicator signal identification
        score = Signals(self.ticker, self.df.copy(), self.indicator_success_rates, self.base_path)

        score.calculate("demandIndex")
        score.calculate("williamsR14")
        score.calculate("movingAverageConvergenceDivergence")
        score.calculate("CCI20")
        score.calculate("RSI14")
        score.calculate("bollingerSMA20")
        score.calculate("knowSureThing")
        score.calculate("stochasticK20")
        score.calculate("averageDirectionalIndex")
        score.calculate("parabolicSAR")
        score.calculate("candlestick", identify=False)
        # Sums technical indicator scores
        score.total_indicator_score()

        get_logger().info(f"Technical indicator signals scored for {self.ticker}.")

        self.df = score.persist()
        # Initiates backtesting
        IndicatorSuccessRates(
            self.df.copy(),
            self.ticker,
            self.base_path,
        ).execute()
        return

    def load(self):
        # Converts Pandas DataFrame to PyArrow Table
        parquet_table = pa.Table.from_pandas(self.df)

        # Writes PyArrow Table to a parquet dataset at proper partition
        pq.write_to_dataset(
            parquet_table,
            root_path=self.base_path+"/parquet/daily/",
            partition_cols=["exchange", "cap", "sector", "ticker"],
            compression="zstd",
            existing_data_behavior="delete_matching"  # Overwrites existing parquet files
        )

class FundamentalsExecutor:
    "Facilitates the recursive calculation, scoring, and storing of candlesticks, technical indicators, EMAs, etc."

    def __init__(self, ticker: str, exchange: str, cap: str, sector: str, indicator_success_rates, base_path: str, db):
        self.ticker = ticker
        self.exchange = exchange
        self.cap = cap
        self.sector = sector
        self.base_path = base_path
        self.db = db

    def execute(self):
        try:
            self.extract()
            if self.fund_df is None:
                return
            self.transform()
            self.load()
            get_logger().info(f"Fundamental ratios calculated and saved to parquet {self.exchange}, partition {self.ticker}.")
        except Exception as e:
            get_logger().error(f"Unexpected error calculating fundamental indicators for {self.ticker}: {traceback.format_exc()}")
            return

    def extract(self):
        # Identifies ticker statement tables
        statements = self.db.execute(f'''
            SELECT 
                table_name 
            FROM 
                duckdb_tables
            WHERE 
                table_name SIMILAR TO '{self.ticker}_.*'; 
            '''
        ).fetchall()
        stmts = [stmt[0] for stmt in statements]

        # Merges all available ticker statement data
        if len(stmts) == 1:
            self.fund_df = self.db.execute(f'''
                SELECT
                    '{self.ticker}' AS ticker,
                    '{self.exchange}' AS exchange,
                    '{self.cap}' AS cap,
                    '{self.sector}' AS sector,
                    * 
                FROM 
                    "{stmts[0]}"
                ORDER BY 
                    date;
                '''
            ).df()
        elif len(stmts) == 2:
            self.fund_df = self.db.execute(f'''
                SELECT
                    '{self.ticker}' AS ticker,
                    '{self.exchange}' AS exchange,
                    '{self.cap}' AS cap,
                    '{self.sector}' AS sector,
                    "{stmts[0]}".*, 
                    "{stmts[1]}".*
                FROM 
                    "{stmts[0]}"
                    JOIN "{stmts[1]}" ON
                        "{stmts[0]}".date = "{stmts[1]}".date;
                '''
            ).df()
        elif len(stmts) == 3:
            self.fund_df = self.db.execute(f'''
                SELECT
                    '{self.ticker}' AS ticker,
                    '{self.exchange}' AS exchange,
                    '{self.cap}' AS cap,
                    '{self.sector}' AS sector,
                    "{stmts[0]}".*, 
                    "{stmts[1]}".*,
                    "{stmts[2]}".*
                FROM 
                    "{stmts[0]}"
                    JOIN "{stmts[1]}" ON
                        "{stmts[0]}".date = "{stmts[1]}".date
                        JOIN "{stmts[2]}" ON
                            "{stmts[1]}".date = "{stmts[2]}".date;
                '''
            ).df()
        else:  # No ticker statement tables detected
            self.fund_df = None

    def transform(self):
        df = self.fund_df.copy()

        # Handles duplicate date columns
        if "date" not in df.columns and "date_1" in df.columns:
            df = df.rename(columns={"date_1": "date"})
        elif "date_1" in df.columns:
            df = df.drop(columns=["date_1"])
        if "date_2" in df.columns:
            df = df.drop(columns=["date_2"])
        if "date_3" in df.columns:
            df = df.drop(columns=["date_3"])

        ratios = [
            "netMargin = df.netinc / df.revenue",
            "grossMargin = df.grossProfit / df.revenue",
            "operatingMargin = df.opinc / df.revenue",
            "returnOnEquity = df.netinc / df.equity",
            "returnOnAssets = df.netinc / df.totalAssets",
            "currentRatio = df.assetsCurrent / df.liabilitiesCurrent",
            "quickRatio = (df.cashAndEq + df.acctRec) / df.liabilitiesCurrent",
            "debtEquity = df.debt / df.equity",
            "debtAssets = df.totalLiabilities / df.totalAssets",
            "interestCoverage = df.ebit / df.intexp",
            "assetTurnover = df.revenue / df.totalAssets",
            "receivablesTurnover = df.revenue / df.acctRec"
        ]

        for formula in ratios:
            try:
                # I.E. 'netMargin'
                name = formula.split("=")[0].strip()

                # Adds the variable embedded in each ratio object as a column
                df = pd.eval(formula, target=df)

                # Ensures ratio is a float
                df[name] = df[name].astype(np.float64)
            except (KeyError, AttributeError, ValueError) as e:
                get_logger().warning(f"Issue calculating {name} on {self.ticker}: {str(e)}")
                df[name] = 0.0

        self.final_df = df

    def load(self):
        # Converts Pandas DataFrame to PyArrow Table
        parquet_table = pa.Table.from_pandas(self.final_df)

        # Writes PyArrow Table to a parquet dataset at proper partition
        pq.write_to_dataset(
            parquet_table,
            root_path=self.base_path + "/parquet/quarterly/",
            partition_cols=["exchange", "cap", "sector", "ticker"],
            compression="zstd",
            existing_data_behavior="delete_matching"  # Overwrites existing parquet files
        )