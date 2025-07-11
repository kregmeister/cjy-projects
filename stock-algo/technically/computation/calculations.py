#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 21:23:24 2025

@author: cjymain
"""

import pandas as pd
import numpy as np
import pyarrow.parquet as pq
import pyarrow as pa
import traceback

from technically.computation.formulas import TechnicalFormulas
from technically.computation.signals import TechnicalIndicatorSignals as Signals
from technically.core.cleaning import PriceAdjustments
from technically.computation.backtesting import IndicatorSuccessRates
from technically.computation.prep import ModelPreparation
from technically.utils.handlers.db import DuckDB
from technically.utils.log import get_logger


class TechnicalsExecutor:
    """
    Facilitates the recursive calculation, scoring, and storing of candlesticks, technical indicators, EMAs, etc.
    """

    def __init__(self, ticker: str, exchange: str, cap: str, sector: str, indicator_success_rates, base_path: str):
        self.ticker = ticker
        self.exchange = exchange
        self.cap = cap
        self.sector = sector
        self.indicator_success_rates = indicator_success_rates[indicator_success_rates["sector"] == sector]
        self.base_path = base_path
        self.price_db = base_path + "/sql/prices.duck"

    def execute(self):
        try:
            self.extract()
            integrity = self.check()  # True or False returned

            if integrity:
                self.calculate()
                self.score()
                self.backtest()
                self.feature_engineering()
                self.load()

                get_logger().info(f"Technicals calculated and saved to parquet {self.exchange}, partition {self.ticker}.")
        except Exception:
            get_logger().error(f"Unexpected error calculating technical indicators for {self.ticker}: {traceback.format_exc()}")
            return

    def extract(self):
        """
        Gathers price data from DuckDB and stores it in self.df (pd.DataFrame).

        Returns:
            None
        """
        with DuckDB(self.price_db) as db:
            # Acquires price data for ticker from prices.duck
            self.df = db.sql(f'''
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
            return

    def check(self):
        """
        Triggers series of checks to determine whether to conduct technical analysis or not.

        Returns:
            bool: True for yes, False for no.
        """
        # Verifies ticker has consistent price activity
        return (PriceAdjustments(self.base_path, self.df).
                priceActivityCheck(self.ticker))

    def calculate(self):
        """
        Triggers calculations that produce technical indicators as new columns in self.df.

        Returns:
            None
        """
        # Initialize class that handles indicator calculations
        tc = TechnicalFormulas(self.df.copy())

        # Use self.df to solve for functions
        tc.average_true_range()
        tc.exponential_moving_average(5)
        tc.exponential_moving_average(10)
        tc.exponential_moving_average(20)
        tc.exponential_moving_average(50)
        tc.exponential_moving_average(100)
        tc.exponential_moving_average(250)
        tc.kaufman_adaptive_moving_average(self.df["close"].copy(), 20)
        tc.current_trend()
        #tc.currentRange()
        tc.price_ceilings_floors()
        tc.kalman_filter_single()
        tc.percent_from_extreme()
        tc.percent_from_extreme(type="min")
        tc.demand_index()
        tc.aroon_oscillator()
        tc.commodity_channel_index()
        tc.relative_strength_index()
        tc.stdev_percent_of_sma()
        tc.short_know_sure_thing()
        tc.bollinger_bands()
        tc.moving_average_convergence_divergence()
        tc.williams_percent_r()
        tc.stochastic_oscillator()
        tc.parabolic_sar()
        tc.average_directional_index()
        tc.on_balance_volume()
        tc.trend_strength_score()
        tc.trend_swing_confirmation()

        # Pulls calculation results from calculation handler class
        self.df = tc.persist()
        return

    def score(self):
        """
        Triggers calculations that check for technical indicator signals
        (pre-defined in indicatorSignals.json) and
        assigns scores (determined by backtesting) to indicator occurrences.
        Results are assigned as new columns in self.df.

        Returns:
            None
        """
        # Initialize class that handles indicator signal identification
        score = Signals(self.ticker, self.df.copy(), self.indicator_success_rates, self.base_path)

        score.calculate("demandIndex")
        score.calculate("williamsR14")
        score.calculate("macd")
        score.calculate("cci20")
        score.calculate("rsi14")
        score.calculate("bollingerSMA20")
        score.calculate("kst")
        score.calculate("stochasticK20")
        score.calculate("adx")
        score.calculate("psar")
        score.calculate("candlestick", identify=False)
        # Sums technical indicator scores
        score.total_indicator_score()

        get_logger().info(f"Technical indicator signals scored for {self.ticker}.")

        self.df = score.persist()
        return

    def backtest(self):
        """
        Triggers backtesting of technical indicator signals.

        Returns:
            None
        """
        # Initiates backtesting
        IndicatorSuccessRates(
            self.df.copy(),
            self.ticker,
            self.base_path,
        ).execute()
        return

    def feature_engineering(self):
        """
        Scales, lags, and rolls self.df columns specified in ml_config.json
        to help prepare data for Machine Learning models.

        Returns:
            None
        """
        self.df = ModelPreparation(self.df, mode="technicals").execute()
        self.df = ModelPreparation(self.df, mode="fundamentals").execute()
        return


    def load(self):
        """
        Writes self.df to a parquet file (includes all columns generated from above functions).

        Returns:
            None
        """
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
        return

class FundamentalsExecutor:
    """
    Facilitates the recursive calculation, scoring, and storing of candlesticks, technical indicators, EMAs, etc.
    """

    def __init__(self, ticker: str, exchange: str, cap: str, sector: str, indicator_success_rates, base_path: str):
        self.ticker = ticker
        self.exchange = exchange
        self.cap = cap
        self.sector = sector
        self.base_path = base_path
        self.fund_db = base_path + "/sql/fundamentals.duck"

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
        """
        Gathers fundamental statements data from DuckDB and stores it in self.fund_df (pd.DataFrame).

        Returns:
            None
        """
        with DuckDB(self.fund_db) as db:
            # Identifies ticker statement tables
            statements = db.execute(f'''
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
                self.fund_df = db.execute(f'''
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
                self.fund_df = db.execute(f'''
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
                self.fund_df = db.execute(f'''
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
        """
        Triggers calculations that produce fundamental indicators as new columns in self.final_df (pd.DataFrame).

        Returns:
            None
        """
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
            # I.E. 'netMargin'
            name = formula.split("=")[0].strip()
            try:
                # Adds the variable embedded in each ratio object as a column
                df = pd.eval(formula, target=df)

                # Ensures ratio is a float
                df[name] = df[name].astype(np.float64)
            except (KeyError, AttributeError, ValueError) as e:
                get_logger().warning(f"Issue calculating {name} on {self.ticker}: {str(e)}")
                df[name] = 0.0

        self.final_df = df

    def load(self):
        """
        Writes self.final_df to parquet file (includes all columns from above functions).

        Returns:
            None
        """
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
        return