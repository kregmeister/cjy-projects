#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 17:19:08 2024

@author: cjymain
"""

import pandas as pd

from technically.utils.handlers.db import DuckDB
from technically.utils.log import get_logger


class BreadthExecutor:
    """
    Calculates database-wide breadth metrics.
    """

    def __init__(self, base_path):
        self.base_path = base_path
        self.price_db = base_path + "/sql/prices.duck"

    def execute(self):
        # Aggregate outcomes for tickers in exchange (advances, declines, etc.)
        tickers = self.extract()
        df = self.calculate(tickers)
        self.load(df)

        get_logger().info("Breadth metrics calculated.")
        return

    def extract(self):
        """
        Gathers the list of tickers to be included in breadth metrics.

        Returns:
            list: list of tickers.
        """
        with DuckDB(self.price_db) as db:
            tickers = db.sql('''
                SELECT
                    CASE
                        WHEN duplicated = False THEN ticker
                        ELSE permaticker
                    END AS ticker
                FROM
                    metadata
                WHERE
                    designation != 'deadTicker';
                '''
            ).fetchall()
        return [ticker[0] for ticker in tickers]
    
    def calculate(self, tickers: list):
        """
        Calculates breadth metrics for each ticker and aggregates.

        Args:
            tickers (list): list of tickers.

        Returns:
            breadth_df (pd.DataFrame): DataFrame containing aggregated breadth metrics.
        """
        with DuckDB(self.price_db) as db:
            if not db.has_table("breadth"):  # Initialize
                self.init = True

                dates = db.execute('''
                    SELECT 
                        date
                    FROM 
                        pnc
                    ORDER BY 
                        date DESC;
                    '''
                ).df()
            else:  # Append
                self.init = False

                most_recent_date = db.sql('''
                    SELECT 
                        date 
                    FROM 
                        breadth 
                    ORDER BY 
                        date DESC 
                        LIMIT 1;
                    '''
                ).fetchone()

                dates = db.execute('''
                    SELECT
                        date
                    FROM
                        pnc
                    WHERE
                        date >= ?
                    ORDER BY
                        date DESC;
                    ''', [most_recent_date]
                ).df()

            breadth_df = pd.DataFrame(
                {
                    "advances": 0,
                    "declines": 0,
                    "unchanged": 0,
                    "adv_volume": 0,
                    "decl_volume": 0,
                    "unchanged_volume": 0,
                    "new_highs": 0,
                    "new_lows": 0,
                },
                index=dates["date"]
            )

            for ticker in tickers:
                # Uses SQL to gather basic breadth metrics by-ticker
                df = db.execute(f'''
                    SELECT
                        date,
                        CASE
                          WHEN close > LAG(close) OVER (ORDER BY date)
                          THEN 1
                          ELSE 0
                        END AS advances,
                        CASE
                          WHEN close < LAG(close) OVER (ORDER BY date) THEN 1
                          ELSE 0
                        END AS declines,
                        CASE
                          WHEN close = LAG(close) OVER (ORDER BY date) THEN 1
                          ELSE 0
                        END AS unchanged,
                        CASE
                          WHEN close > LAG(close) OVER (ORDER BY date) THEN volume
                          ELSE 0
                        END AS adv_volume,
                        CASE
                          WHEN close < LAG(close) OVER (ORDER BY date) THEN volume
                          ELSE 0
                        END AS decl_volume,
                        CASE
                          WHEN close = LAG(close) OVER (ORDER BY date) THEN volume
                          ELSE 0
                        END AS unchanged_volume,
                        CASE 
                          WHEN close = MIN(close) OVER (
                            ORDER BY date
                            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                            )
                          THEN 1
                          ELSE 0
                        END AS new_lows,
                        CASE
                        WHEN close = MAX(close) OVER (
                            ORDER BY date
                            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                            )
                          THEN 1
                          ELSE 0
                        END AS new_highs
                    FROM
                        "{ticker}"
                    ORDER BY 
                        date DESC;
                    '''
                ).df()
                df.index = df["date"]
                df = df.drop(columns=["date"])

                # Adds ticker's breadth metrics from 'df' to cumulative 'breadth_df'
                breadth_df = breadth_df.combine(df, lambda a, b: a + b, fill_value=0)

        breadth_df = breadth_df.astype(int)

        # Calculates breadth indicators
        breadth_df["advanceDeclineLine"] = (breadth_df["advances"] - breadth_df["declines"]).cumsum()

        ad_ema19 = breadth_df["advanceDeclineLine"].ewm(span=19).mean()
        ad_ema39 = breadth_df["advanceDeclineLine"].ewm(span=39).mean()
        breadth_df["mcclellanOscillator"] = ad_ema19 - ad_ema39

        high_low_idx = breadth_df["new_highs"] / (breadth_df["new_highs"] + breadth_df["new_lows"])
        breadth_df["newHighLowIndex"] = high_low_idx.rolling(window=10).mean()

        breadth_df["TRIN"] = ((breadth_df["advances"] / breadth_df["declines"]) /
                              (breadth_df["adv_volume"] + breadth_df["decl_volume"]))

        breadth_df["breadthOBV"] = (breadth_df["adv_volume"] / breadth_df["decl_volume"]).cumsum()

        return breadth_df

    def load(self, df: pd.DataFrame):
        """
        Writes breadth metrics to DuckDB.

        Args:
            df (pd.DataFrame): Resultant DataFrame.

        Returns:
            None
        """
        with DuckDB(self.price_db) as db:
            if self.init:  # Creates
                db.sql('''
                    CREATE TABLE breadth AS SELECT * FROM df;
                    '''
                )
            else:  # Appends
                db.sql('''
                    INSERT OR REPLACE INTO breadth SELECT * FROM df;
                    '''
                )

