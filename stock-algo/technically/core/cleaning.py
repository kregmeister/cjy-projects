#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 15:59:57 2024

@author: cjymain
"""
from technically.utils.handlers.db import DuckDB
from technically.utils.log import log_method, get_logger
from technically.utils.exceptions import (
    NotEnoughDataError,
    DeadTickerError
)

from glob import glob
import shutil
import os


class DatabaseChecks:
    """
    Deletes deadTicker data, then vacuums DuckDB to reclaim disk space.
    """

    def __init__(self, base_path: str, friday: bool):
        self.base_path = base_path
        self.friday = friday
        self.price_db = base_path + "/sql/prices.duck"
        self.fund_db = base_path + "/sql/fundamentals.duck"
        self.model_db = base_path + "/sql/models.duck"

    def execute(self):
        # Open multiple database connections simultaneously
        with DuckDB(self.price_db) as db:
            self.tickers_df = db.sql('''
                SELECT
                    CASE 
                        WHEN duplicated = False THEN ticker 
                        ELSE permaTicker 
                    END AS ticker,
                    exchange,
                    capCategory,
                    sector,
                    designation,
                    pricesInitialized,
                    fundamentalsInitialized
                FROM 
                    metadata;
                '''
            ).df()

        self.remove_dead_tickers()
        self.remove_duplicate_parquet()
        # Needs to re-connect to databases (context block for above connections closed)
        if type(self.friday) == int:
            self.vacuum_db()

    def remove_duplicate_parquet(self):
        """
        When a ticker's parquet partition changes (I.E. large cap --> mega cap), the old parquet must be removed.

        Returns:
            None
        """

        for idx, row in self.tickers_df.iterrows():
            # How parquet directories format spaces
            row['sector'] = row['sector'].replace(' ', '%20')

            for pq_type in ["daily", "quarterly"]:
                ticker_parquet_paths = glob(self.base_path + f"/parquet/{pq_type}/**/ticker={row['ticker']}", recursive=True)
                if len(ticker_parquet_paths) > 1:
                    for path in ticker_parquet_paths:
                        # Path to ticker's parquet does not match its metadata (metadata has changed)
                        if path != f"{self.base_path}/parquet/{pq_type}/exchange={row['exchange']}/cap={row['capCategory']}/sector={row['sector']}/ticker={row['ticker']}":
                            shutil.rmtree(path)
                            get_logger().info(f"Removed duplicate {pq_type} parquet for {row['ticker']} at {path}.")

    def remove_dead_tickers(self):
        """
        Removes all traces of dead tickers (except for its metadata entry).

        Returns:
            None
        """

        # Extracts tickers designated as 'dead' and ensures that have not already been removed from databases
        dead_tickers = self.tickers_df['ticker'][
            (self.tickers_df['designation'] == 'deadTicker') &
            (self.tickers_df['pricesInitialized'] == True) &
            (self.tickers_df['fundamentalsInitialized'] == True)
        ].tolist()

        for ticker in dead_tickers:
            with DuckDB(self.price_db) as db:
                # Updates its metadata entry
                db.execute('''
                    UPDATE
                        metadata
                    SET
                        pricesInitialized = false,
                        lastPricesCheck = DEFAULT,
                        fundamentalsInitialized = false,
                        lastFundamentalsCheck = DEFAULT
                    WHERE
                        CASE
                            WHEN duplicated = False THEN ticker 
                            ELSE permaTicker END = ?;
                    ''', [ticker]
                )

                # Drops price table
                db.execute(f'''
                    DROP TABLE IF EXISTS "{ticker}";
                    '''
                )

            with DuckDB(self.fund_db) as db:
                # Drops fundamentals tables
                db.execute(f'''
                    DROP TABLE IF EXISTS "{ticker}_incomeStatement";
                    '''
                )
                db.execute(f'''
                    DROP TABLE IF EXISTS "{ticker}_balanceSheet";
                    '''
                )
                db.execute(f'''
                    DROP TABLE IF EXISTS "{ticker}_cashFlow";
                    '''
                )

            with DuckDB(self.model_db) as db:
                # Drops models data
                db.execute(f'''
                    DELETE FROM indicatorSuccessRates 
                        WHERE ticker = ?;
                    ''', [ticker]
                )

            try:
                # Removes parquet files associated with ticker
                for pq_path in glob(self.base_path+f"/parquet/**/ticker={ticker}", recursive=True):
                    shutil.rmtree(pq_path)
            except Exception as e:
                get_logger().error(f"Dead ticker {ticker} could not have its parquet removed: {str(e)}")
                continue

            get_logger().info(f"Dead ticker {ticker} removed.")

    @log_method
    def vacuum_db(self):
        """
        Vacuums (in effect) all connected databases.

        Returns:
            None
        """

        for db_path in [self.price_db, self.fund_db, self.model_db]:
            # Extracts database name
            name = db_path.split("/")[-1].split(".")[0]

            # Creates new database name/path
            altered_path = db_path.replace(".duck", ".duck.old")

            # Renames existing db
            os.rename(db_path, altered_path)

            # Connects to original database path (creates new database)
            try:
                with DuckDB(db_path) as db:
                    # Attaches renamed (old) database to new database
                    db.execute(f'''
                        ATTACH '{altered_path}' 
                        AS existingdb;
                        '''
                    )
                    # Copy from existing DB to new DB (saves disk space)
                    db.execute(f'''
                        COPY FROM DATABASE existingdb 
                        TO {name};
                        '''
                    )
            except Exception:
                get_logger().error(f"Could not vacuum {name} database.")
                # If error, restores old database
                os.rename(altered_path, db_path)
                continue

            # Removes old database once complete
            os.remove(altered_path)
        return


class PriceAdjustments:
    """
    Examines price data for corporate actions (splits, dividends) and adjusts/unadjusts for them.
    """

    def __init__(self, base_path, price_df, columns_to_adjust=("open", "high", "low", "close", "volume")):
        self.price_db = base_path + "/sql/prices.duck"
        self.df = price_df
        self.columns = columns_to_adjust

    def priceActivityCheck(self, ticker: str, threshold=25000):
        """
        Cuts out data points where price activity is lower than threshold.
        Tickers are assigned different designations based on whether the most recent period is cut off ("deadTicker")
        or if the valid stretches of data are less than 120 periods ("notEnoughData").

        Args:
            ticker (str): Ticker symbol.
            thresh (int): The minimum 5-day average in dollars traded (share price * volume) for a ticker to remain active.
        """

        self.df["date"] = self.df["date"].astype(str)

        df_len = len(self.df)

        # Tickers with less than 120 periods skipped
        if df_len < 120:
            with DuckDB(self.price_db) as db:
                db.execute('''
                           UPDATE
                               metadata
                           SET designation = 'notEnoughData'
                           WHERE CASE
                                     WHEN duplicated = False THEN ticker
                                     ELSE permaTicker END = ?
                             AND isActive = true;
                           ''', [ticker]
                           )
                get_logger().warning(NotEnoughDataError(ticker, df_len))
            return False

        # Finds where a 5-day MA of dollarsTraded is less than threshold
        dollarsTraded = self.df["close"] * self.df["volume"]
        weekly_volume_sums = dollarsTraded.rolling(window=20).mean()
        violations = self.df["date"][weekly_volume_sums <= threshold].tolist()

        # All dates pass check
        if violations == []:
            return True

        # Filter violations out
        df_filtered = self.df[~self.df["date"].isin(violations)].copy()

        # Measures length of passing sequences
        index_diffs = df_filtered.index.to_series().diff().fillna(1)
        sequences = (index_diffs != 1).cumsum()

        # Filters out failing sequences
        df_filtered.loc[:, 'idx_group'] = sequences
        sequence_sizes = df_filtered.groupby("idx_group").size()

        # Filters out passing sequences less than 120
        valid_sequences = sequence_sizes[sequence_sizes > 120].index

        # Final filter
        passing_dates = tuple(
            df_filtered["date"][df_filtered['idx_group'].isin(valid_sequences)]
        )

        with DuckDB(self.price_db) as db:
            if passing_dates == ():  # No passing sequences
                db.execute('''
                           UPDATE
                               metadata
                           SET designation     = 'deadTicker',
                               lastPricesCheck = DEFAULT
                           WHERE CASE
                                     WHEN duplicated = False THEN ticker
                                     ELSE permaTicker END = ?;
                           ''', [ticker]
                           )
                get_logger().warning(DeadTickerError(ticker))
                return False
            elif passing_dates[-1] == self.df.date.values[-1]:  # Most recent data passes
                db.execute(f'''
                    DELETE FROM "{ticker}"
                    WHERE 
                        date < ?;
                    ''', [passing_dates[0]]
                           )
                get_logger().info(f"Dates preceding {passing_dates[0]} have been removed from {ticker}.")
                return False
            else:  # Some historical data passes
                db.execute(f'''
                    DELETE FROM "{ticker}"
                    WHERE date > ?;
                    ''', [passing_dates[-1]]
                           )
                db.execute(f'''
                    UPDATE 
                        metadata
                    SET 
                        designation = 'recentDatesInactive',
                        lastPricesCheck = ?
                    WHERE 
                        CASE 
                            WHEN duplicated = False THEN ticker 
                            ELSE permaTicker END = ?;
                    ''', [passing_dates[-1], ticker]
                           )
                get_logger().info(f"Dates exceeding {passing_dates[-1]} have been removed from {ticker}.")
                return False

def adjustForSplits(df):
    """
    Adjusts incoming Tiingo price data for splits, accounting for API inconsistency in applying stock splits.

    Args:
        df (pd.DataFrame): Dataframe containing incoming Tiingo price data.

    Returns:
        df (pd.DataFrame): Dataframe containing adjusted and rounded Tiingo price data.
    """
    for idx in df.index[df["splitFactor"] != 1]:
        if idx == 0:
            continue
        close = df["close"].loc[idx]

        # Close previous the split with the splitFactor applied
        prev_close = df["close"].loc[idx - 1] / df["splitFactor"].loc[idx]
        likely_split = prev_close + (prev_close * 0.25) > close > prev_close - (prev_close * 0.25)
        if not likely_split:
            continue

        for column in df.columns:  # Applies split-factor
            if column == "volume":
                df.loc[:idx - 1, column] = (
                        df.loc[:idx - 1, column] * df.loc[idx, "splitFactor"]
                ).round().astype("int64")
            elif column in ["open", "high", "low", "close"]:
                df.loc[:idx - 1, column] /= df.loc[idx, "splitFactor"]
    return df.round(4)


def unadjustForSplits(df):  # Currently unused
    for idx in df.index[df["splitFactor"] != 1]:  # Finds splits
        for column in df.columns:  # Applies split-factor
            if column == "volume":
                df.loc[:idx - 1, column] = (
                        df.loc[:idx - 1, column] / df.loc[idx, "splitFactor"]
                ).round().astype("int64")
            elif column in ["open", "high", "low", "close"]:
                df.loc[:idx - 1, column] *= df.loc[idx, "splitFactor"]
    return df.round(4)

def marketCapCategory(cap: int or float):
    """
    Assigns a market cap category for a ticker based on its most current market cap value.

    Args:
        cap: Most current market cap value.

    Returns:
        category (str): Market cap category.
    """
    try:
        if cap >= 2.0 * (10 ** 11):
            return "Mega"
        elif cap >= 1.0 * (10 ** 10):
            return "Large"
        elif cap >= 2.0 * (10 ** 9):
            return "Mid"
        elif cap >= 3.0 * (10 ** 8):
            return "Small"
        else:
            return "Micro"
    except TypeError:  # Market cap is null
        return "Unknown"

