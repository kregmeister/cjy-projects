#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 15:59:57 2024

@author: cjymain
"""
from technically.utils.handlers.duckdb import DuckDB
from technically.utils.log import log_method, get_logger
from technically.utils.log import get_logger
from technically.utils.exceptions import (
    NotEnoughDataError,
    DeadTickerError
)

from glob import glob
import shutil
import os


class DatabaseChecks:
    "Deletes deadTicker data, then vacuums DuckDB to reclaim disk space."
    
    def __init__(self, base_path: str, friday: bool):
        self.base_path = base_path
        self.friday = friday
        
        self.db_paths = [
            base_path+"/sql/prices.duck", 
            base_path+"/sql/fundamentals.duck",
            base_path+"/sql/models.duck",
        ]
    
    def execute(self):
        # Open multiple database connections simultaneously
        with DuckDB(db_path=self.db_paths, read_only=[False, False, False], mode="multi") as self.dbs:
            self.tickers_df = self.dbs.sql['prices']('''
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
        "When a ticker's parquet partition changes (I.E. large cap --> mega cap), the old parquet must be removed."

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
        "Removes all traces of dead tickers (except for its metadata entry)."

        # Extracts tickers designated as 'dead' and ensures that have not already been removed from databases
        dead_tickers = self.tickers_df['ticker'][
            (self.tickers_df['designation'] == 'deadTicker') &
            (self.tickers_df['pricesInitialized'] == True) &
            (self.tickers_df['fundamentalsInitialized'] == True)
        ].tolist()
        
        for ticker in dead_tickers:
            self.dbs.execute['prices']('''
                SELECT
                    ''')

            # Updates its metadata entry
            self.dbs.execute['prices']('''
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
            self.dbs.execute['prices'](f'''
                DROP TABLE IF EXISTS "{ticker}";
                '''
            )
            # Drops fundamentals tables
            self.dbs.execute['fundamentals'](f'''
                DROP TABLE IF EXISTS "{ticker}_incomeStatement";
                '''
            )
            self.dbs.execute['fundamentals'](f'''
                DROP TABLE IF EXISTS "{ticker}_balanceSheet";
                '''
            )
            self.dbs.execute['fundamentals'](f'''
                DROP TABLE IF EXISTS "{ticker}_cashFlow";
                '''
            )
            # Drops models data
            self.dbs.execute['models'](f'''
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
        Vacuums all connected databases.
        NOTE: Due to how this method handles connections, 
        it must be the final method called for the class instance.
        """
        
        for db_path in self.db_paths:
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
    "Examines price data for corporate actions (splits, dividends) and adjusts/unadjusts for them."

    def __init__(self, price_df, columns_to_adjust=["open", "high", "low", "close", "volume"]):
        self.df = price_df
        self.columns = columns_to_adjust

    def priceActivityCheck(self, ticker: str, db, thresh=25000):
        """
        First, dollarsTraded is calculated via close * volume.
        If at some point there is a 5 day stretch the average dollarsTraded less than the threshold, trading is too inconsistent.
        The most recent date where this holds true is where the dataframe is cut at.
        If a DataFrame's len is less than 120, the ticker is designated as 'notEnoughData'.
        If the dataframe is cut at the most recent data point, the ticker is designated as 'deadTicker'.

        """

        self.df["date"] = self.df["date"].astype(str)

        df_len = len(self.df)

        # Tickers with less than 120 periods skipped
        if df_len < 120:
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
        violations = self.df["date"][weekly_volume_sums <= thresh].tolist()

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
    Tiingo data sometimes applies adjustments and sometimes doesn't.
    If the close previous the split, with the splitFactor applied, is not within 25% of the close when the split occurs,
    Then the split was more than likely already applied by Tiingo.
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

        for column in df.columns:  # Applies split factor
            if column == "volume":
                df.loc[:idx - 1, column] = (
                        df.loc[:idx - 1, column] * df.loc[idx, "splitFactor"]
                ).round().astype("int64")
            elif column in ["open", "high", "low", "close"]:
                df.loc[:idx - 1, column] /= df.loc[idx, "splitFactor"]
    return df.round(4)


def unadjustForSplits(df):
    for idx in df.index[df["splitFactor"] != 1]:  # Finds splits
        for column in df.columns:  # Applies split factor
            if column == "volume":
                df.loc[:idx - 1, column] = (
                        df.loc[:idx - 1, column] / df.loc[idx, "splitFactor"]
                ).round().astype("int64")
            elif column in ["open", "high", "low", "close"]:
                df.loc[:idx - 1, column] *= df.loc[idx, "splitFactor"]
    return df.round(4)


# Categories for market capitalization
def marketCapCategory(cap: int or float):
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

