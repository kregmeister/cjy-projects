#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 10:17:22 2024

@author: cjymain
"""

from technically.core.cleaning import (
    adjustForSplits,
    marketCapCategory
)
from technically.utils.exceptions import NoDataReturnedError
from technically.utils.log import get_logger

from duckdb import (
    InvalidInputException,
    ConstraintException,
    BinderException,
    CatalogException,
    ConversionException
)
import pandas as pd
import numpy as np
import traceback

class DataAcquisitionController:
    "Controls metadata retrieval, Tiingo data acquisition, and writing data to DuckDB."

    def __init__(self, api_session, db_cons, base_path):
        self.api = api_session
        self.dbs = db_cons
        self.base_path = base_path

    def metadata(self):
        """
        Retrieves metadata for all qualifying tickers.

        :return [prices_metadata, fundamentals_metadata]: Two tuples containing metadata for prices and fundamentals, respectively.
        """

        prices_metadata = self.dbs.sql['prices']('''
            SELECT
                CASE 
                    WHEN duplicated = False THEN ticker 
                    ELSE permaTicker 
                END AS ticker,
                exchange,
                capCategory,
                sector,
                assetType,
                lastPricesCheck,
                dailyLastUpdated,
                pricesInitialized
            FROM 
                metadata
            WHERE 
                designation = 'active'
                OR 
                    designation != 'deadTicker' 
                    AND pricesInitialized = false;
            '''
        ).fetchall()

        fundamentals_metadata = self.dbs.sql['prices']('''
            SELECT
                CASE 
                    WHEN duplicated = False THEN ticker 
                    ELSE permaTicker 
                END AS ticker,
                exchange,
                capCategory,
                sector,
                lastFundamentalsCheck,
                statementLastUpdated,
                fundamentalsInitialized
            FROM 
                metadata
            WHERE 
                designation = 'active'
                AND assetType = 'stock'
                AND statementLastUpdated IS NOT NULL
                OR 
                    designation != 'active'
                    AND statementLastUpdated IS NOT NULL
                    AND fundamentalsInitialized = false;
            '''
        ).fetchall()

        return [prices_metadata, fundamentals_metadata]

    def models(self, group_by="sector"):
        """
        Retrieves technical indicator success rates or creates the database table if it doesn't exist.

        :param group_by: Which column to group success rates by. Defaults to sector.

        :return indicator_success_rates: DataFrame of indicator success rates.:
        """
        if not self.dbs.has_table("indicatorSuccessRates", db_name="models"):
            get_logger().info("No indicator success rate table found. Creating empty table")
            self.dbs.execute['models']('''
                CREATE TABLE indicatorSuccessRates (
                    date DATE, 
                    ticker VARCHAR, 
                    sector VARCHAR,
                    cap VARCHAR,
                    exchange VARCHAR,
                    indicator VARCHAR,
                    signalName VARCHAR,
                    reversalCount INTEGER,
                    reversalSuccess INTEGER,
                    reversalFailure INTEGER,
                    continuationCount INTEGER,
                    continuationSuccess INTEGER,
                    continuationFailure INTEGER,
                    PRIMARY KEY (ticker, signalName)
                );
                '''
            )

        indicator_success_rates = self.dbs.sql['models']('''
            SELECT 
                sector, 
                signalName, 
                AVG(reversalSuccess / reversalCount) AS reversalSuccessRate, 
                AVG(continuationSuccess / continuationCount) AS continuationSuccessRate 
            FROM 
                indicatorSuccessRates 
            GROUP BY 
                sector, 
                signalName;
            '''
        ).df()

        if indicator_success_rates.empty:
            get_logger().warning(
                "No indicator success rates found. Indicator scores will be defaults until backtesting is conducted."
            )
            return indicator_success_rates

        bearish_mask = indicator_success_rates["signalName"].str.startswith("bearish")

        indicator_success_rates.loc[bearish_mask, ["reversalSuccessRate", "continuationSuccessRate"]] = \
            indicator_success_rates[["reversalSuccessRate", "continuationSuccessRate"]].apply(lambda x: x * -1)

        # Fills in nulls and infinite values with 0.0
        indicator_success_rates.fillna(0.0, inplace=True)
        indicator_success_rates.replace([np.inf, -np.inf], 0.0, inplace=True)

        return indicator_success_rates

    def prices(self, ticker: str, today: str, exchange: str, capCategory: str, sector: str, assetType: str, latestData: str, initialized: bool):
        """
        Handles the acquisition of price data and writes it to the database.

        :param ticker:
        :param today:
        :param exchange:
        :param capCategory:
        :param sector:
        :param assetType:
        :param latestData:
        :param initialized:

        :return:
        """
        # Obtain the data from Tiingo
        try:
            daily_df = self.api.daily_prices(
                ticker, assetType, latestData
            )
        except Exception as e:
            get_logger().error(f"Unexpected error retreiving price data for {ticker}: {traceback.format_exc()}")
            return

        if daily_df is None:
            get_logger().error(NoDataReturnedError("prices", ticker))
            return

        # Determines cap (if data available)
        if "marketCap" in daily_df.columns:
            current_cap = daily_df["marketCap"].iloc[-1]
            capCategory = marketCapCategory(current_cap)

        # Casts volume column as integer
        daily_df["volume"] = daily_df["volume"].astype(int)

        if not initialized:  # Table doesn't exist
            # Fully adjust the newly acquired price data (if a split is present)
            if any(daily_df["splitFactor"] != 1.0):
                daily_df = adjustForSplits(daily_df.copy())

            try:
                self.dbs.execute['prices'](f'''
                    CREATE TABLE "{ticker}" AS 
                        SELECT * FROM daily_df;
                    '''
                )
            except CatalogException:
                get_logger().warning(f"Prices table for {ticker} already exists. "
                                     "Table will be deleted to allow for proper initialization.")
                self.dbs.sql['prices'](f'''
                    DROP TABLE IF EXISTS "{ticker}";
                    '''
                )
                return

            self.dbs.execute['prices'](f'''
                ALTER TABLE "{ticker}"
                    ADD PRIMARY KEY (date);
                '''
            )
        else:  # Table exists
            df_cols = ", ".join(daily_df.columns)

            try:
                for i, sf in enumerate(daily_df["splitFactor"]):
                    # Adjusts existing ticker's table for a new split (if it's present)
                    if sf != 1.0:
                        self.dbs.execute['prices'](f'''
                            UPDATE 
                                "{ticker}"
                            SET 
                                open = (open / {sf}),
                                high = (high / {sf}),
                                low = (low / {sf}),
                                close = (close / {sf}),
                                volume = (volume * {sf});
                            '''
                        )

                self.dbs.execute['prices'](f'''
                    INSERT INTO "{ticker}" ({df_cols})
                        SELECT 
                            *
                        FROM 
                            daily_df
                        WHERE 
                            daily_df.date NOT IN (
                                SELECT 
                                    date 
                                FROM 
                                    "{ticker}"
                            );
                    '''
                )
            except ConversionException as e:
                # Isolates the problematic value from the error message
                problematic_value = float(str(e).split("value")[1].split("can")[0].strip())

                # Finds out which column the problematic value is in
                occurences = daily_df.isin([problematic_value]).any()
                problematic_col = occurences[occurences == True].index[0]

                get_logger().error(f"Conversion error for {ticker}. Value {problematic_value} in column {problematic_col} cannot be converted to INT32.")
                return
            except BinderException:
                get_logger().info(f"Table for {ticker} likely does not yet have daily fundamentals columns. Adding them now.")
                self.dbs.add_missing_columns_to_table(
                    ticker,
                    list(daily_df.columns),
                    db_name="prices"
                )
                return
            except CatalogException:
                get_logger().warning(f"{ticker} is marked as initialized but does not have a prices table. Addressing.")
                # Updates metadata so that ticker will be treated as un-initialized next time
                self.dbs.execute['prices']('''
                    UPDATE 
                        metadata
                    SET
                        lastPricesCheck = DEFAULT,
                        pricesInitialized = false
                    WHERE
                        CASE 
                            WHEN duplicated = False THEN ticker 
                            ELSE permaTicker END = ?;
                    ''', [ticker]
                )
                return

        latest_date = daily_df["date"].iloc[-1]

        self.dbs.execute['prices'](f'''
            UPDATE 
                metadata 
            SET
                capCategory = ?,
                lastPricesCheck = ?,
                pricesInitialized = true,
            WHERE 
                CASE 
                    WHEN duplicated = False THEN ticker 
                    ELSE permaTicker END = ?;
            ''', [capCategory, latest_date, ticker]
        )
        return True

    def fundamentals(self, ticker: str, today: str, start_date: str, initialized: bool):
        try:
            quarterly_dfs = self.api.fundamentals_statements(
                ticker, start_date
            )
        except Exception as e:
            get_logger().error(f"Unexpected error retreiving statement data for {ticker}: {traceback.format_exc()}")
            return

        # If no data returned, sets check to constant so its not re-checked daily (resets every Friday)
        if quarterly_dfs is None:
            self.dbs.execute['prices']('''
                UPDATE 
                    metadata
                SET 
                    lastFundamentalsCheck = '3000-01-01'
                WHERE 
                    CASE 
                        WHEN duplicated = False THEN ticker 
                        ELSE permaTicker END = ?;
                ''', [ticker]
            )
            get_logger().warning(NoDataReturnedError("fundamentals", ticker))
            return

        # Writes each statement df to its own table
        for stmt_type, df in quarterly_dfs.items():
            if df is None:
                continue

            df = df.sort_values("date")
            df_cols = ", ".join(df.columns)
            table_name = f"{ticker}_{stmt_type}"

            if not initialized:  # If not in system
                try:
                    self.dbs.execute['fundamentals'](f'''
                        CREATE OR REPLACE TABLE "{table_name}" AS
                            SELECT * FROM df;
                        '''
                    )
                except CatalogException:
                    get_logger().warning(f"Fundamentals table for {ticker} already exists. ",
                                         "Tables will be deleted to allow for proper initialization.")
                    self.dbs.execute['prices']('''
                        UPDATE 
                            metadata
                        SET
                            lastFundamentalsCheck = DEFAULT
                        WHERE
                            CASE 
                                WHEN duplicated = False THEN ticker 
                                ELSE permaTicker END = ?;
                        ''', [ticker]
                    )
                    self.dbs.execute['fundamentals'](f'''
                        DROP TABLE IF EXISTS "{table_name}";
                        '''
                    )
                    continue

                self.dbs.execute['fundamentals'](f'''
                    ALTER TABLE "{table_name}"
                        ADD PRIMARY KEY (date);
                    '''
                )
            else:  # In system
                try:
                    self.dbs.execute['fundamentals'](f'''
                        INSERT INTO "{table_name}" ({df_cols})
                            SELECT 
                                *
                            FROM 
                                df
                            WHERE df.date NOT IN (
                                SELECT 
                                    date
                                FROM 
                                    "{table_name}"
                            );
                        '''
                    )
                # Triggered when a ticker does not have data for stmt_type
                except (InvalidInputException, ConstraintException, CatalogException) as e:
                    if str(e) == f"Table with name {table_name} does not exist!":
                        get_logger().warning(f"{table_name} for {ticker} does not exist. Setting fundamentalsInitialized to false.")
                        # Resets fundamentals metadata so that statements, likely newly available to Tiingo, can be fully gathered
                        self.dbs.execute['prices']('''
                            UPDATE 
                                metadata
                            SET 
                                lastFundamentalsCheck = '2010-01-01',
                                fundamentalsInitialized = false
                            WHERE 
                                CASE 
                                    WHEN duplicated = False THEN ticker 
                                    ELSE permaTicker END = ?;
                            ''', [ticker]
                        )
                    else:
                        get_logger().warning(f"Unexpected error appending to table {table_name}: {traceback.format_exc()}")
                    continue
                except BinderException:  # Mismatched col
                    get_logger().info(f"Statement df columns do not match columns in {table_name}. "
                                      "Adding them.")
                    self.dbs.add_missing_columns_to_table(
                        table_name,
                        list(df.columns),
                        db_name="fundamentals"
                    )
                    continue

        self.dbs.execute['prices']('''
            UPDATE 
                metadata
            SET 
                lastFundamentalsCheck = ?,
                fundamentalsInitialized = true
            WHERE 
                CASE 
                    WHEN duplicated = False THEN ticker 
                    ELSE permaTicker END = ?;
            ''', [today, ticker]
        )
        return True
