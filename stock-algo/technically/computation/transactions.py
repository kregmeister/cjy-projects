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
from technically.utils.handlers.db import DuckDB
from technically.utils.log import get_logger

from duckdb import (
    InvalidInputException,
    ConstraintException,
    BinderException,
    CatalogException,
    ConversionException
)
import numpy as np
import traceback

class DataAcquisitionController:
    """
    Controls metadata retrieval, Tiingo data acquisition, and writing data to DuckDB.
    """

    def __init__(self, base_path):
        self.base_path = base_path
        self.price_db = base_path + "/sql/prices.duck"
        self.fund_db = base_path + "/sql/fundamentals.duck"
        self.model_db = base_path + "/sql/models.duck"

    def metadata(self):
        """
        Retrieves metadata for all qualifying tickers.

        Returns:
            [prices_metadata, fundamentals_metadata]: Two tuples containing metadata for prices and fundamentals, respectively.
        """

        with DuckDB(self.price_db) as db:
            prices_metadata = db.sql('''
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

            fundamentals_metadata = db.sql('''
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

        Args:
            group_by (str): Which column to group success rates by. Default is sector.

        Returns:
            indicator_success_rates: DataFrame of indicator success rates.
        """

        with DuckDB(self.model_db) as db:
            if not db.has_table("indicatorSuccessRates"):
                get_logger().info("No indicator success rate table found. Creating empty table")
                db.execute('''
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

            indicator_success_rates = db.sql(f'''
                SELECT 
                    sector, 
                    signalName, 
                    AVG(reversalSuccess / reversalCount) AS reversalSuccessRate, 
                    AVG(continuationSuccess / continuationCount) AS continuationSuccessRate 
                FROM 
                    indicatorSuccessRates 
                GROUP BY 
                    {group_by}, 
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

    def prices(self, api_session, ticker: str, today: str, exchange: str, capCategory: str, sector: str, assetType: str, latestData: str, initialized: bool):
        """
        Handles the ingestion of price data, formats it, and writes it to DuckDB.

        Args:
            api_session: The active HTTPS session used for making API calls.
            ticker (str): Ticker symbol.
            today (str): Today's date in YYYY-MM-DD string format.
            exchange: Ticker exchange.
            capCategory (str): Ticker market cap category.
            sector (str): Ticker sector.
            assetType (str): Ticker asset type. Can equal "stock" or "etf".
            latestData (str): Latest data stored in database for ticker, formatted as YYYY-MM-DD.
            initialized (bool): Indicates if the database table should be initialized.

        Returns:
            Union[None, True]:
                - None: Any variety of errors/inconsistencies occurred.
                - True: Indicates the data transmission was successful.
        """

        # Obtain the data from Tiingo
        try:
            daily_df = api_session.daily_prices(
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

        with DuckDB(self.price_db) as db:
            if not initialized:  # Table doesn't exist
                # Fully adjust the newly acquired price data (if a split is present)
                if any(daily_df["splitFactor"] != 1.0):
                    daily_df = adjustForSplits(daily_df.copy())

                try:
                    db.execute(f'''
                        CREATE TABLE "{ticker}" AS 
                            SELECT * FROM daily_df;
                        '''
                    )
                except CatalogException:
                    get_logger().warning(f"Prices table for {ticker} already exists. "
                                         "Table will be deleted to allow for proper initialization.")
                    db.sql(f'''
                        DROP TABLE IF EXISTS "{ticker}";
                        '''
                    )
                    return

                db.execute(f'''
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
                            db.execute(f'''
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

                    db.execute(f'''
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
                    db.add_missing_columns_to_table(
                        ticker,
                        list(daily_df.columns)
                    )
                    return
                except CatalogException:
                    get_logger().warning(f"{ticker} is marked as initialized but does not have a prices table. Addressing.")
                    # Updates metadata so that ticker will be treated as un-initialized next time
                    db.execute('''
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

            db.execute(f'''
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

    def fundamentals(self, api_session, ticker: str, today: str, start_date: str, initialized: bool):
        """
        Handles the ingestion of fundamental statements data, formats it, and writes it to DuckDB.

        Args:
            api_session: The active HTTPS session used for making API calls.
            ticker (str): Ticker symbol.
            today (str): Today's date in YYYY-MM-DD string format.
            start_date (str): The earliest date to search for statements for ticker, formatted as YYYY-MM-DD.
            initialized (bool): Indicates if the database table should be initialized.

        Returns:
            Union[None, True]:
                - None: Any variety of errors/inconsistencies occurred.
                - True: Indicates the data transmission was successful.
        """
        try:
            quarterly_dfs = api_session.fundamentals_statements(
                ticker, start_date
            )
        except Exception as e:
            get_logger().error(f"Unexpected error retreiving statement data for {ticker}: {traceback.format_exc()}")
            return

        # If no data returned, sets check to constant so it's not re-checked daily (resets every Friday)
        if quarterly_dfs is None:
            with DuckDB(self.price_db) as db:
                db.execute('''
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

            with DuckDB(self.fund_db) as db:
                if not initialized:  # If not in system
                    try:
                        db.execute(f'''
                            CREATE OR REPLACE TABLE "{table_name}" AS
                                SELECT * FROM df;
                            '''
                        )
                    except CatalogException:
                        get_logger().warning(f"Fundamentals table for {ticker} already exists. ",
                                             "Tables will be deleted to allow for proper initialization.")
                        db.execute('''
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
                        db.execute(f'''
                            DROP TABLE IF EXISTS "{table_name}";
                            '''
                        )
                        continue

                    db.execute(f'''
                        ALTER TABLE "{table_name}"
                            ADD PRIMARY KEY (date);
                        '''
                    )
                else:  # In system
                    try:
                        db.execute(f'''
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
                            db.execute('''
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
                        db.add_missing_columns_to_table(
                            table_name,
                            list(df.columns)
                        )
                        continue

            db.execute('''
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
