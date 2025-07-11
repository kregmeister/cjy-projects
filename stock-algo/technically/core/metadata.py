#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 18:02:18 2023

@author: craigyingling321
"""

import pandas as pd

from technically.utils.handlers.auth import get_credentials
from technically.utils.handlers.db import DuckDB
from technically.api.tiingo import TiingoAPI
from technically.utils.log import log_class
from technically.api.yahoofinance import YFinanceAPI
from technically.core.cleaning import marketCapCategory


@log_class(critical=True)
class ManageMetadata:
    """
    Acquires metadata for all tickers and updates/initializes metadata table.
    """

    def __init__(self, base_path, init, friday, dev=False):
        self.base_path = base_path
        self.init = init
        self.friday = friday
        self.is_dev = dev
        # Retrieves API credentials once
        self.tiingo_credentials = get_credentials(["tiingo_api_key"])

    def execute(self):
        tickers_df = self.get_metadata()

        if self.init:
            self.initialize_metadata(tickers_df)
        else:
            self.update_metadata(tickers_df)

    def get_metadata(self):
        """
        Acquires and ingests ticker metadata, updating it when/where necessary.

        Returns:
            df (pd.DataFrame): Full, formatted metadata from Tiingo.
        """
        # Retrieves dataframe of ticker metadata
        with TiingoAPI(self.base_path, self.tiingo_credentials) as api:
            df = api.daily_metadata()
            profile_df = api.fundamentals_meta()

        included_exchanges = ["PINK", "OTCGREY", "EXPM", "OTC", "NYSE", "NASDAQ", "ARCA",
                              "OTCBB", "OTCQB", "OTCCE", "OTCMKTS", "OTCQX", "NYSE ARCA", "NYSE MKT", "BATS"]

        # Development tests only include a small subset of tickers
        if self.is_dev:
            df = df[
                (df["exchange"].isin(included_exchanges)) &
                (df["assetType"].isin(["Stock", "ETF"])) &
                (df["startDate"].notnull()) &
                (df["startDate"] != df["endDate"]) &
                (df["ticker"].isin(
                    [
                        "ally", "bk", "cfg", "jpm", "mtb", "pnc", "td", "tfc",
                        "wfc", "bac", "c", "gs", "ms", "usb", "cof", "schw",
                        "stt", "bmo", "axp", "hsbc", "fitb", "bcs", "key", "amp",
                        "nee", "gev", "fslr", "nxt", "cwen", "bepc", "ora", "be",
                        "run", "plug", "flnc", "amps", "arry", "rex", "shls", "amrc",
                        "mntk", "gevo", "gpre", "ff", "vgas", "spwr", "cslr", "nrgv", "actv"
                    ]
                ))
            ]
        else:
            # Cleans Tiingo metadata dataframe
            df = df[
                (df["exchange"].isin(included_exchanges)) &
                (df["assetType"].isin(["Stock", "ETF"])) &
                (df["startDate"].notnull()) &
                (df["startDate"] != df["endDate"])
            ]

        # Ensures most recently traded ticker is listed ahead of duplicates
        df.sort_values(["ticker", "startDate"], ascending=[True, False], inplace=True)
        df["duplicated"] = df.duplicated("ticker")

        # Map for renaming exchanges
        consolidations = {
            "OTC": ["OTCBB", "OTCQB", "OTCCE", "OTCMKTS", "OTCQX"],
            "ARCA": ["NYSE ARCA", "NYSE MKT", "BATS"],
            "OTCPINK": ["PINK"],
            "OTCEXPM": ["EXPM"]
        }

        def map_to_key(value, dict_map):
            for key, values in dict_map.items():
                if value in values:
                    return key
            return value

        # Consolidates exchanges and lowercases exchange & assetType columns
        df["exchange"] = df["exchange"].apply(
            map_to_key, args=(consolidations,)
        ).str.lower()
        df["assetType"] = df["assetType"].str.lower()

        # Sets date columns to date format (comes from Tiingo as string)
        df["startDate"] = pd.to_datetime(df["startDate"], format="ISO8601").dt.date
        df["endDate"] = pd.to_datetime(df["endDate"], format="ISO8601").dt.date

        # Calculates tenure using startDate and endDate columns
        def difference_in_years(start, end):
            return [e.year - s.year for e, s in zip(end, start)]

        df["yearsListed"] = difference_in_years(df["startDate"], df["endDate"])

        # Maps designation to active/dead
        df["designation"] = df["isActive"].apply(lambda x: "active" if x == True else "delisted")

        df = pd.merge(df, profile_df, how="left", on=["permaTicker", "ticker"])
        return df

    def update_metadata(self, df):
        """
        Updates metadata table only where necessary using the existing table and the new data from df.

        Args:
            df: Full, formatted metadata from Tiingo.

        Returns:
            None
        """
        with DuckDB(self.base_path + "/sql/prices.duck") as db:
            if type(self.friday) == int:
                # Resets data designations on Friday so that they can be re-checked on Monday
                db.sql('''
                    UPDATE 
                        metadata 
                    SET 
                        lastFundamentalsCheck = DEFAULT 
                    WHERE 
                        lastFundamentalsCheck = '3000-01-01';
                        AND 
                            designation != 'deadTicker'
                            AND isActive = true;
                    '''
                )
                db.sql('''
                    UPDATE
                        metadata
                    SET 
                        designation = 'active'
                    WHERE
                        designation = 'notEnoughData'
                        OR 
                            designation = 'deadTicker'
                            AND isActive = true;
                    '''
                )

            # Updates attributes that could be subject to change for existing tickers
            db.sql('''
                UPDATE 
                    metadata 
                SET
                    endDate = df.endDate,
                    isActive = df.isActive,
                    yearsListed = df.yearsListed,
                    companyWebsite = df.companyWebsite,
                    statementLastUpdated = df.statementLastUpdated,
                    dailyLastUpdated = df.dailyLastUpdated
                FROM 
                    df
                WHERE 
                    metadata.permaTicker = df.permaTicker
                    AND metadata.designation = 'active';
                '''
            )

            # Adds rows for new tickers
            columns = ", ".join(df.columns)
            db.sql(f'''
                INSERT INTO metadata ({columns}) 
                    SELECT 
                        * 
                    FROM 
                        df
                    WHERE
                        df.permaTicker NOT IN 
                        (SELECT metadata.permaTicker FROM metadata);
                '''
            )

            # Deletes rows for tickers that are no longer listed in Tiingo
            db.sql('''
                DELETE FROM metadata
                    WHERE
                        metadata.permaticker NOT IN
                        (SELECT df.permaTicker FROM df);
                '''
            )

        self.update_etf_metadata()
        return

    def update_etf_metadata(self):
        """
        Attempts to gather details for ETF's from Yahoo Finance if it has not yet been acquired.

        Returns:
            None
        """
        with DuckDB(self.base_path + "/sql/prices.duck") as db:
            # Gathers all ETFs without category data
            etf_lst = db.sql('''
                SELECT 
                    CASE 
                        WHEN duplicated = False THEN ticker 
                        ELSE permaTicker 
                    END AS ticker,
                FROM 
                    metadata 
                WHERE 
                    assetType = 'etf' 
                    AND isActive = true 
                    AND etfInfoInitialized = false;
                '''
            ).fetchall()

            # Opens Yahoo Finance API session
            with YFinanceAPI(self.base_path) as api:
                for (ticker,) in etf_lst:
                    # Attempts to gather category data
                    netAssets, family, category = api.fund_profile(ticker)
                    if type(netAssets) == float:
                        capCategory = marketCapCategory(netAssets)
                    else:
                        capCategory = "Unknown"

                    # Writes category data to metadata table
                    db.execute('''
                        UPDATE 
                            metadata
                        SET
                            etfInfoInitialized = true,
                            capCategory = ?,
                            sector = ?,
                            industry = ?,
                            sicSector = ?,
                            sicIndustry = ?
                        WHERE
                            CASE
                                WHEN duplicated = False THEN ticker
                                ELSE permaTicker END = ?
                        ''',
                        [capCategory, category, family, category, family, ticker]
                    )
        return

    def initialize_metadata(self, df):
        """
        Process for creating metadata table if it does not yet exist.

        Args:
            df: Full, formatted metadata from Tiingo.

        Returns:

        """
        with DuckDB(self.base_path + "/sql/prices.duck") as db:
            columns = ", ".join(df.columns)
            # Creates metadata table
            db.sql('''
                CREATE TABLE metadata(
                    permaTicker VARCHAR, 
                    ticker VARCHAR,
                    duplicated BOOLEAN,
                    \"name\" VARCHAR, 
                    exchange VARCHAR, 
                    assetType VARCHAR,
                    capCategory VARCHAR DEFAULT('Unknown'),
                    sector VARCHAR DEFAULT('Unknown'),
                    industry VARCHAR, 
                    sicSector VARCHAR, 
                    sicIndustry VARCHAR,
                    companyWebsite VARCHAR,
                    isActive BOOLEAN,
                    startDate DATE, 
                    endDate DATE,  
                    yearsListed BIGINT,
                    designation VARCHAR,
                    lastPricesCheck DATE DEFAULT('1990-01-01'), 
                    lastFundamentalsCheck DATE DEFAULT('1990-01-01'),
                    statementLastUpdated DATE, 
                    dailyLastUpdated DATE,   
                    pricesInitialized BOOLEAN DEFAULT(CAST('f' AS BOOLEAN)),  
                    fundamentalsInitialized BOOLEAN DEFAULT(CAST('f' AS BOOLEAN)),
                    etfInfoInitialized BOOLEAN DEFAULT(CAST('f' AS BOOLEAN)),
                    PRIMARY KEY(permaTicker)
                );
                '''
            )
            # Populates metadata table
            db.sql(f'''
                INSERT INTO metadata ({columns})
                    SELECT * FROM df;
                '''
            )

            # Corrects behavior where statementLastUpdated is re-cast as a VARCHAR, not DATE on update/insert
            db.sql('''
                ALTER TABLE metadata
                    ALTER statementLastUpdated TYPE DATE;
                '''
            )

        self.update_etf_metadata()
        return
