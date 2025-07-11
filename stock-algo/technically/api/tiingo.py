#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 15:59:11 2024

@author: cjymain
"""

import requests
from requests import session
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from zipfile import ZipFile
import os
import pandas as pd
import traceback

from technically.utils.time import LimitAPICalls
from technically.utils.log import get_logger


class TiingoAPI:
    """
    Initializes requests.session object for Tiingo API calls.
    """

    def __init__(self, base_path: str, api_key):
        self.base_path = base_path
        self.api_key = api_key
        self.session = None
        self.headers = None

    def __enter__(self):
        # Limits Tiingo calls to 160 per 60 seconds (~10,000 per hour)
        self.call_limiter = LimitAPICalls(160, 60)

        # Initiates and configures https session
        self.session = session()
        self.headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Token {self.api_key}'
        }
        self._configure_session()
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.session.close()
        if exc_type is not None:
            error_code = "".join(
                traceback.format_exception(
                    exc_type, exc_value, exc_traceback
                )
            )
            get_logger().error(error_code)
            return True

    def _configure_session(self):
        """
        Configures Tiingo API session.

        Returns:
            None
        """
        # Set timeout
        self.timeout = 3
        # Create connection pool
        adapter = HTTPAdapter(
            # Exponential backoff
            max_retries=Retry(
                total=5,
                backoff_factor=0.5,
                status_forcelist=[429, 500, 502, 503, 504],
                allowed_methods=["HEAD", "GET", "OPTIONS"]
            )
        )
        self.session.mount('https://', adapter)

    def make_request(self, url: str, date_columns: list = [], rounded=True):
        """
        Makes a request to the Tiingo API.

        Args:
            url (str): The API endpoint to get data from.
            date_columns (list, optional): The fields from the API endpoint that return date strings.
                They will be converted into ISO 8601 dates. Defaults to [].
            rounded (bool, optional): Whether to round API response or not. Defaults to True.

        Returns:
            pd.DataFrame: The API response.
        """
        try:
            call = self.session.get(
                url, headers=self.headers, timeout=self.timeout
            ).json()
        except Exception as e:
            get_logger().error(f"Error making request to {url}: {str(e)}")
            return

        self.call_limiter.increment()
        if call == []:
            return

        try:
            df = pd.DataFrame.from_dict(call, orient='columns')
        except Exception as e:
            get_logger().warning(f"Request returned unexpected output. {str(e)}")
            return

        for column in date_columns:
            df[column] = pd.to_datetime(
                df[column], format='ISO8601'
            ).dt.date

        if rounded:
            df = df.round(decimals=4)

        return df

    def daily_metadata(self):
        """
        Retrieves listing metadata for all tickers from API.

        Returns:
             pd.DataFrame: API response properly formatted to pandas DataFrame.
        """
        url = ("https://api.tiingo.com/tiingo/daily/meta"
               "?columns=ticker,permaTicker,name,exchange,assetType,isActive,startDate,endDate")
        df = self.make_request(url, rounded=False)

        return df

    def fundamentals_meta(self):
        """
        Retrieves categorical metadata for all tickers from API.

        Returns:
            pd.DataFrame: API response properly formatted to pandas DataFrame.
        """
        profile_url = "https://api.tiingo.com/tiingo/fundamentals/meta"

        profile_df = self.make_request(profile_url, ["statementLastUpdated", "dailyLastUpdated"], rounded=False)

        return profile_df[
            ["permaTicker",
             "ticker",
             "sector",
             "industry",
             "sicSector",
             "sicIndustry",
             "companyWebsite",
             "statementLastUpdated",
             "dailyLastUpdated"]
        ]

    def daily_supported_tickers(self):
        """
        Retrieves updated list of supported tickers from API and saves it to CSV.

        Returns:
            None
        """
        tickers = requests.get(
            "https://apimedia.tiingo.com/docs/tiingo/daily/supported_tickers.zip"
        )

        zip_path = self.base_path + "/files/supported_tickers.zip"

        with open(zip_path, 'wb') as zf:
            zf.write(tickers.content)

        with ZipFile(zip_path) as zf:
            save_dir = self.base_path + "/files"
            zf.extractall(save_dir)

        os.remove(zip_path)

    def daily_prices(self, ticker: str, assetType: str, start_date: str):
        """
        Retrieves price data from API.

        Args:
            ticker (str): Security to get data for.
            assetType (str): 'stock' or 'etf'.
            start_date (str): The earliest date to get data for.

        Returns:
            Union[None, pd.DataFrame]:
                - None: If no data is returned from API call.
                - pd.DataFrame: If data is returned from API call.
                    Must contain price data. May contain daily fundamentals if available.
        """
        price_url = (
            f"https://api.tiingo.com/tiingo/daily/{ticker}/prices?startDate="
            f"{start_date}&columns=date,open,high,low,close,volume,splitFactor"
        )
        fund_url = (
            "https://api.tiingo.com/tiingo/fundamentals/"
            f"{ticker}/daily?startDate={start_date}"
        )

        # Obtains API price endpoint as DataFrame
        price_df = self.make_request(price_url, ["date"])

        if price_df is None:
            return

        if assetType == "stock":
            fund_df = self.make_request(fund_url, ["date"])
            if fund_df is None:
                return price_df

            try:
                merged_df = price_df.merge(fund_df, how="left", on="date")
            except Exception as e:
                get_logger().error((f"Daily prices and daily fundamentals could not be merged for ",
                                    f"{ticker}: {str(e)}"))
                return price_df

            # Converts all columns except date to float type
            dtype_conversion = {col: float for col in merged_df.columns if col not in ["date", "volume"]}
            merged_df = merged_df.astype(dtype_conversion)

            return merged_df
        else:
            return price_df

    def fundamentals_statements(self, ticker: str, start_date: str):
        """
        Retrieves income statement, balance sheet, and cash flow statement data from API (when available).

        Args:
            ticker (str): Security to get data for.
            start_date (str): The earliest date to get data for.

        Returns:
            Union[None, dict]:
                - None: If no data is returned from API call.
                - dict: If data is returned from API call.
                    Dict keys are statement names, values are statement data.

        Example:
            >>> stmt_dfs_dict = {"balanceSheet": pd.DataFrame, "cashFlow": pd.DataFrame}

        """
        statement_url = (
            "https://api.tiingo.com/tiingo/fundamentals/"
            f"{ticker}/statements?asReported=true&startDate={start_date}"
        )
        stmts_df = self.make_request(statement_url, ["date"])

        if stmts_df is None:
            return

        stmts_df = stmts_df[["date", "statementData"]][stmts_df["quarter"] != 0]

        stmt_dfs_dict = {"balanceSheet": [], "incomeStatement": [], "cashFlow": []}
        for stmt_date, stmts in stmts_df.values.tolist():
            for stmt_type in stmt_dfs_dict.keys():
                try:
                    if stmts[stmt_type] is None:
                        continue
                except KeyError:
                    continue

                raw_df = pd.DataFrame(stmts[stmt_type])
                statement_df = raw_df.pivot_table(
                    columns='dataCode', values='value'
                )

                statement_df.reset_index(drop=True, inplace=True)
                statement_df.rename_axis(None, axis=1, inplace=True)
                statement_df.insert(0, 'date', stmt_date)

                # Converts all columns except date to float type
                dtype_conversion = {col: float for col in statement_df.columns if col != "date"}
                statement_df = statement_df.astype(dtype_conversion)

                stmt_dfs_dict[stmt_type].append(statement_df)
        try:
            stmt_dfs_dict = {
                key: pd.concat(value, ignore_index=True)
                for key, value in stmt_dfs_dict.items()
            }
        except ValueError:
            get_logger().warning(f"No fundamentals found for {ticker}.")
            return

        return stmt_dfs_dict