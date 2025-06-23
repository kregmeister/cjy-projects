#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 15:33:42 2024

@author: cjymain
"""
import yfinance as yf
from yfinance.exceptions import YFDataException, YFRateLimitError
from curl_cffi.requests.exceptions import HTTPError as curlHTTPError
from requests.exceptions import HTTPError as requestsHTTPError
import pandas as pd
import json
import traceback

from technically.utils.time import LimitAPICalls
from technically.utils.log import get_logger


class YFinanceAPI:
    "Initializes requests.session object for Tiingo API calls."

    def __init__(self, base_path: str):
        self.base_path = base_path
        self.session = None
        self.headers = None
        # Limits yfinance calls to 2 per second (~7,200 per hour)
        self.call_limiter = LimitAPICalls(2, 1)
        
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if exc_type is not None:
            error_code = "".join(
                traceback.format_exception(
                    exc_type, exc_value, exc_traceback
                )
            )
            get_logger().error(error_code)
            return True

    def fund_profile(self, ticker: str):
        try:
            data = yf.Ticker(ticker)
            overview = data.funds_data.fund_overview

            netAssets = data.info['netAssets']
            fundFamily = overview['family']
            fundCategory = overview['categoryName']
        except (YFDataException, requestsHTTPError, curlHTTPError, KeyError, YFRateLimitError) as e:
            get_logger().warning(f"No fund overview found for {ticker}: {str(e)}")
            # fundCategory is a partition column in parquet, cannot be None
            return ["Unknown", "Unknown", "Unknown"]

        # Makes 2 api calls
        self.call_limiter.increment(i=2)

        return [netAssets, fundFamily, fundCategory]