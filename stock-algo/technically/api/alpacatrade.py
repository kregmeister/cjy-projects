#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 15:26:06 2024

@author: cjymain
"""
import traceback
from alpaca.trading.client import TradingClient

from technically.utils.handlers.auth import get_credentials
from technically.utils.log import get_logger


class AlpacaTradeAPI:
    """
    Manages a TradingClient session to place orders, track P/L, etc.
    """

    def __init__(self, base_path: str):
        self.base_path = base_path

    def __enter__(self):
        key, sec = get_credentials(["alpaca_api_key", "alpaca_secret_token"])
        self.client = TradingClient(key, sec, paper=True)
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

    def get_portfolio(self):
        """
        Extracts detailed portfolio data from Alpaca API.

        Returns:
            pd.DataFrame
        """
        positions = self.client.get_all_positions()
        return positions
