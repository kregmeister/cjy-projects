#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 13:45:35 2023

@author: cjymain
"""

from datetime import timedelta, date

from technically.api.tiingo import TiingoAPI
from technically.computation.transactions import DataAcquisitionController
from technically.computation.calculations import TechnicalsExecutor, FundamentalsExecutor
from technically.computation.breadth import BreadthExecutor

from technically.utils.handlers.auth import get_credentials
from technically.utils.log import log_method
from technically.utils.handlers.queue import CalculationsOrchestrator


class DataAcquisitionCoordinator:
    """
    Coordinates data acquisition and analysis for each ticker.
    """

    def __init__(self, base_path, today, regen=False):
        self.base_path = base_path
        self.today = today
        # When true, new data is not acquired and parquets are regenerated
        self.regen = regen
        # Pass api session and database connections to child classes
        self.handler = DataAcquisitionController(
            self.base_path,
        )

        # Retrieves ticker metadata from databases
        self.prices_metadata, self.fundamentals_metadata = self.handler.metadata()

        # Retrieves backtesting metrics to apply in technical analysis
        self.indicator_success_rates = self.handler.models()

        # Retrieves API credentials once
        self.tiingo_credentials = get_credentials(["tiingo_api_key"])

    def execute(self):
        # Retrieves ticker price data from Tiingo
        self.get_prices()
        self.price_queue.stop_workers_signal()

        # Retrieves ticker fundamental statements from Tiingo
        self.get_statements()
        self.fund_queue.stop_workers_signal()

        # Calculates breadth metrics for all tickers
        BreadthExecutor(self.base_path).execute()

    @log_method(critical=True)
    def get_prices(self):
        """
        Parent-level coordinator for ticker-by-ticker price data acquisition,
        cleaning, technical analysis, indicator scoring, and feature engineering.

        Returns:
            None
        """
        # Initializes queue for technical calculations
        self.price_queue = CalculationsOrchestrator(
            TechnicalsExecutor,
            self.base_path,
            self.base_path + "/sql/prices.duck",
            self.indicator_success_rates
        )

        with TiingoAPI(self.base_path, self.tiingo_credentials) as api:
            # Loops through every ticker's metadata
            for ticker, exchange, capCategory, sector, assetType, lastDailyCheck, dailyLastUpdated, isInitialized in self.prices_metadata:
                # Prevents retrieval of existing dates from Tiingo
                if isInitialized:
                    lastDailyCheck = lastDailyCheck + timedelta(days=1)

                if not self.regen:  # Skips data acquisition
                    # Acquires data, cleans it, and writes it to database
                    resp = self.handler.prices(
                        api,
                        ticker,
                        self.today,
                        exchange,
                        capCategory,
                        sector,
                        assetType,
                        lastDailyCheck,
                        isInitialized
                    )
                    if resp is None:  # No new data/error
                        continue  # No technical analysis

                # Formats required inputs for technical calculations queue
                q_lst = [ticker, exchange, capCategory, sector]
                self.price_queue.add_ticker(q_lst)  # Places in queue
        return

    @log_method(critical=True)
    def get_statements(self):
        """
        Parent-level coordinator for ticker-by-ticker statements data acquisition,
        cleaning, fundamental analysis, and feature engineering.

        Returns:
            None
        """
        # Queue for near-real time fundamentals calculations
        self.fund_queue = CalculationsOrchestrator(
            FundamentalsExecutor,
            self.base_path,
            self.base_path + "/sql/fundamentals.duck",
            None
        )
        with TiingoAPI(self.base_path, self.tiingo_credentials) as api:
            # Loops through every ticker's metadata
            for ticker, exchange, capCategory, sector, lastStatementCheck, statementLastUpdated, isInitialized in self.fundamentals_metadata:
                # Sets fundamentals statement data cutoff to within API boundaries
                if lastStatementCheck == date(1990, 1, 1):
                    lastStatementCheck = date(2014, 1, 1)
                elif lastStatementCheck < statementLastUpdated:
                    lastStatementCheck = lastStatementCheck - timedelta(days=3)
                else:
                    continue

                if not self.regen:  # Skips data acquisition
                    # Acquires data, cleans it, and writes it to database
                    resp = self.handler.fundamentals(
                        api,
                        ticker,
                        self.today,
                        lastStatementCheck,
                        isInitialized
                    )
                    if resp is None:  # No new data/error
                        continue  # No fundamental analysis

                # Formats required inputs for fundamental calculations queue
                q_lst = [ticker, exchange, capCategory, sector]
                self.fund_queue.add_ticker(q_lst)  # Places in queue
        return
