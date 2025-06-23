#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 13:45:35 2023

@author: cjymain
"""

from datetime import timedelta, date
import traceback

from technically.utils.log import log_method, get_logger
from technically.utils.handlers.duckdb import DuckDB
from technically.utils.handlers.queue import CalculationsOrchestrator
from technically.api.tiingo import TiingoAPI
from technically.api.yahoofinance import YFinanceAPI
from technically.computation.transactions import DataAcquisitionController
from technically.computation.calculations import TechnicalsExecutor, FundamentalsExecutor


class DataAcquisitionCoordinator:
    "Coordinates data acquisition and analysis for each ticker."

    def __init__(self, base_path, date):
        self.base_path = base_path
        self.date = date

    def execute(self):
        with DuckDB(mode="multi",
                    db_path=[
                        self.base_path + "/sql/fundamentals.duck",
                        self.base_path + "/sql/prices.duck",
                        self.base_path + "/sql/models.duck"
                    ],
                    read_only=[False, False, False]) as dbs:

            # All data acquisition managed by one HTTP session
            with TiingoAPI(self.base_path) as api:
                # Pass api session and database connections to child classes
                self.handler = DataAcquisitionController(
                    api,
                    dbs,
                    self.base_path,
                )

                # Retreives ticker metadata from databases
                self.prices_metadata, self.fundamentals_metadata = self.handler.metadata()

                # Retreives backtesting metrics to apply in technical analysis
                self.indicator_success_rates = self.handler.models()
                dbs.close("models")  # Closes models DB connection

                # Retreives ticker price data from Tiingo
                self.get_prices()
                self.price_queue.stop_workers_signal()

                # Retreives ticker fundamental statements from Tiingo
                self.get_statements()
                self.fund_queue.stop_workers_signal()

    @log_method(critical=True)
    def get_prices(self):
        # Initializes queue for technical calculations
        self.price_queue = CalculationsOrchestrator(
            TechnicalsExecutor,
            self.base_path,
            self.base_path+"/sql/prices.duck",
            self.indicator_success_rates
        )

        # Loops through every ticker's metadata
        for ticker, exchange, capCategory, sector, assetType, lastDailyCheck, dailyLastUpdated, isInitialized in self.prices_metadata:
            # Prevents retrieval of existing dates from Tiingo
            if isInitialized:
                lastDailyCheck = lastDailyCheck + timedelta(days=1)

            # Acquires data, cleans it, and writes it to database
            resp = self.handler.prices(
                ticker,
                self.date,
                exchange,
                capCategory,
                sector,
                assetType,
                lastDailyCheck,
                isInitialized
            )
            if resp is None:  # Error occurred during data acquisition
                continue  # No technical analysis

            # Formats required inputs for queue
            q_lst = [ticker, exchange, capCategory, sector]
            self.price_queue.add_ticker(q_lst)  # Places in queue
        return

    @log_method(critical=True)
    def get_statements(self):
        # Queue for near-real time fundamentals calculations
        self.fund_queue = CalculationsOrchestrator(
            FundamentalsExecutor,
            self.base_path,
            self.base_path+"/sql/fundamentals.duck",
            None
        )

        # Loops through every ticker's metadata
        for ticker, exchange, capCategory, sector, lastStatementCheck, statementLastUpdated, isInitialized in self.fundamentals_metadata:
            # Sets fundamentals statement data cutoff to within API boundaries
            if lastStatementCheck == date(1990, 1, 1):
                lastStatementCheck = date(2014, 1, 1)
            elif lastStatementCheck < statementLastUpdated:
                lastStatementCheck = lastStatementCheck - timedelta(days=3)
            else:
                continue

            # Acquires data, cleans it, and writes it to database
            resp = self.handler.fundamentals(
                ticker,
                self.date,
                lastStatementCheck,
                isInitialized
            )
            if resp is None:  # No new data/error
                continue  # No fundamental analysis

            # Formats required inputs for queue
            q_lst = [ticker, exchange, capCategory, sector]
            self.fund_queue.add_ticker(q_lst)  # Places in queue
        return
