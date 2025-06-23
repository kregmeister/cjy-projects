#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 11:49:55 2024

@author: cjymain
"""

import os
from datetime import datetime, date, timedelta

from technically.core import (
    initialization,
    metadata,
    coordination,
    cleaning,
    modeling
)
from technically.utils import log, date_checks


class Foundation:
    "Executes each core step of Technically."
    
    def __init__(self):
        self.date = date.today()

        # Establishes user directory
        self.home_dir = os.path.expanduser("~")
        self.base_path = self.home_dir + "/.technically"

        # Creates program data directory tree if it does not exist
        self.init = False
        if not os.path.isdir(self.base_path):
            self.init = True
            initialization.TechnicallyInitialization(self.base_path).execute()

        # Initializes log for the run
        log.log_setup("DEBUG", self.base_path+f"/logs/dev/tc_{self.date}.log")

        # Logs initialization info
        if self.init == True:
            log.get_logger().info("Directories setup complete.")
            log.get_logger().info("DuckDB setup complete.")
            log.get_logger().info("Baseline technical info loaded to data directory.")
            log.get_logger().info("Ensure that API keys are stored in AWS Secrets Manager and that AWS CLI is installed and configured on system.")

        self.date = date.today()
        self.trading_day = date_checks.market_calendar_check(self.date)
        # Returns False or digit (I.E. 2 if today is 2nd Friday of the month)
        self.friday = date_checks.which_friday(self.date)
    
    def execute(self):
        print(f"Start: {datetime.today()}")

        # Program does not execute if NYSE is not open today
        if not self.trading_day:
            log.get_logger().info("NYSE is not trading today. Not executing.")
            return

        # Determines how the program aggregates tickers and their metrics program-wide
        group_by = "sector"

        metadata.ManageMetadata(self.base_path, self.init, self.friday).execute()
        log.get_logger().info("Ticker metadata acquired/updated.")

        coordination.DataAcquisitionCoordinator(self.base_path, self.date).execute()
        log.get_logger().info("Ticker data acquisition/analysis/backtesting completed.")

        if self.init == False:
            cleaning.DatabaseChecks(self.base_path, self.friday).execute()
            log.get_logger().info("Data cleaning tasks completed.")

        #modeling.tc_spark(self.date, self.base_path, "sector")
        #log.get_logger().info("PySpark modeling tasks completed.")

        print(f"End: {datetime.today()}")
        log.get_logger().info("Execution completed.")

def execute_technically():
    "Runs Technically in full."
    executor = Foundation()
    executor.execute()
