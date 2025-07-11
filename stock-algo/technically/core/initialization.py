#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 13:34:44 2024

@author: cjymain
"""

import os
import duckdb

class TechnicallyInitialization:
    """
    Initializes the main directory structure and database connections.
    """

    def __init__(self, base_path):
        self.base_path = base_path

    def execute(self):
        self.directories_setup()
        self.duckdb_setup()

    def directories_setup(self):
        os.mkdir(self.base_path)
        # Creates main data subdirectories
        for subdir in ["sql", "parquet", "models", "backtests", "logs", "files"]:
            os.mkdir(self.base_path + f"/{subdir}")
            if subdir == "csv":
                os.mkdir(self.base_path + "/csv/results")

        # Creates subdirectories for parquet
        for subdir in ["daily", "quarterly"]:
            os.mkdir(self.base_path + f"/parquet/{subdir}")

        # Creates subdirectories for logs
        for subdir in ["main", "dev", "backtests", "resources"]:
            os.mkdir(self.base_path + f"/logs/{subdir}")


    def duckdb_setup(self):
        # Creates empty databases
        for name in ["prices", "fundamentals", "models"]:
            db_path = self.base_path + f"/sql/{name}.duck"
            conn = duckdb.connect(db_path)
            conn.close()
        
