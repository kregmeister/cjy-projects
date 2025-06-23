#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 11:35:28 2024

@author: cjymain
"""

from technically.utils.duckdb import DuckDB
from technically.utils.log import get_logger, log_method

import threading
import queue
import time


class CalculationsOrchestrator:
    "Calculates and stores technicals in near-live time as price data is consumed."

    def __init__(self, worker_function, base_path, db_path, indicator_success_rates, threads=2, max_idle_time=90):
        self.function = worker_function  # Must have specific parameters
        self.base_path = base_path
        self.db_path = db_path
        self.model_db_path = "/".join(db_path.split("/")[:-1]) + "/models.duck"
        self.indicator_success_rates = indicator_success_rates

        self.ticker_queue = queue.Queue()
        self.num_threads = threads
        self.threads = []
        self.stop_event = threading.Event()
        self.max_idle_time = max_idle_time

        self.start_workers()

    def start_workers(self):
        for i in range(self.num_threads):
            t = threading.Thread(target=self.worker, name=f'Worker-{i+1}')
            t.start()
            self.threads.append(t)

    @log_method
    def worker(self):
        thread_name = threading.current_thread().name
        # Workers get their own thread-safe DuckDB connection
        with DuckDB(self.db_path) as db:
            while True:
                try:
                    ticker, exchange, cap, sector = self.ticker_queue.get(timeout=1)
                except queue.Empty:
                    if self.stop_event.is_set():
                        if self.ticker_queue.empty():
                            get_logger().info((f"{thread_name} ",
                                              "stopping. Stop signal received and queue is empty."))
                            break
                        else:
                            continue  # Queue is not empty, continue processing
                    else:
                        continue  # Continue waiting for items
                except Exception as e:
                    get_logger().critical(f"Unexpected error in worker {thread_name}. Shutting down: {str(e)}")
                    self.stop_workers_signal()
                    break
                else:
                    # Process the item
                    self.function(ticker, exchange, cap, sector,
                                  self.indicator_success_rates, self.base_path, db).execute()
                    self.ticker_queue.task_done()

    def add_ticker(self, ticker: list):
        self.ticker_queue.put(ticker)

    def stop_workers_signal(self):
        # Signal workers to stop once the queue is empty
        self.stop_event.set()
        self.wait_on_workers()

    def wait_on_workers(self):
        self.ticker_queue.join()
        for t in self.threads:
            t.join()
        get_logger().info("All worker threads have terminated gracefully.")
