#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  5 11:52:22 2025

@author: cjymain
"""

import pandas as pd
from datetime import date

from technically.utils.log import get_logger
from technically.utils.handlers.duckdb import DuckDB


class IndicatorSuccessRates:
    "Calculates historical technical indicator success rates."

    def __init__(self, df, ticker, base_path):
        self.df = df
        self.ticker = ticker
        self.date = date.today()
        self.base_path = base_path

    def execute(self):
        with DuckDB(self.base_path + "/sql/models.duck") as self.db:
            ready = self.check()
            if not ready:
                return

            get_logger().info(f"Backtesting initiated for {self.ticker}")

            signals_df = self.test_conditions(self.df)
            self.to_duckdb(signals_df)

            get_logger().info(f"Backtesting complete for {self.ticker}")
            return

    def check(self):
        # Checks if self.ticker has been backtested
        exists = self.db.execute('''
            SELECT CASE WHEN EXISTS (SELECT 1 FROM metadata WHERE ticker = ?) THEN 1 ELSE 0 END
            ''', [self.ticker]
        ).fetchone()

        # Immediately triggers backtesting where no entry exists
        if exists == 0:
            return True

        # Finds the date of the latest backtest for self.ticker
        if self.date.weekday() == 4 or self.date.weekday() == 5:  # Only re-trains on Fri or Sat
            last_trained = self.db.execute('''
                SELECT date FROM indicatorSuccessRates
                    WHERE ticker = ? LIMIT 1;
                ''', [self.ticker]
            ).fetchone()[0]

            # Triggers backtesting if 28 days or more since last backtest
            if (self.date - last_trained).days >= 28:
                return True
            else:  # Less than 28 days since trained
                return False
        else:  # Not the weekend
            return False

    def to_duckdb(self, df):
        # Replaces old training results if exists, creates new entry if not
        self.db.execute('''
            INSERT OR REPLACE INTO indicatorSuccessRates
                SELECT * FROM df;
            '''
        )

    def test_conditions(self, df):
        df["date"] = pd.to_datetime(
            df["date"], format='ISO8601'
        ).dt.date

        # Scalers needed for dict item
        date = self.date
        ticker = df["ticker"].iloc[0]
        sector = df["sector"].iloc[0]
        cap = df["cap"].iloc[0]
        exchange = df["exchange"].iloc[0]

        # Selects each indicator signal column
        columns = [col for col in df.columns if col.endswith("_sig")]

        results = []
        for signal in columns:
            # Handles syntactic exception to isolate signal name
            indicator_name = signal.split("_")[1] if signal[7] == "_" else "candlestick"
            sig_direction = signal[:7]  # Equals 'bullish' or 'bearish'

            # Initialize counters
            reversal_count = 0
            reversal_success = 0
            reversal_failure = 0
            continuation_count = 0
            continuation_success = 0
            continuation_failure = 0

            # Defines masks for signal occurrence, counts them, and isolates data where they occur
            if sig_direction == "bullish":
                # Masks for occurrence of reversal & continuation signals
                bullish_rev = (df[signal] > 0) & (df["retracementTrend"] == -1)
                bullish_cont = (df[signal] > 0) & (df["retracementTrend"] == 1)
                # Count signal occurrences
                reversal_count = len(bullish_rev[bullish_rev])
                continuation_count = len(bullish_cont[bullish_cont])
                # Creates Dataframe of occurrences with columns required for confirmation
                signal_df = df[["retracementTrend", "reversalConf"]][bullish_rev | bullish_cont]
            elif sig_direction == "bearish":
                bearish_rev = (df[signal] < 0) & (df["retracementTrend"] == 1)
                bearish_cont = (df[signal] < 0) & (df["retracementTrend"] == -1)
                reversal_count = len(bearish_rev[bearish_rev])
                continuation_count = len(bearish_cont[bearish_cont])
                signal_df = df[["retracementTrend", "reversalConf"]][bearish_rev | bearish_cont]

            # Iterates through signal occurrences
            for idx in signal_df.index:
                current_trend = signal_df.loc[idx, "retracementTrend"]  # 1 or -1

                # Constructs window for signal to confirm/deny
                end_idx = min(idx + 60, len(df) - 1)
                segment = df.loc[idx:end_idx]

                if sig_direction == "bullish":
                    if current_trend == 1:
                        # Continuation confirmed if trend persists for window
                        if all(segment["reversalConf"] == "None"):
                            continuation_success += 1
                        # Continuation denied if trend reverses within window
                        else:
                            continuation_failure += 1
                    elif current_trend == -1:
                        # Reversal confirmed if trend reverses within window
                        if any(segment["reversalConf"] == "toUp"):
                            reversal_success += 1
                        # Reversal denied if trend persists for window
                        else:
                            reversal_failure += 1
                elif sig_direction == "bearish":
                    if current_trend == -1:
                        if all(segment["reversalConf"] == "None"):
                            continuation_success += 1
                        else:
                            continuation_failure += 1
                    elif current_trend == 1:
                        if any(segment["reversalConf"] == "toDown"):
                            reversal_success += 1
                        else:
                            reversal_failure += 1

            results.append({
                "date": date,
                "ticker": ticker,
                "sector": sector,
                "cap": cap,
                "exchange": exchange,
                "indicatorName": indicator_name,
                "signalName": signal,
                "reversalCount": reversal_count,
                "reversalSuccess": reversal_success,
                "reversalFailure": reversal_failure,
                "continuationCount": continuation_count,
                "continuationSuccess": continuation_success,
                "continuationFailure": continuation_failure
            })
        return pd.DataFrame(results)