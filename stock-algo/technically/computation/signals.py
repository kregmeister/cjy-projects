#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  14 10:58:21 2025

@author: cjymain
"""
import numpy as np
import pandas as pd
from scipy.special import expit
import importlib_resources as rs
import traceback

from technically.utils import optimizations as utils
from technically.utils.log import get_logger


class TechnicalIndicatorSignals:
    """
    Identifies technical indicator signals.
    """

    def __init__(self, ticker: str, df: pd.DataFrame, indicator_success_rates, base_path: str):
        self.ticker = ticker
        self.df = df
        self.indicator_success_rates = indicator_success_rates
        self.signal_parameters = pd.read_json(
            rs.open_text("technically", "conf/indicatorSignals.json")
        )
        self.counter = 0

    def calculate(self, indicator: str, identify=True):
        """
        Orchestrates steps to identify, categorize, and score technical indicator signal occurrences.

        Args:
            indicator (str): Name of indicator to find signals for.
            identify (bool): Whether the signals should be identified or not. Default is True.

        Returns:
            None
        """
        try:
            if identify:  # Func signal_identification identifies signals
                results = self.signal_identification(indicator)
            else:  # Func other than signal_identification identifies signals
                results = self.candlestick_patterns()
            if results is None:
                return
            # Assigns signals as being reversal or confirmation
            categorized_results = self.assign_reversal_continuation(results)
            # Applies historic signal success rates as a percent to be applied to the signal 'score'
            self.apply_success_rates(categorized_results)
        except Exception:
            get_logger().error(f"Error calculating signals for {indicator} on {self.ticker}: {traceback.format_exc()}")
        self.counter += 1
        return

    def persist(self):
        return self.df

    def signal_identification(self, indicator: str):
        """
        Handles the conditional logic of an indicator signal and finds periods where true.

        Args:
            indicator (str): The indicator to find signals for, must be equal to its column counterpart in self.df.

        Returns:
            all_conditions (dict): Keys are signal name.
                Values are boolean pd.Series that indicate whether a signal occurs during that period.
        """
        # Skips indicators that do not have an entry in 'indicatorSignals.json' or that are all null in the df
        if (indicator not in self.signal_parameters["indicator"].values or
                self.df[indicator].isnull().all()):
            return

        self.indicator = indicator
        # Isolates signals for the indicator
        indicator_parameters = self.signal_parameters[self.signal_parameters["indicator"] == indicator]
        all_conditions = {}

        for signal_direction in ["bullishSignals", "bearishSignals"]:
            signal_prefix = signal_direction[:-7]+"_"
            # Isolates collection of conditions required for a signal to confirm
            parameters = indicator_parameters[signal_direction].to_dict()[self.counter]
            # Iterates through each condition of the signal's set
            for signal_type, signal_conditions in list(parameters.items())[1:]:
                self.signal_type = signal_type
                signal_key = signal_prefix+signal_conditions["name"]

                # When a signal consists of multiple condition types
                if signal_type.startswith("multi"):
                    res = []
                    for multi_signal_type, multi_signal_conditions in list(signal_conditions.items())[1:]:
                        res_part = self._signal_construction(multi_signal_type, multi_signal_conditions)

                        if multi_signal_type[-1].isdigit():
                            res[-1] = pd.concat([res[-1], res_part], axis=1).any(axis=1)
                            continue

                        res.append(res_part)
                    res = pd.concat(res, axis=1).all(axis=1)
                else:
                    res = self._signal_construction(signal_type, signal_conditions)

                all_conditions[signal_key] = res
        return all_conditions

    def assign_reversal_continuation(self, signals: dict):
        """
        Uses current trend metric to categorize signal occurrences as reversal or continuation.
        The boolean pd.Series dict values are converted to string pd.Series: False will now equal "False".
        True will now equal "rev" or "cont".

        Args:
            signals: Dictionary of signal outcomes (Keys=signal_name, Values=pd.Series).

        Returns:
            signals (dict): Dictionary of signal outcomes. Keys are signal name.
                Values are string pd.Series that indicate whether a signal occurs during that period
                and whether that signal indicates reversal or continuation.
        """
        for sig_name, res in signals.items():
            res = res.astype(str)

            if sig_name.startswith("bullish"):
                res[res == "True"] = np.where(self.df["retracementTrend"] == -1, "rev", "cont")
            elif sig_name.startswith("bearish"):
                res[res == "True"] = np.where(self.df["retracementTrend"] == 1, "rev", "cont")

            signals[sig_name] = res
        return signals

    def apply_success_rates(self, signals: dict):
        """
        Applies success rates to indicator signal occurrences to quantify their impact on price activity.
        The string pd.Series dict values are converted to float pd.Series: "False" mapped to 0.0.
        "rev" and "cont" will be assigned floats based on backtested (or default) success rates.
        Generates self.df columns: {signal_name}_sig

        Args:
            signals: Dictionary of categorized signal outcomes.

        Returns:
            None
        """
        for sig_name, res in signals.items():
            sig_direction = sig_name[:7]

            try:
                matching_rates = self.indicator_success_rates[
                    self.indicator_success_rates["signalName"] == sig_name
                ]
                # Applies backtested success rates
                res[res == "rev"] = matching_rates["reversalSuccessRate"].iloc[0]
                res[res == "cont"] = matching_rates["continuationSuccessRate"].iloc[0]
            except KeyError:  # Signal not backtested
                # Applies arbitrary default success rates so that models can interpret signal occurrence
                if sig_direction == "bullish":  # Defaults
                    res[res == "rev"] = 0.5
                    res[res == "cont"] = 0.25
                elif sig_direction == "bearish":
                    res[res == "rev"] = -0.5
                    res[res == "cont"] = -0.25

            # No signal always equals 0.0
            res[res == "False"] = 0.0

            res = res.astype(float)
            self.df[f"{sig_name}_sig"] = res
        return

    def _signal_construction(self, signal_type: str, signal_conditions: dict):
        """
        Constructs all signal conditions, checks where all are true, and creates boolean results for each period.

        Args:
            signal_type (str): Can equal 'threshold', 'crossover', 'failureSwing', 'divergence', or 'convergence'
            signal_conditions (dict): Dictionary of conditions required for a signal to occur.
                Keys are condition type, values are condition comparison points.

        Returns:
            pd.Series: Boolean pd.Series indicating whether all signal conditions are met.
        """
        if "feature" in signal_conditions:
            feature = self.df[signal_conditions["feature"]].values.copy()
        else:
            feature = self.df[self.indicator].values.copy()

        conditions = []
        if signal_type.startswith("threshold"):
            if "trend" in signal_conditions:
                feature = self._trend(feature, signal_conditions["trend"])
            if "zscored" in signal_conditions:
                feature = utils.rolling_zscore(feature, signal_conditions["zscored"])
            if "above" in signal_conditions:
                above = self._aboveBelow(feature, signal_conditions["above"])
                cond = feature >= above
                conditions.append(cond)
            if "below" in signal_conditions:
                below = self._aboveBelow(feature, signal_conditions["below"])
                cond = feature <= below
                conditions.append(cond)
            if "length" in signal_conditions:
                length, rolling_sum = self._length(conditions[-1], signal_conditions["length"])
                cond = rolling_sum == length
                conditions.append(cond)
            if "currentTrend" in signal_conditions:
                cond = self._currentTrend(signal_conditions["currentTrend"])
                conditions.append(cond)

        elif signal_type.startswith("crossover"):
            feature_shifted = utils.npshift(feature, 1)
            if "above" in signal_conditions:
                above, above_shifted = self._aboveBelow(None, signal_conditions["above"], "crossover")
                cond = (feature >= above) & (feature_shifted <= above_shifted)
                conditions.append(cond)
            if "below" in signal_conditions:
                below, below_shifted = self._aboveBelow(None, signal_conditions["below"], "crossover")
                cond = (feature <= below) & (feature_shifted >= below_shifted)
                conditions.append(cond)
            if "currentTrend" in signal_conditions:
                cond = self._currentTrend(signal_conditions["currentTrend"])
                conditions.append(cond)

        elif signal_type.startswith("failureSwing"):
            thresh = signal_conditions["thresh"]
            if "bottom" in signal_conditions:
                period = signal_conditions["bottom"]
                cond = utils.nprolling(
                    feature,
                    period,
                    "func",
                    {"func": utils.failureSwings,
                     "type": "bottom",
                     "threshold": thresh}
                )
                conditions.append(cond)
            if "top" in signal_conditions:
                period = signal_conditions["top"]
                cond = utils.nprolling(
                    feature,
                    period,
                    "func",
                    {"func": utils.failureSwings,
                     "type": "top",
                     "threshold": thresh}
                )
                conditions.append(cond)
            if "currentTrend" in signal_conditions:
                cond = self._currentTrend(signal_conditions["currentTrend"])
                conditions.append(cond)

        elif signal_type.startswith(("divergence", "convergence")):
            # CoFeature must be included in divergence
            cofeature = self.df[signal_conditions["coFeature"]].values.copy()

            # To assure similar scale, divergence features are always zscored
            feature = utils.rolling_zscore(feature, 250)
            cofeature = utils.rolling_zscore(cofeature, 250)

            if "trend" in signal_conditions:
                feature = self._trend(feature, signal_conditions["trend"])
                cofeature = self._trend(cofeature, signal_conditions["trend"])

            if "direction" in signal_conditions:
                dir_mask, diff = self._direction(feature, cofeature, signal_conditions["direction"])
                if not np.any(dir_mask):  # Ticker has never had specified trend direction
                    # Condition can never occur if the required direction never occurs; returns all False
                    return pd.Series(np.full(len(feature), False, dtype=bool), index=self.df.index)
            else:
                diff = abs(feature - cofeature)
                dir_mask = np.full(len(feature), True, dtype=bool)

            # Acts as a placeholder so conditions series has same index as entire dataframe
            ph = np.full(len(feature), False, dtype=bool)
            valid_indices = np.nonzero(dir_mask)[0]
            diff = diff[valid_indices]
            if "above" in signal_conditions:
                above = self._aboveBelow(diff, signal_conditions["above"])
                cond = diff >= above
            else:
                below = self._aboveBelow(diff, signal_conditions["below"])
                cond = diff <= below
            ph[valid_indices] = cond
            conditions.append(ph)

            if "length" in signal_conditions:
                length, rolling_sum = self._length(conditions[-1], signal_conditions["length"])
                cond = rolling_sum == length
                conditions.append(cond)

            if "currentTrend" in signal_conditions:
                cond = self._currentTrend(signal_conditions["currentTrend"])
                conditions.append(cond)

        conditions_array = np.column_stack(conditions)
        # Checks where all conditions are true for a period
        return pd.Series(np.all(conditions_array, axis=1), index=self.df.index)

    def _aboveBelow(self, feature, threshold, type="threshold"):
        """
        Gathers points of comparison for thresdhold and crossover-based conditions.

        Args:
            feature: The self.df column containing the comparison point.
            threshold: If a string, the comparison point is a percentile of param feature (i.e. "75th").
                If not, its a scalar (i.e. 30).
            type: Equals "threshold" or "crossover". Threshold is true every time, for example, RSI is above 70.
                Crossover is only true, for example, when RSI is below 70 one period, and above 70 the next period.

        Returns:
            Union[threshold, [threshold, threshold]]:
                - threshold: when type equals "threshold", returns one comparison point.
                - [threshold, threshold]: when type equals "crossover", returns two comparison points.

        Examples:
            >>> return 70  # When self.df["rsi20"] > 70
            >>> return self.df["bollingerUpper"]  # When self.df["close"] > self.df["bollingerUpper"]

            >>> return [0, 0]  # When self.df["rsi20"] crosses above 0
            >>> return [self.df["dmiPlus"], self.df["dmiPlus"].shift(1)]  # When self.df["dmiMinus"] crosses above self.df["dmiPlus"]

        """
        if type == "threshold":
            if threshold in self.df.columns:
                t = self.df[threshold].values.copy()
                return t
            elif isinstance(threshold, str):
                return utils.percentile(feature, threshold)
            else:
                return threshold
        elif type == "crossover":
            if isinstance(threshold, str):
                t = self.df[threshold].values.copy()
                t_shifted = utils.npshift(t, 1)
                return [t, t_shifted]
            else:
                return [threshold, threshold]

    def _length(self, condition, length):
        """
        A condition that requires another condition to satisfy for X (length) periods in a row.
        Uses rolling sum of 1's (conditions hitting) at the length required to determine when length is satisfied.

        Args:
            condition: The condition to satisfy.
            length: The number of periods in a row.

        Returns:
            [length, rolling_sum]
        """
        return [length, utils.nprolling(condition, length, "sum")]

    def _currentTrend(self, target):
        """
        When a condition requires a trend to satisfy (uptrend or downtrend).

        Args:
            target: Type of trend.

        Returns:
            pd.Series: Boolean Series of whether the correct trend is satisfied.
        """
        return self.df["retracementTrend"] == target

    def _trend(self, feature, period):
        """
        Fits feature to a trendline at X (period) periods.

        Args:
            feature: The feature to fit.
            period: Number of periods to fit.

        Returns:
            pd.Series: Rate of change in X.
        """
        return utils.lineBestFit(feature, period, return_as="numpy")

    def _direction(self, feature, cofeature, direction):
        """
        The direction in which to measure convergence and divergence.

        Args:
            feature: The self.df column containing the feature being tested for signals.
            cofeature: The self.df column containing the feature's chosen comparison point.
            direction: 1 or -1. 1 is only when feature > cofeature; -1 is only when feature < cofeature.

        Returns:
            list: First item is a mask (np.array) containing only values adhering to the correct direction.
                Second item is an empty array with len equal to len(dir_mask).
        """
        if direction == 1:
            dir_mask = feature > cofeature
            diff = np.zeros_like(feature)
            diff[dir_mask] = abs(feature[dir_mask] - cofeature[dir_mask])
        else:
            dir_mask = feature < cofeature
            diff = np.zeros_like(feature)
            diff[dir_mask] = abs(feature[dir_mask] - cofeature[dir_mask])
        return [dir_mask, diff]

    def total_indicator_score(self):
        """
        Sums all indicator signal scores to summarize price direction indication for the period.
        Generates self.df columns: totalScore

        Returns:
            None
        """
        signal_columns = [col for col in self.df.columns if col.endswith("_sig")]
        total_score = self.df[[col for col in signal_columns]].sum(axis=1)
        self.df["totalScore"] = total_score.rolling(3).mean()
        return

    def candlestick_patterns(self):
        """
        Tests price data for over 40 candlestick patterns.
        Generates self.df columns: candlestick

        Returns:
            None
        """
        df = self.df[["open", "high", "low", "close"]].copy()

        # Data preparation
        candle_body = df["close"] - df["open"]
        upper_tail = np.where(
            candle_body >= 0, df["high"] - df["close"], df["high"] - df["open"]
        )
        lower_tail = np.where(
            candle_body >= 0, df["open"] - df["low"], df["close"] - df["low"]
        )
        midpoint = (df["open"] + df["close"]) / 2

        # Format offset columns to gather historical relationships
        shifts = {}
        for i in range(1, 5):
            for column in ["open", "high", "low", "close", "midpoint"]:
                if column == "midpoint":
                    shifts[f"{column}-{i}"] = midpoint.shift(i)
                else:
                    shifts[f"{column}-{i}"] = df[column].shift(i)

        # Assistive functions/variables
        green_candles = candle_body[candle_body >= 0]
        red_candles = candle_body[candle_body < 0]

        # Definitions for common pattern qualifiers based on percentiles
        long_green, long_red = [
            np.percentile(green_candles, 60),
            np.percentile(abs(red_candles), 60)
        ]
        short_green, short_red = [
            np.percentile(green_candles, 40),
            np.percentile(abs(red_candles), 40)
        ]
        doji, price_match = [
            np.percentile(abs(candle_body), 7.5),
            np.percentile(abs(candle_body), 4)
        ]
        long_upper_tail, long_lower_tail = [
            np.percentile(upper_tail, 80),
            np.percentile(lower_tail, 80)
        ]
        short_upper_tail, short_lower_tail = [
            np.percentile(upper_tail, 20),
            np.percentile(lower_tail, 20)
        ]

        # Candlestick selection functions

        # 5-candle bullish
        def bullishBreakaway():
            return (
                    ((shifts["open-4"] - shifts["close-4"]) >= long_red) &
                    ((shifts["open-3"] - shifts["close-3"]) <= short_red) &
                    (shifts["open-3"] > shifts["close-3"]) &
                    ((shifts["open-1"] - shifts["close-1"]) <= short_red) &
                    (shifts["open-1"] > shifts["close-1"]) &
                    (df["close"] > df["open"]) &
                    (shifts["close-4"] > shifts["open-3"]) &
                    (shifts["close-1"] < shifts["close-3"]) &
                    (shifts["close-1"] < shifts["midpoint-2"]) &
                    (shifts["close-4"] > df["close"]) &
                    (df["close"] > shifts["open-3"])
            )

        def bullishLadder():
            return (
                    ((shifts["open-4"] - shifts["close-4"]) >= long_red) &
                    ((shifts["open-3"] - shifts["close-3"]) >= long_red) &
                    ((shifts["open-2"] - shifts["close-2"]) >= long_red) &
                    (shifts["close-1"] < shifts["open-1"]) &
                    (df["close"] > df["open"]) &
                    (shifts["open-4"] > shifts["open-3"]) &
                    (shifts["open-3"] > shifts["open-2"]) &
                    (shifts["open-2"] > shifts["open-1"]) &
                    (shifts["close-4"] > shifts["close-3"]) &
                    (shifts["close-3"] > shifts["close-2"]) &
                    (shifts["close-2"] > shifts["close-1"]) &
                    ((shifts["high-1"] - shifts["open-1"]) > (shifts["open-1"] - shifts["close-1"])) &
                    (df["open"] > shifts["open-1"]) &
                    (df["close"] > shifts["high-1"])
            )

        # 5-candle bearish
        def bearishBreakaway():
            return (
                    ((shifts["close-4"] - shifts["open-4"]) >= long_green) &
                    ((shifts["close-3"] - shifts["open-3"]) <= short_green) &
                    (shifts["open-3"] < shifts["close-3"]) &
                    ((shifts["close-1"] - shifts["open-1"]) <= short_green) &
                    (shifts["open-1"] < shifts["close-1"]) &
                    (df["close"] < df["open"]) &
                    (shifts["close-4"] < shifts["open-3"]) &
                    (shifts["close-1"] > shifts["close-3"]) &
                    (shifts["close-1"] > shifts["midpoint-2"]) &
                    (shifts["close-4"] < df["close"]) &
                    (df["close"] < shifts["open-3"])
            )

        def bearishLadder():
            return (
                    ((shifts["close-4"] - shifts["open-4"]) >= long_green) &
                    ((shifts["close-3"] - shifts["open-3"]) >= long_green) &
                    ((shifts["close-2"] - shifts["open-2"]) >= long_green) &
                    (shifts["close-1"] > shifts["open-1"]) &
                    (df["close"] < df["open"]) &
                    (shifts["open-4"] < shifts["open-3"]) &
                    (shifts["open-3"] < shifts["open-2"]) &
                    (shifts["open-2"] < shifts["open-1"]) &
                    (shifts["close-4"] < shifts["close-3"]) &
                    (shifts["close-3"] < shifts["close-2"]) &
                    (shifts["close-2"] < shifts["close-1"]) &
                    ((shifts["open-1"] - shifts["low-1"]) > (shifts["close-1"] - shifts["open-1"])) &
                    (df["open"] < shifts["open-1"]) &
                    (df["close"] < shifts["low-1"])
            )

        # 3-candle bullish
        def bullishStickSandwich():
            return (
                    (shifts["close-2"] < shifts["open-2"]) &
                    (shifts["open-1"] > shifts["close-2"]) &
                    (shifts["close-1"] > shifts["open-2"]) &
                    (shifts["open-2"] > shifts["open-1"]) &
                    (df["open"] > shifts["open-1"]) &
                    (df["close"] < df["open"]) &
                    (abs(df["close"] - shifts["close-2"]) <= price_match)
            )

        def bullishUniqueThreeRivers():
            return (
                    ((shifts["open-2"] - shifts["close-2"]) >= long_red) &
                    (shifts["close-1"] > shifts["close-2"]) &
                    (shifts["close-1"] < shifts["open-1"]) &
                    (shifts["open-1"] < shifts["open-2"]) &
                    (shifts["low-1"] < shifts["low-2"]) &
                    ((df["close"] - df["open"]) <= short_green) &
                    (df["close"] > df["open"]) &
                    (df["open"] > shifts["low-1"])
            )

        def bullishMorningStar():
            return (
                    ((shifts["open-2"] - shifts["close-2"]) >= long_red) &
                    ((abs(shifts["close-1"] - shifts["open-1"])) <= short_green) &
                    ((df["close"] - df["open"]) >= long_green) &
                    (shifts["close-2"] > shifts["open-1"]) &
                    (shifts["close-2"] > shifts["close-1"]) &
                    (shifts["close-1"] < df["open"]) &
                    (df["open"] > shifts["open-1"]) &
                    (df["close"] > shifts["midpoint-2"])
            )

        def bullishTriStar():
            return (
                    (abs(shifts["close-2"] - shifts["open-2"]) <= doji) &
                    (abs(shifts["close-1"] - shifts["open-1"]) <= doji) &
                    (abs(df["close"] - df["open"]) <= doji) &
                    (shifts["midpoint-1"] < shifts["low-2"]) &
                    (shifts["midpoint-1"] < df["low"])
            )

        def bullishThreeWhiteSoldiers():
            return (
                    (shifts["open-2"] < shifts["open-1"]) &
                    (shifts["open-1"] < df["open"]) &
                    (shifts["close-2"] < shifts["close-1"]) &
                    (shifts["close-1"] < df["close"]) &
                    ((shifts["high-2"] - shifts["close-2"]) <= short_upper_tail) &
                    ((shifts["high-1"] - shifts["close-1"]) <= short_upper_tail) &
                    ((df["high"] - df["close"]) <= short_upper_tail)
            )

        # 3-candle bearish
        def bearishThreeBlackCrows():
            return (
                    (shifts["open-2"] > shifts["open-1"]) &
                    (shifts["open-1"] > df["open"]) &
                    (shifts["close-2"] > shifts["close-1"]) &
                    (shifts["close-1"] > df["close"]) &
                    ((shifts["close-2"] - shifts["low-2"]) <= short_lower_tail) &
                    ((shifts["close-1"] - shifts["low-1"]) <= short_lower_tail) &
                    ((df["close"] - df["low"]) <= short_lower_tail)
            )

        def bearishEveningStar():
            return (
                    ((shifts["close-2"] - shifts["open-2"]) >= long_green) &
                    ((abs(shifts["close-1"] - shifts["open-1"])) <= short_green) &
                    ((df["open"] - df["close"]) >= long_red) &
                    (shifts["close-2"] < shifts["open-1"]) &
                    (shifts["close-2"] < shifts["close-1"]) &
                    (shifts["close-1"] > df["open"]) &
                    (df["open"] < shifts["open-1"]) &
                    (df["close"] < shifts["midpoint-2"])
            )

        def bearishTriStar():
            return (
                    (abs(shifts["close-2"] - shifts["open-2"]) <= doji) &
                    (abs(shifts["close-1"] - shifts["open-1"]) <= doji) &
                    (abs(df["close"] - df["open"]) <= doji) &
                    (shifts["midpoint-1"] > shifts["high-2"]) &
                    (shifts["midpoint-1"] > df["high"])
            )

        # 2-candle bullish
        def bullishEngulfing():
            return (
                    (df["close"] > shifts["open-1"]) &
                    (shifts["open-1"] > shifts["close-1"]) &
                    (shifts["close-1"] > df["open"])
            )

        def bullishMeetingLines():
            return (
                    ((shifts["open-1"] - shifts["close-1"]) >= long_red) &
                    (df["close"] > df["open"]) &
                    (abs(df["close"] - shifts["close-1"]) <= price_match)
            )

        def bullishHarami():
            return (
                    ((shifts["open-1"] - shifts["close-1"]) >= long_red) &
                    (shifts["open-1"] > df["close"]) &
                    (df["close"] > df["open"]) &
                    (df["open"] > shifts["close-1"])
            )

        def bullishHaramiCross():
            return (
                    ((shifts["open-1"] - shifts["close-1"]) >= long_red) &
                    (abs(df["close"] - df["open"]) <= doji) &
                    (shifts["open-1"] > midpoint) &
                    (shifts["close-1"] < midpoint)
            )

        def bullishPiercingLine():
            return (
                    ((shifts["open-1"] - shifts["close-1"]) >= long_red) &
                    (df["open"] < shifts["close-1"]) &
                    (df["close"] > shifts["midpoint-1"])
            )

        def bullishKicking():
            return (
                    (shifts["close-1"] < shifts["open-1"]) &
                    (df["close"] > df["open"]) &
                    ((shifts["high-1"] - shifts["open-1"]) <= short_upper_tail) &
                    ((df["high"] - df["close"]) <= short_upper_tail) &
                    ((shifts["close-1"] - shifts["low-1"]) <= short_lower_tail) &
                    ((df["open"] - df["low"]) <= short_lower_tail) &
                    (shifts["open-1"] < df["open"])
            )

        def bullishHomingPigeon():
            return (
                    ((shifts["open-1"] - shifts["close-1"]) >= long_red) &
                    ((df["open"] - df["close"]) <= short_red) &
                    (df["open"] > df["close"]) &
                    (shifts["open-1"] > df["open"]) &
                    (shifts["close-1"] < df["close"])
            )

        def bullishMatchingLow():
            return (
                    (shifts["close-1"] < shifts["open-1"]) &
                    (df["close"] < df["open"]) &
                    (abs(shifts["close-1"] - df["close"]) <= price_match)
            )

        def bullishDojiStar():
            return (
                    ((shifts["open-1"] - shifts["close-1"]) >= long_red) &
                    (abs(df["close"] - df["open"]) <= doji) &
                    (midpoint < shifts["close-1"])
            )

        # 2-candle bearish
        def bearishShootingStar():
            return (
                    (midpoint > shifts["close-1"]) &
                    (shifts["close-1"] > shifts["open-1"]) &
                    ((abs(df["close"] - df["open"])) <= short_green) &
                    (((df["open"] - df["low"]) <= short_lower_tail) |
                     ((df["close"] - df["low"]) <= short_lower_tail)) &
                    (((df["high"] - df["open"]) >= long_upper_tail) |
                     ((df["high"] - df["close"]) >= long_upper_tail))
            )

        def bearishEngulfing():
            return (
                    (shifts["close-1"] > shifts["open-1"]) &
                    (df["close"] < df["open"]) &
                    (shifts["close-1"] < df["open"]) &
                    (shifts["open-1"] > df["close"])
            )

        def bearishMeetingLines():
            return (
                    ((shifts["close-1"] - shifts["open-1"]) >= long_green) &
                    (df["close"] < df["open"]) &
                    (abs(df["close"] - shifts["close-1"]) <= price_match)
            )

        def bearishHarami():
            return (
                    ((shifts["close-1"] - shifts["open-1"]) >= long_green) &
                    (shifts["open-1"] < df["close"]) &
                    (df["close"] < df["open"]) &
                    (df["open"] < shifts["close-1"])
            )

        def bearishHaramiCross():
            return (
                    ((shifts["close-1"] - shifts["open-1"]) >= long_green) &
                    (abs(df["close"] - df["open"]) <= doji) &
                    (shifts["open-1"] < midpoint) &
                    (shifts["close-1"] > midpoint)
            )

        def bearishDarkCloudCover():
            return (
                    ((shifts["close-1"] - shifts["open-1"]) >= long_green) &
                    (df["open"] > shifts["close-1"]) &
                    (df["close"] < shifts["midpoint-1"])
            )

        def bearishKicking():
            return (
                    (shifts["close-1"] > shifts["open-1"]) &
                    (df["close"] < df["open"]) &
                    ((shifts["high-1"] - shifts["close-1"]) <= short_upper_tail) &
                    ((df["high"] - df["open"]) <= short_upper_tail) &
                    ((shifts["open-1"] - shifts["low-1"]) <= short_lower_tail) &
                    ((df["close"] - df["low"]) <= short_lower_tail) &
                    (shifts["open-1"] > df["open"])
            )

        def bearishMatchingHigh():
            return (
                    (shifts["close-1"] > shifts["open-1"]) &
                    (df["close"] > df["open"]) &
                    (abs(shifts["close-1"] - df["close"]) <= price_match)
            )

        # 1-candle bullish
        def bullishBeltHold():
            return (
                    ((df["close"] - df["open"]) >= long_green) &
                    ((df["high"] - df["close"]) >= short_upper_tail) &
                    ((df["open"] - df["low"]) <= short_lower_tail)
            )

        def bullishInvertedHammer():
            return (
                    ((abs(df["close"] - df["open"])) <= short_green) &
                    (((df["open"] - df["low"]) <= short_lower_tail) |
                     ((df["close"] - df["low"]) <= short_lower_tail)) &
                    (((df["high"] - df["open"]) >= long_upper_tail) |
                     ((df["high"] - df["close"]) >= long_upper_tail))
            )

        def bullishHammer():
            return (
                    (shifts["close-1"] > df["close"]) &
                    ((abs(df["close"] - df["open"])) <= short_green) &
                    (((df["high"] - df["open"]) <= short_upper_tail) |
                     ((df["high"] - df["close"]) <= short_upper_tail)) &
                    (((df["open"] - df["low"]) >= long_lower_tail) |
                     ((df["close"] - df["low"]) >= long_lower_tail))
            )

        # 1-candle bearish
        def bearishBeltHold():
            return (
                    ((df["open"] - df["close"]) >= long_red) &
                    ((df["high"] - df["open"]) <= short_upper_tail) &
                    ((df["close"] - df["low"]) >= short_lower_tail)
            )

        def bearishHangingMan():
            return (
                    (shifts["close-1"] < df["close"]) &
                    ((abs(df["close"] - df["open"])) <= short_green) &
                    (((df["high"] - df["open"]) <= short_upper_tail) |
                     ((df["high"] - df["close"]) <= short_upper_tail)) &
                    (((df["open"] - df["low"]) >= long_lower_tail) |
                     ((df["close"] - df["low"]) >= long_lower_tail))
            )

        patterns = [
            bullishBreakaway, bullishLadder, bearishBreakaway, bearishLadder,
            bullishStickSandwich, bullishUniqueThreeRivers, bullishMorningStar, bullishTriStar,
            bullishThreeWhiteSoldiers, bearishThreeBlackCrows, bearishEveningStar, bearishTriStar,
            bullishEngulfing, bullishMeetingLines, bullishHarami, bullishHaramiCross,
            bullishPiercingLine, bullishKicking, bullishHomingPigeon, bullishMatchingLow,
            bullishDojiStar, bearishShootingStar, bearishEngulfing, bearishMeetingLines,
            bearishHarami, bearishHaramiCross, bearishDarkCloudCover, bearishKicking,
            bearishMatchingHigh, bullishBeltHold, bullishInvertedHammer, bullishHammer,
            bearishBeltHold, bearishHangingMan
        ]
        # Checks for occurrences of all above candlestick patterns
        all_patterns = {func.__name__: func() for func in patterns}
        return all_patterns
