#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  5 10:58:21 2023

@author: craigyingling321
"""

import re
import numpy as np
import pandas as pd
from filterpy.kalman import KalmanFilter
import pywt

from technically.utils import optimizations as utils


class TechnicalFormulas:
    """
    Calculates technical indicators and other features.
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()  # Columns added throughout
        self.og_df = df.copy()  # Includes only price data

    def persist(self):
        """
        Updates the current state of self.df to the space that instantiated the class.

        Returns:
            pd.DataFrame: updated self.df.
        """
        return self.df

    def price_ceilings_floors(self, percentiles: tuple = (99, 1)):
        """
        Creates price ceilings and floors based on percentiles of daily price changes for each period.
        Generates self.df columns: priceCeiling, priceFloor, priceChange.

        Args:
            percentiles (tuple): Top and bottom percentiles for price ceiling and floor.
                Default is (99, 1).

        Returns:
            None
        """
        # Separates self.df into multiple DataFrames by date
        buckets = utils.bucketizer(self.og_df[["date", "close"]].copy())

        ceilings = []
        floors = []
        diffs = []
        for df in buckets:
            # Calculates daily change in price and percentiles of those changes
            price_changes = df["close"].diff()
            ceiling = utils.percentile(price_changes, percentiles[0])
            floor = utils.percentile(price_changes, percentiles[1])

            # Applies current ceilings and floors to detect extreme price changes
            diffs.append(price_changes)
            ceilings.append(df["close"].shift(1) + ceiling)
            floors.append(df["close"].shift(1) + floor)

        self.df["priceChange"] = np.concatenate(diffs)
        self.df["priceCeiling"] = np.concatenate(ceilings)
        self.df["priceFloor"] = np.concatenate(floors)
        return

    def demand_index(self, period=10, priceRange=2, smoothing=10):
        """
        Recursively calculates demand index for each period.
        Generates self.df columns: demandIndex

        Args:
            period (int): Number of days for rolling mean of price range. Default is 10.
            priceRange (int): Number of days for price range calculation. Default is 2.
            smoothing (int): Number of days for smoothing. Default is 10.

        Returns:
            None
        """
        df = self.og_df[["open", "high", "low", "close", "volume"]].copy()
        P = (df["close"] - df["open"]) / df["open"]

        two_day_price_range = df["high"].rolling(window=priceRange).max() - df["low"].rolling(window=priceRange).min()
        VA = two_day_price_range.rolling(window=period).mean()
        K = (3 * df["close"]) / VA

        P = P * K
        V = df["volume"]

        change_mask = df["close"] > df["open"]
        BP = np.where(change_mask, V, V / P)

        SP = np.where(change_mask, V / P, V)

        pressure_mask = abs(BP) > abs(SP)
        DI = np.where(pressure_mask, SP / BP, BP / SP)
        DI_smoothed = pd.Series(DI).ewm(span=smoothing).mean()
        self.df["demandIndex"] = DI_smoothed
        return

    def kalman_filter_multi(self, N=1, processNoise=0.01, measurementNoise=15.0, initialErrorCovariance=100.0):
        """
        Kalman filter with multiple state vectors.
        Generates self.df columns: kalmanClose, kalmanTrend

        Args:
        N (int): Number of states. Default is 1.
        processNoise (float): Noise to add to each state vector. Default is 0.01.
        measurementNoise (float): Noise to add to each resulting value. Default is 15.0.
        initialErrorCovariance (float): Initial error covariance. Default is 100.0.

        Returns:
            None
        """
        values = self.og_df["close"].copy().values

        kf = KalmanFilter(dim_x=N, dim_z=1)

        kf.F = np.eye(N)  # State transition matrix
        kf.H = np.ones((1, N))  # Measurement matrix
        kf.P *= initialErrorCovariance
        kf.R = measurementNoise
        kf.Q = processNoise

        # Starting with the first price and zero velocity
        kf.x = np.array([[values[0]]])

        filtered_values = []
        trends = []

        # Previous filtered price for trend calculation
        prev_filtered_value = None

        trend = 0
        for z in values:
            # Predict step
            kf.predict()

            # Update step with the new measurement
            kf.update(np.array([[z]]))

            # Extract the filtered price estimate
            filtered_value = kf.x[0, 0]
            filtered_values.append(filtered_value)

            # Tracks direction and length of trend
            if prev_filtered_value is not None:
                if filtered_value > prev_filtered_value:
                    if trend < 0:
                        trend = 0
                    trend += 1  # Upward trend
                elif filtered_value < prev_filtered_value:
                    if trend > 0:
                        trend = 0
                    trend += -1  # Downward trend
                else:
                    if trend > 0:
                        trend += 1  # No change
                    else:
                        trend += -1
            trends.append(trend)
            # Update previous filtered price
            prev_filtered_value = filtered_value

        self.df["kalmanClose"] = filtered_values
        self.df["kalmanTrend"] = trends
        return

    def kalman_filter_single(self, Q=0.01, R=15.0, P0=100.0, trending=True):
        """
        Kalman filter with a single state vector.
        Generates self.df columns: kalmanClose, kalmanTrend.

        Args:
            Q (float): Covariance of process noise. Default is 0.01.
            R (float): Covariance of measurement noise. Default is 15.0.
            P0 (float): Initial error covariance. Default is 100.0.
            trending (bool): Whether to track trend direction and length based on kalman filter outputs. Default is True.

        Returns:
            None
        """
        values = self.og_df["close"].copy().values

        filtered_values = []
        trends = []

        prev_x = None
        x = values[0]
        P = P0
        trend = 0
        for z in values:
            # Predict step
            x_pred = x
            P_pred = P + Q
            # Kalman Gain
            K = P_pred / (P_pred + R)
            # Update step
            x = x_pred + K * (z - x_pred)
            P = (1 - K) * P_pred

            filtered_values.append(x)

            # Tracks direction and length of trend
            if trending:
                if prev_x is not None:
                    if x > prev_x:
                        if trend < 0:
                            trend = 0
                        trend += 1  # Reverse up
                    elif x < prev_x:
                        if trend > 0:
                            trend = 0
                        trend += -1  # Reverse down
                    else:
                        if trend > 0:
                            trend += 1  # No reversal
                        else:
                            trend += -1
                trends.append(trend)
            # Update previous filtered price
            prev_x = x

        self.df["kalmanClose"] = filtered_values
        self.df["kalmanTrend"] = trends
        return

    def wavelets(self, wavelet="db6", scale=0.3):
        """
        Calculates wavelet coefficients.
        Generates self.df columns: wavelet_db6

        Args:
            wavelet (str): Wavelet to calculate coefficients. Default is "db6".
            scale (float): Scaling factor. Default is 0.3.

        Returns:
            None
        """
        values = self.og_df["close"].copy().values

        # Deconstruct price coefficients
        coefficients = pywt.wavedec(values, wavelet, mode='per')
        # Filter out coefficients based on scale
        coefficients[1:] = [
            pywt.threshold(i, value=scale * values.max())
            for i in coefficients[1:]
        ]
        # Reconstruct de-noised values
        reconstructed_signal = pywt.waverec(coefficients, wavelet, mode='per')

        self.df[f"wavelet_{wavelet}"] = reconstructed_signal
        return

    def percent_from_extreme(self, period=250, min_len=120, type="max"):
        """
        Measures the percent difference between a given rolling window's max/min and closing price.
        Generates self.df columns: percentDiffMax or percentDiffMin

        Args:
            period (int): Size of the rolling window. Default is 250.
            min_len (int): Minimum length of the DataFrame. Default is 120.
            type (str): Options are "max" or "min". Default is "max".

        Returns:
            None
        """
        df = self.og_df[["high", "low", "close"]].copy()

        # Sets period to min_len if DataFrame length falls between them
        if min_len <= len(df) <= period:
            period = min_len

        # Percent difference formula
        def _pc_diff(close, extreme):
            pc = (close - extreme).abs() / ((close + extreme) / 2) * 100
            return pc

        # Calculates
        if type == "max":
            year_high = df["high"].rolling(window=period).max()
            self.df["percentDiffMax"] =  _pc_diff(df["close"], year_high)
        else:
            year_low = df["low"].rolling(window=period).min()
            self.df["percentDiffMin"] = _pc_diff(df["close"], year_low)
        return

    def average_true_range(self, period=20, min_periods=120):
        """
        Calculates the Average True Range (ATR) of a given time period.
        Generates self.df columns: atr20

        Args:
            period (int): EMA length of True Range (TR). Default is 20.
            min_periods (int): Minimum length of the DataFrame. Default is 120.

        Returns:
            None
        """
        df = self.og_df[["high", "low", "close"]].copy()

        alpha = 1/period

        # TR
        h_l = df['high'] - df['low']
        h_c = np.abs(df['high'] - df['close'].shift(1))
        l_c = np.abs(df['low'] - df['close'].shift(1))
        tr = pd.concat([h_l, h_c, l_c], axis=1).max(axis=1)

        # ATR
        atr = tr.ewm(alpha=alpha, adjust=False, min_periods=min_periods).mean()

        self.df[f"atr{period}"] = atr
        return

    def aroon_oscillator(self, period=20):
        """
        Calculates the Aroon Oscillator of a given time period.
        Generates self.df columns: aroon20

        Args:
            period (int): Window length of checking for new highs/lows. Default is 20.

        Returns:
            None
        """
        df = self.og_df[["high", "low"]].copy()

        # Counts how many periods since the window's max/min
        periods_since_high = df['high'].rolling(window=period).apply(
            lambda x: period - 1 - x.argmax(), raw=True
        )
        periods_since_low = df['low'].rolling(window=period).apply(
            lambda x: period - 1 - x.argmin(), raw=True
        )

        # Calculates aroon directions
        aroon_up = ((period - periods_since_high) / period) * 100
        aroon_down = ((period - periods_since_low) / period) * 100

        aroon = aroon_up - aroon_down

        self.df[f"aroon{period}"] = aroon
        return

    def commodity_channel_index(self, period=20, smoothing=14):
        """
        Calculates the Commodity Channel Index (CCI) of a given time period and its moving average.
        Generates self.df columns: cci20, cci20_ma14

        Args:
            period (int): Length of the moving average window. Default is 20.
            smoothing (int): Length of EMA smoothing window. Default is 14.

        Returns:
            None
        """
        df = self.og_df[["high", "low", "close"]].copy()

        # Typical price
        tp = (df['high'] + df['low'] + df['close']) / 3

        # Calculates moving average and mean deviation
        ma = tp.rolling(window=period).mean()
        md = tp.rolling(window=period).apply(
            lambda x: np.mean(np.abs(x - np.mean(x))), raw=True
        )

        # Calculates CCI and smooths
        cci = (tp - ma) / (.015 * md)
        ccima = (cci.rolling(window=smoothing).sum()) / smoothing

        self.df[f"cci{period}"] = cci
        self.df[f"cci{period}_ma{smoothing}"] = ccima
        return

    def relative_strength_index(self, period=14):
        """
        Calculates the Relative Strength Index (RSI) of a given time period.
        Generates self.df columns: rsi20

        Args:
            period (int): Length of each RS window. Default is 14.

        Returns:
            None
        """
        close = self.og_df['close'].copy().values

        diffs = np.diff(close, prepend=np.nan)

        gains = np.where(diffs > 0, diffs, 0)
        losses = np.where(diffs < 0, -diffs, 0)

        avg_gain = np.zeros_like(close)  # Length of DataFrame
        avg_loss = np.zeros_like(close)

        # Initializes first RSI value
        avg_gain[period] = np.mean(gains[1:period+1])  # Exclude the first NaN
        avg_loss[period] = np.mean(losses[1:period+1])

        rsi = np.full_like(close, np.nan)

        for i in range(period+1, len(close)):  # Skips first window
            # Calculates average gain/loss for each window
            avg_gain[i] = (avg_gain[i-1] * (period - 1) + gains[i]) / period
            avg_loss[i] = (avg_loss[i-1] * (period - 1) + losses[i]) / period

            if avg_loss[i] == 0:
                rs = np.inf  # Avoid division by zero
            else:
                # Calculate Relative Strength
                rs = avg_gain[i] / avg_loss[i]

            # Calculate RSI for period
            rsi[i] = 100 - (100 / (1 + rs))

        self.df[f"rsi{period}"] = rsi
        return

    def stdev_percent_of_sma(self, period=20):
        """
        Calculates the ratio of standard deviation to moving average for a given time period.
        Generates self.df columns: STDevPercent20

        Args:
            period (int): Length of the moving average and standard deviation window. Default is 20.

        Returns:
            None
        """
        series = self.og_df["close"].copy()

        sma = series.rolling(window=period).mean()
        std = series.rolling(window=period).std()

        pc_std = (std / sma) * 100

        self.df[f"STDevPercent{period}"] = pc_std
        return

    def bollinger_bands(self, period=20):
        """
        Calculates the Bollinger Bands of a given time period.
        Generates self.df columns: BollingerUpper20, BollingerSMA20, BollingerLower20, BollingerRange20

        Args:
            period (int): Length of the moving average and standard deviation window. Default is 20.

        Returns:
            None
        """
        series = self.og_df["close"].copy()

        sma = series.rolling(window=period).mean()
        std = series.rolling(window=period).std()

        upper_band = (sma + (std * 2))
        lower_band = (sma - (std * 2))

        self.df[f"bollingerUpper{period}"] = upper_band
        self.df[f"bollingerSMA{period}"] = sma
        self.df[f"bollingerLower{period}"] = lower_band
        self.df[f"bollingerRange{period}"] = upper_band - lower_band
        return

    def short_know_sure_thing(self, roc1=10, roc2=15, roc3=20, roc4=30, sma1=10, sma2=10, sma3=10, sma4=15, sig=9):
        """
        Calculates the short-term Know-sure-thing (KST) of a given time period.
        Generates self.df columns: kst, kstSig

        Args:
            roc1 (int): Rate of change period 1. Default is 10.
            roc2 (int): Rate of change period 2. Default is 15.
            roc3 (int): Rate of change period 3. Default is 20.
            roc4 (int): Rate of change period 4. Default is 30.
            sma1 (int): Moving average period 1. Default is 10.
            sma2 (int): Moving average period 2. Default is 10.
            sma3 (int): Moving average period 3. Default is 10.
            sma4 (int): Moving average period 4. Default is 15.
            sig (int): Length of the signal line moving average window. Default is 9.

        Returns:
            None
        """
        series = self.og_df["close"].copy()

        # Calculates rate of change for each period
        shift_1 = series.shift(roc1)
        roc_10 = ((series / shift_1) - 1) * 100

        shift_2 = series.shift(roc2)
        roc_15 = ((series / shift_2) - 1) * 100

        shift_3 = series.shift(roc3)
        roc_20 = ((series / shift_3) - 1) * 100

        shift_4 = series.shift(roc4)
        roc_30 = ((series / shift_4) - 1) * 100

        # Calculates moving averages of the rates of change
        rcma1 = roc_10.rolling(window=sma1).mean()
        rcma2 = roc_15.rolling(window=sma2).mean()
        rcma3 = roc_20.rolling(window=sma3).mean()
        rcma4 = roc_30.rolling(window=sma4).mean()

        # Calculates KST and signal line
        kst = (rcma1 * 1) + (rcma2 * 2) + (rcma3 * 3) + (rcma4 * 4)
        signal = kst.rolling(window=sig).mean()

        self.df["kst"] = kst
        self.df["kstSig"] = signal
        return

    def moving_average_convergence_divergence(self, ema_short=12, ema_long=26, min_periods=100):
        """
        Calculates the Moving Average Convergence Divergence (MACD).
        Generates self.df columns: macd, macdSig, macdHist

        Args:
        ema_short (int): Period of EMA short moving average. Default is 12.
        ema_long (int): Period of EMA long moving average. Default is 26.
        min_periods (int): Minimum number of periods required to calculate MACD. Default is 100.

        Returns:
            None
        """
        series = self.og_df["close"].copy()

        twelve = series.ewm(span=ema_short, min_periods=min_periods).mean()
        twenty_six = series.ewm(span=ema_long, min_periods=min_periods).mean()

        macd = twelve - twenty_six
        signal = macd.ewm(span=9).mean()

        self.df["macd"] = macd
        self.df["macdSig"] = signal
        self.df["macdHist"] = macd - signal
        return

    def williams_percent_r(self, period=14):
        """
        Calculates the Williams %R of a given time period.
        Generates self.df columns: williamsR14

        Args:
            period (int): Length of the rolling max/min window. Default is 14.

        Returns:
            None
        """
        df = self.og_df[["high", "low", "close"]].copy()

        maxes = df['high'].rolling(period).max()
        mins = df['low'].rolling(period).min()

        wpr = ((maxes - df['close']) / (maxes - mins)) * -100

        self.df[f"williamsR{period}"] = wpr
        return

    def stochastic_oscillator(self, period=20):
        """
        Calculates the stochastic oscillator of a given time period.
        Generates self.df columns: stochasticK20, stochasticD20

        Args:
            period (int): Length of the rolling max/min window. Default is 20.

        Returns
            None
        """
        df = self.og_df[["high", "low", "close"]].copy()

        mins = df['low'].rolling(period).min()
        maxes = df['high'].rolling(period).max()

        # Calculates percentK and percentD
        percent_k = (df['close'] - mins) / (maxes - mins) * 100
        percent_d = percent_k.rolling(3).mean()

        self.df[f"stochasticK{period}"] = percent_k
        self.df[f"stochasticD{period}"] = percent_d
        return

    def parabolic_sar(self, start=0.02, increment=0.02, maximum=0.2):
        """
        Calculates the Parabolic SAR at standard parameters.
        Generates self.df columns: psar

        Args:
            start (float): Starting value. Default is 0.02.
            increment (float): Increment value. Default is 0.02.
            maximum (float): Maximum value. Default is 0.2.

        Returns:
            None
        """
        df = self.og_df[["high", "low"]].copy()

        af_init = start
        af_max = maximum

        high = df['high'].values
        low = df['low'].values

        index = [0]
        # Initialize trend direction with first 2 rows
        if high[1] > high[0]:
            sar = [min(low[:2])]
            ep = [max(high[:2])]
            af = [af_init]
            trend = ["up"]
        else:
            sar = [max(high[:2])]
            ep = [min(low[:2])]
            af = [af_init]
            trend = ["down"]

        for i in range(len(df) - 2):
            x = i + 2
            y = i + 1
            if trend[-1] == "up":
                sar_m = sar[-1] + af[-1] * (ep[-1] - sar[-1])
                sar.append(min(sar_m, low[y], low[i]))
                if sar[-1] > low[x]:  # Trend up to down
                    sar[-1] = max(high[index[-1]:x])
                    ep.append(low[x])
                    af.append(af_init)
                    trend.append("down")
                    index.append(x)
                else:  # Trend remains up
                    if high[x] > ep[-1]:
                        ep.append(high[x])
                        af.append(min(af[-1] + increment, af_max))
                    trend.append("up")
            else:
                sar_m = sar[-1] - af[-1] * (sar[-1] - ep[-1])
                sar.append(max(sar_m, high[i], high[y]))
                if sar[-1] < high[x]:  # Trend down to up
                    sar[-1] = min(low[index[-1]:x])
                    ep.append(high[x])
                    af.append(af_init)
                    trend.append("up")
                    index.append(x)
                else:  # Trend remains down
                    if low[x] < ep[-1]:
                        ep.append(low[x])
                        af.append(min(af[-1] + increment, af_max))
                    trend.append("down")
        sar.insert(0, None)
        sar = pd.Series(sar)

        self.df["psar"] = sar
        return

    def average_directional_index(self, smoothing=14, min_periods=100):
        """
        Calculates the Average Directional Index (ADX) of a given time period.
        Generates self.df columns: adx, dmiPlus, dmiMinus

        Args:
            smoothing (int): Denominator of alpha. Default is 14.
            min_periods (int): Minimum number of periods required to calculate ADX. Default is 100.

        Returns:
            None
        """
        df = self.og_df[["high", "low", "close"]].copy()

        alpha = 1/smoothing

        # TR
        h_l = df['high'] - df['low']
        h_c = np.abs(df['high'] - df['close'].shift(1))
        l_c = np.abs(df['low'] - df['close'].shift(1))
        tr = pd.concat([h_l, h_c, l_c], axis=1).max(axis=1)

        # ATR
        atr = tr.ewm(alpha=alpha, adjust=False, min_periods=min_periods).mean()

        # DX+-
        h_ph = df['high'] - df['high'].shift(1)
        pl_l = df['low'].shift(1) - df['low']
        plus_dx = pd.Series(
            np.where(
                (h_ph > pl_l) & (h_ph > 0),
                h_ph,
                0.0
            )
        )
        minus_dx = pd.Series(
            np.where(
                (h_ph < pl_l) & (pl_l > 0),
                pl_l,
                0.0
            )
        )

        # DMI+-
        s_plus_dm = plus_dx.ewm(
            alpha=alpha, adjust=False, min_periods=min_periods
        ).mean()
        s_minus_dm = minus_dx.ewm(
            alpha=alpha, adjust=False, min_periods=min_periods
        ).mean()
        dmi_plus = (s_plus_dm/atr)*100
        dmi_minus = (s_minus_dm/atr)*100

        # DX & ADX
        dx = (np.abs(dmi_plus - dmi_minus)/(dmi_plus + dmi_minus))*100
        adx = dx.ewm(alpha=alpha, adjust=False).mean()

        self.df["adx"] = adx
        self.df["dmiPlus"] = dmi_plus
        self.df["dmiMinus"] = dmi_minus
        return

    def exponential_moving_average(self, period: int, column="close", min_periods=120):
        """
        Calculates the Exponential Moving Average (EMA) of a given time period.
        Generates self.df columns: ema10

        Args:
            period (int): Number of periods to calculate EMA for.
            column (str): Column to calculate EMA for. Default is 'close'.
            min_periods (int): Minimum number of periods required to calculate EMA. Default is 120.

        Returns:
            None
        """
        self.df[f"ema{period}"] = self.og_df[column].copy().ewm(span=period, min_periods=min_periods).mean()
        return

    def kaufman_adaptive_moving_average(self, data: pd.Series, period: int, min_periods=120, apply=True):
        """
        Calculates Kaufman's Adaptive Moving Average (KAMA) of a given time period with Average True Range (ATR) normalization.
        Generates self.df columns: kama20

        Args:
            data (pd.Series): Time series to calculate KAMA for.
            period (int): Number of periods to calculate KAMA for.
            min_periods: Minimum number of periods required to calculate KAMA. Default is 120.
            apply: Whether to apply the result to self.df or return it as a stand-alone Pandas Series. Default is True.

        Returns:
            None
        """
        atr = self.df["atr20"].copy()

        change = np.abs(data.diff(period).to_numpy())
        # Volatility moving average
        volatility = np.abs(np.diff(data)).astype(float)
        volatility = pd.Series(volatility).rolling(window=period).sum().to_numpy()
        volatility = np.insert(volatility, 0, np.nan)

        # Efficiency ratio
        er = np.zeros_like(data, dtype=float)
        mask = volatility > 0
        er[mask] = change[mask] / volatility[mask]

        # Uses normalized Average True Range to dynamically scale fast and slow periods
        atr_normalized = (atr / atr.rolling(window=min_periods).mean()).to_numpy()

        # Ignores Numpy runtime warning
        with np.errstate(invalid="ignore"):
            # Dynamically calculate fast and slow period
            fast_period = np.maximum(2, 5 - atr_normalized * 3).astype(int)
            slow_period = np.maximum(20, 30 + atr_normalized * 10).astype(int)

        # Smoothing constants
        fast_sc = 2.0 / (fast_period + 1)
        slow_sc = 2.0 / (slow_period + 1)
        sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2

        adaptive_ma = np.zeros_like(data, dtype=float)

        start_idx = period
        if start_idx < len(data):
            adaptive_ma[start_idx-1] = np.mean(data[:start_idx])

            # Iterates through data points, calculating KAMA for each
            for i in range(start_idx, len(data)):
                if not np.isnan(data[i]) and not np.isnan(adaptive_ma[i-1]):
                    adaptive_ma[i] = adaptive_ma[i-1] + sc[i] * (data[i] - adaptive_ma[i-1])
                elif not np.isnan(adaptive_ma[i-1]):
                    adaptive_ma[i] = adaptive_ma[i-1]
        adaptive_ma[adaptive_ma == 0.000000] = np.nan

        if apply:
            self.df[f"kama{period}"] = adaptive_ma
        else:
            return adaptive_ma

    def on_balance_volume(self, period=10):
        """
        Calculates the On-Balance Volume (OBV) of a given time period.
        Generates self.df columns: OBV

        Args:
            period (int): Number of periods to calculate OBV for. Default is 10.

        Returns:
            None
        """
        close = self.og_df['close'].copy().values
        volume = self.og_df['volume'].copy().values

        price_diff = np.diff(close)
        direction = np.sign(price_diff)
        direction = np.insert(direction, 0, 0)
        obv_changes = direction * volume
        obv = np.sum(np.lib.stride_tricks.sliding_window_view(obv_changes, period), axis=1)
        obv = np.insert(obv, range(period-1), np.nan)

        self.df["OBV"] = obv
        return

    # Advanced calculations
    def current_trend(self, periods: dict = {"twoWeek": 10, "month": 20, "threeMonth": 60, "sixMonth": 120, "year": 250,
                                            "threeYear": 750}, y_col: str = "KAMA20"):
        """
        Determines the current trend by calculating line best fit at different periods, averaging them, and taking percentiles.
        Generates self.df columns: {key}TrendLine for key in periods dict, prevailingTrendLine.

        Args:
            periods (dict): Keys are corresponding column names, values are the number of time periods for the line best fit.
            y_col: Column used to create trend lines. Default is "KAMA20".

        Returns:
            None
        """
        Y = self.df[y_col].copy().values
        Y = utils.rolling_zscore(Y)

        X_dict = {}
        for name, period in periods.items():
            if len(Y) < period + 20:
                continue
            slopes = pd.Series(utils.lineBestFit(Y, period))
            X_dict[f"{name}TrendLine"] = slopes

        trends_df = pd.DataFrame(X_dict)
        trends_df["prevailingTrendLine"] = trends_df["twoWeekTrendLine"]

        for column in list(trends_df.columns) + ["prevailingTrendLine"]:
            X = trends_df[column]

            if len(X) < 20:
                continue
            try:
                percentile_up = utils.percentile(X, 75)
            except IndexError:
                percentile_up = np.inf

            try:
                percentile_down = utils.percentile(X, 25)
            except IndexError:
                percentile_down = -np.inf

            conditions = [
                (X >= percentile_up),
                (X <= percentile_down)
            ]
            res = np.select(conditions, [1, -1])

            new_column = "".join(re.findall('[a-zA-Z][^A-Z]*', column)[:-1])
            trends_df[new_column] = res

        self.df = pd.concat([self.df, trends_df], axis=1)

    def trend_strength_score(self, weights: dict = {"closeDiff": 25, "movingAverageConvergenceDivergence": 30,
                                                    "dmiDiff": 25, "OBV": 20, }):
        """
        Creates a trend strength score based on a variety of factors.
        Generates self.df columns: trendScore

        Args:
            weights (dict): Keys are columns to be used as weights, values are the weight of the column.

        Returns:
            None
        """

        subdf = self.df[[
            "close",
            "movingAverageConvergenceDivergence",
            "macdHist",
            "dmiPlus",
            "dmiMinus",
            "OBV",
            "averageDirectionalIndex",
            "averageTrueRange20",
            "bollingerRange20"
        ]].copy()

        subdf["closeDiff"] = utils.rolling_zscore(subdf["close"].diff())
        subdf["dmiDiff"] = subdf["dmiPlus"] - subdf["dmiMinus"]
        subdf["OBV"] = utils.rolling_zscore(subdf["OBV"])

        # Scales some subdf values from -1 to 1
        subdf[[
            "closeDiff",
            "movingAverageConvergenceDivergence",
            "dmiDiff",
            "OBV"
        ]] = 2 * utils.scaler(
            "MinMax",
            subdf[["closeDiff", "movingAverageConvergenceDivergence", "dmiDiff", "OBV"]],
            return_as="pandas"
        ) - 1

        # Scales other subdf values from 0 to 1
        subdf[[
            "averageDirectionalIndex",
            "averageTrueRange20",
            "bollingerRange20"
        ]] = utils.scaler(
            "MinMax",
            subdf[["averageDirectionalIndex", "averageTrueRange20", "bollingerRange20"]],
            return_as="pandas"
        )

        # Calculates initial trend scores
        weight_sum = sum(weights.values())
        trend_score = sum(
            subdf[col] * weight
            for col, weight in weights.items()
        ) / weight_sum

        # ADX multiplier (0-2)
        trend_score *= (subdf["averageDirectionalIndex"] * 2)

        # Takes adaptive MA
        subdf["trendScore"] = self.kaufman_adaptive_moving_average(trend_score, 10, apply=False)
        trend_score = subdf["trendScore"]

        # Separates positive trend scores and negative and scales them individually (-1 to 1)
        pos_scores = pd.Series(
            np.where(trend_score > 0, trend_score, np.nan)
        )
        neg_scores = pd.Series(
            np.where(trend_score < 0, abs(trend_score), np.nan)
        )

        scores = []
        for i, subset in enumerate([pos_scores, neg_scores]):
            subset_scaled = utils.scaler(
                "MinMax",
                subset,
                return_as="pandas"
            )
            if i == 1:
                subset_scaled = subset_scaled * -1

            scores.append(subset_scaled)
        self.df["trendScore"] = scores[0].fillna(scores[1])
        return

    def trend_swing_confirmation(self, window=5):
        """
        Calculates dynamic trend retracement targets based on trend swings.
        Generates self.df columns: retracementThresholdCrossover, retracementThreshold, retracementTrend, retraceConfidence

        Args:
            window (int): window size for masks.

        Returns:
            None
        """
        subdf = self.df[[
            "trendScore",
            "high",
            "low",
            "close",
            "averageDirectionalIndex",
            "macdHist",
            "bollingerSMA20",
            "averageTrueRange20"
        ]].copy()

        # Initializes masks for swing point detection
        high_mask = np.ones(len(subdf), dtype=bool)
        low_mask = np.ones(len(subdf), dtype=bool)

        for i in range(1, window + 1):
            # Vectorized comparison with shifted values
            high_shift = subdf['high'].shift(i).fillna(float('-inf')).values
            low_shift = subdf['low'].shift(i).fillna(float('inf')).values

            # Update masks for each position in the lookback window
            high_mask &= (subdf['high'].values >= high_shift)
            low_mask &= (subdf['low'].values <= low_shift)

        # Identifies high and low points in the lookback window
        high_indices = np.where(high_mask)[0]
        low_indices = np.where(low_mask)[0]

        # Initialize swing point columns
        subdf["swingHigh"] = np.nan
        subdf["swingLow"] = np.nan

        # Process high points
        last_high_idx = 0
        for idx in high_indices:
            if idx >= window and idx - last_high_idx >= window:
                subdf.iloc[idx, subdf.columns.get_loc("swingHigh")] = subdf['high'].iloc[idx]
                last_high_idx = idx

        # Process low points
        last_low_idx = 0
        for idx in low_indices:
            if idx >= window and idx - last_low_idx >= window:
                subdf.iloc[idx, subdf.columns.get_loc("swingLow")] = subdf['low'].iloc[idx]
                last_low_idx = idx

        # Forward-fill swing points
        subdf["lastSwingHigh"] = subdf["swingHigh"].ffill()
        subdf["lastSwingLow"] = subdf["swingLow"].ffill()

        # Handle null values by using the first high/low
        subdf["lastSwingHigh"] = subdf["lastSwingHigh"].fillna(subdf["high"].iloc[0])
        subdf["lastSwingLow"] = subdf["lastSwingLow"].fillna(subdf["low"].iloc[0])

        # Calculate swing range
        subdf["swingRange"] = subdf["lastSwingHigh"] - subdf["lastSwingLow"]

        # Base confidence from trend strength score
        confidence_factor = abs(subdf["trendScore"]).clip(0, 1)

        # Fibonacci retracement levels
        fib_shallow = 0.382
        fib_deep = 0.618

        # Create dynamic retracement level based on trend confidence
        subdf["fibRetracementLevel"] = fib_shallow + (confidence_factor * (fib_deep - fib_shallow))

        # Calculate retracement target price based on trend direction
        subdf["retracementTarget"] = np.where(
            subdf["trendScore"] > 0,
            subdf["lastSwingHigh"] - (subdf["swingRange"] * subdf["fibRetracementLevel"]),  # Support
            subdf["lastSwingLow"] + (subdf["swingRange"] * subdf["fibRetracementLevel"])  # Resistance
        )

        # Checks if the trend direction changes
        trend_direction_change = (np.sign(subdf["trendScore"]) != np.sign(subdf["trendScore"].shift(1))) & (subdf["trendScore"].shift(1) != 0)

        # Checks if the trend direction has changed in the last 10 periods
        recent_trend_change = trend_direction_change.rolling(10).sum() > 0

        # Conditions to identify a weak trend (low trend score or recent trend change)
        weak_trend = (abs(subdf["trendScore"]) < utils.percentile(abs(subdf["trendScore"]), 30)) | recent_trend_change

        # Volatility adjustment factor (less buffer for strong trends, more for weak trends)
        volatility_buffer = 4 - (confidence_factor * 0.5)

        subdf["retracementThreshold"] = np.where(
            subdf["trendScore"] > 0,
            # In uptrend: support lowered by volatility factor (creates buffer zone)
            subdf["retracementTarget"] - (subdf["averageTrueRange20"] * volatility_buffer),

            # In downtrend: resistance raised by volatility factor
            subdf["retracementTarget"] + (subdf["averageTrueRange20"] * volatility_buffer)
        )

        # Apply a slow and fast adaptive MA to support/resistance thresholds
        kama_fast = self.kaufman_adaptive_moving_average(subdf["retracementThreshold"], 10, apply=False)
        kama_slow = self.kaufman_adaptive_moving_average(subdf["retracementThreshold"], 20, apply=False)
        blend_weight = np.where(weak_trend, 0.95, 0.4)

        # Calculates the weighted average of the slow and fast adaptive MA thresholds with blend weight
        threshold = pd.Series(
            (kama_slow * blend_weight) + (kama_fast * (1 - blend_weight))
        )

        # Finds slope of the retracement threshold line
        retracement_slope = pd.Series(utils.lineBestFit(threshold, 10))

        # Finds thresholds for strong retracement up or down slopes
        slope_up_thresh = utils.percentile(retracement_slope, 80)
        slope_down_thresh = utils.percentile(retracement_slope, 20)

        close = self.df["close"]
        # When the threshold crosses over closing price in either direction
        reversal_conditions = (
            (threshold.shift(1) < close.shift(1)) & (threshold > close),
            (threshold.shift(1) > close.shift(1)) & (threshold < close)
        )

        # These threshold/close crossovers are considered trend reversals
        reversals = pd.Series(np.select(reversal_conditions, ["toDown", "toUp"], "None"))
        reversal_idxs = reversals[reversals != "None"].index

        # Confirms crossovers by checking if there are 30 periods between crossover events
        self.df["reversalConf"] = "None"
        for i, idx in enumerate(reversal_idxs):
            try:
                next_idx = reversal_idxs[i+1]
            except IndexError:
                next_idx = len(reversals) - 1

            slope_segment = retracement_slope.iloc[idx:next_idx]
            if len(slope_segment) < 30:
                continue

            # Further confirms crossovers by checking if the retracement threshold segment between crossover events contains a strong up/down slope
            reversal_type = reversals.iloc[idx]
            if reversal_type == "toUp":
                try:
                    pos_slope = slope_segment[slope_segment > slope_up_thresh].index[0]
                except IndexError:
                    continue
                self.df.loc[pos_slope, "reversalConf"] = reversal_type
            elif reversal_type == "toDown":
                try:
                    neg_slope = slope_segment[slope_segment < slope_down_thresh].index[0]
                except IndexError:
                    continue
                self.df.loc[neg_slope, "reversalConf"] = reversal_type

        self.df["retracementThresholdCrossover"] = reversals
        self.df["retracementThreshold"] = threshold
        self.df["retracementTrend"] = np.where(self.df["retracementThreshold"] > self.df["close"], -1, 1)
        self.df["retraceConfidence"] = confidence_factor
        return
