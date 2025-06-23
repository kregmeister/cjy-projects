#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 11:11:44 2025

@author: cjymain
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from technically.utils.optimizations import bucketizer


def double_line_plot(x, y1, y2, line1label: str, line2label: str, xlabel: str, ylabel: str, title: str):
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(16, 8), gridspec_kw={'height_ratios': [3, 1]})


    line1, = ax1.plot(x, y1, alpha=0.4, color="red")
    line1.set_label(line1label)
    
    line2, = ax2.plot(x, y2, alpha=0.4)
    line2.set_label(line2label)
    
    ax1.legend()
    ax2.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel)
    ax1.set_title(title)
    ax1.grid()
    ax2.grid()
    plt.show()
    
    #results_lst = [[date, act, pred] for date, act, pred in zip(x, y1, y2) if act > pred]
    #print(results_lst)

def scatter_plot_shared_ax(x, y1, y1a, z, y1b, y2, sort_col):
    z = x[z != "None"]

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(16, 8),
                                   gridspec_kw={'height_ratios': [3, 1]})

    colors = {-1: 'red', 1: 'green'}
    df = pd.DataFrame({"date": x, "close": y1, "kalmanTrend": sort_col})
    df["kalmanTrend"] = np.where(df["kalmanTrend"] > 0, 1, -1)
    df["date"] = pd.to_datetime(df["date"])

    #for key, group in df.groupby("kalmanTrend"):
    #    ax1.scatter(group["date"], group["close"], color=colors[key], label=str(key))
    ax1.plot(x, y1, color="blue", label="Close")
    ax1.plot(x, y1a, color="red", label="Trend Retracement Threshold")
    ax1.plot(x, y1b, color="green", label="KAMA 20")
    ax2.plot(x, y2, color="purple", alpha=0.7)

    ymin, ymax = ax1.get_ylim()
    ax1.vlines(x=z, ymin=ymin, ymax=ymax, linestyle="--", color="black")

    # Add labels and title
    ax2.set_xlabel('Date')
    ax1.set_ylabel('Close')
    ax2.set_ylabel('Trend Score')
    ax2.set_xticks(x.iloc[::40])
    for label in ax2.get_xticklabels():
        label.set_rotation(90)
    ax1.set_title('Close vs Date with Trend Score')

    ax1.grid()
    ax2.grid()
    ax1.legend()
    plt.show()

def scatter_plot_conditional_coloring(df: pd.DataFrame, column: str):
    # Create figure and axes
    fig, ax = plt.subplots(figsize=(16, 8))

    df['date'] = pd.to_datetime(df['date'])

    # Define colors for each category
    colors = {-1: 'red', 0: 'blue', 1: 'green'}

    # Group by category and plot each group
    for key, group in df.groupby(column):
        ax.scatter(group['date'], group['close'], color=colors[key], label=str(key))
    plt.plot(df['date'], df['close'], color='0.7')

    # Add labels and title
    ax.set_xlabel('Date')
    ax.set_ylabel('Close')
    ax.set_xticks(df['date'].iloc[::40])
    for label in ax.get_xticklabels():
        label.set_rotation(90)
    ax.set_title(f'Close vs Date by {column}')

    plt.grid()
    plt.legend()
    plt.show()

def histogram_trend_distribution(df):
    # Histogram of trend distribution
    dfs = bucketizer(df)

    for b in dfs:
        print(b["prevailingTrend"].value_counts(normalize=True))
