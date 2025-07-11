#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 19:45:46 2024

@author: cjymain
"""

from datetime import timedelta
import calendar


def market_calendar_check(d):
    import pandas_market_calendars as mcal

    nyse = mcal.get_calendar("NYSE").schedule(
        start_date=str(d), end_date=str(d + timedelta(days=7))
    ).reset_index()
    
    trading_dates = [d8.date() for d8 in nyse['index']]

    today = trading_dates[0]
    if today == d:
        return True
    else:
        return False

def which_friday(d):
    if d.weekday() != 4:
        return False
    else:
        friday = 0
        day_of_month = d.replace(day=1)
        while day_of_month <= d:
            if day_of_month.weekday() == 4:
                friday += 1
            day_of_month += timedelta(days=1)
        return friday

