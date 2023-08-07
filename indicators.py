import pandas as pd
from util import get_data
import datetime as dt


def author():
    return 'aishwary'


def ema(sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), symbol='JPM', plot=False, window=20):
    """Third indicator: Exponential Moving Average"""
    prices = get_data([symbol], pd.date_range(sd - dt.timedelta(window * 2), ed))
    prices.fillna(method='ffill', inplace=True)
    prices.fillna(method='bfill', inplace=True)
    if symbol != 'SPY':
        prices.drop(['SPY'], axis=1, inplace=True)
    ema_df = prices[[symbol]].ewm(span=window, adjust=False).mean()[sd:]
    return ema_df


def macd(sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), symbol='JPM', plot=False):
    """Fourth indicator: MACD"""
    df_price = get_data([symbol], pd.date_range(sd - dt.timedelta(52), ed))
    df_price = df_price[[symbol]].ffill().bfill()
    ema1 = df_price.ewm(span=12, adjust=False).mean()
    ema2 = df_price.ewm(span=26, adjust=False).mean()
    macd_raw = ema1 - ema2
    macd_signal = macd_raw.ewm(span=9, adjust=False).mean()
    df_price = df_price[sd:]
    ema1 = ema1[sd:]
    ema2 = ema2[sd:]
    macd_raw = macd_raw[sd:]
    macd_signal = macd_signal[sd:]
    return macd_raw, macd_signal


def tsi(sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 12, 31), symbol='JPM', plot=False):
    """Fifth indicator: TSI"""
    price = get_data([symbol], pd.date_range(sd-dt.timedelta(50), ed))[[symbol]].ffill().bfill()
    diff = price.diff()
    ema1 = diff.ewm(span=25, adjust=False).mean()
    ema2 = ema1.ewm(span=13, adjust=False).mean()
    abs_diff = abs(diff)
    abs_ema1 = abs_diff.ewm(span=25, adjust=False).mean()
    abs_ema2 = abs_ema1.ewm(span=13, adjust=False).mean()
    tsi_df = ema2 / abs_ema2
    tsi_df = tsi_df[sd:]
    return tsi_df


if __name__ == "__main__":
    pass
