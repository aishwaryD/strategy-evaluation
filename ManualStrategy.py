from util import get_data
import pandas as pd
import indicators


class ManualStrategy:

    def testPolicy(self, symbol, sd, ed, sv):
        symbol = symbol[0]
        df = get_data([symbol], pd.date_range(sd, ed))
        price = df[[symbol]].ffill().bfill()
        normalized_price = price[symbol] / price[symbol][0]
        df_trades = df[['SPY']].rename(columns={'SPY': symbol}).astype({symbol: 'int32'}).assign(**{symbol: 0})
        ema_20 = indicators.ema(sd, ed, symbol, plot=True, window=20)[symbol] / indicators.ema(sd, ed, symbol, plot=True, window=20)[symbol][0]
        tsi = indicators.tsi(sd, ed, symbol, plot=True)
        macd_raw, macd_signal = indicators.macd(sd, ed, symbol, plot=True)
        position = 0
        step = 0
        for trade in range(len(df_trades.index)):
            present_day = df_trades.index[trade]
            step += 1
            normalized_price_today = normalized_price.loc[present_day]
            ema_20_today = ema_20.loc[present_day]
            if normalized_price_today > ema_20_today:
                ema_adjusted = 1
            elif normalized_price_today < ema_20_today:
                ema_adjusted = -1
            else:
                ema_adjusted = 0
            macd_raw_today = macd_raw.loc[present_day].loc[symbol]
            macd_signal_today = macd_signal.loc[present_day].loc[symbol]
            if macd_signal_today > macd_raw_today:
                macd_adjusted = 2
            elif macd_signal_today < macd_raw_today:
                macd_adjusted = -10
            else:
                macd_adjusted = 1
            tsi_today = tsi.loc[present_day].loc[symbol]
            if tsi_today > 0.1:
                tsi_adjusted = 1
            elif tsi_today < 0.1:
                tsi_adjusted = -1
            else:
                tsi_adjusted = 0
            if (macd_adjusted + tsi_adjusted + ema_adjusted) >= 3:
                act = 1000 - position
            elif (macd_adjusted + tsi_adjusted + ema_adjusted) <= -3:
                act = - 1000 - position
            else:
                act = -position
            if step >= 3:
                df_trades.loc[df_trades.index[trade]].loc[symbol] = act
                position = position + act
                step = 0 #reset
        return df_trades


def author():
    return 'aishwary'

#for testing


def print_stat(benchmark_portvals, theoretical_portvals):
    benchmark, theoretical = benchmark_portvals['value'], theoretical_portvals['value']
    dret_ben = (benchmark_portvals / benchmark_portvals.shift(1) - 1).iloc[1:]
    dret_the = (theoretical / theoretical.shift(1) - 1).iloc[1:]
    cret_ben = benchmark_portvals[-1] / benchmark_portvals[0] - 1
    cret_the = theoretical[-1] / theoretical[0] - 1
    adret_ben = dret_ben.mean()
    adret_the = dret_the.mean()
    sddret_ben = dret_ben.std()
    sddret_the = dret_the.std()
    print("")
    print("--Manual Strategy Execution--")
    print("Cumulative Return: " + str(cret_the))
    print("Standard Deviation of Daily Returns: " + str(sddret_the))
    print("Mean of Daily Returns: " + str(adret_the))
    print("")
    print("--Our Benchmark--")
    print("Cumulative Return: " + str(cret_ben))
    print("Standard Deviation of Daily Returns: " + str(sddret_ben))
    print("Mean of Daily Returns: " + str(adret_ben))
    print("")


if __name__ == "__main__":
    print("Not Applicable")
