import pandas as pd
import datetime as dt
import random
import util as ut
import indicators
import QLearner as ql
from marketsimcode import compute_portvals


class StrategyLearner(object):  		  	   		  		 			  		 			     			  	 

    def __init__(self, verbose=False, impact=0.0, commission=0.0):  		  	   		  		 			  		 			     			  	 
        """  		  	   		  		 			  		 			     			  	 
        Constructor method  		  	   		  		 			  		 			     			  	 
        """  		  	   		  		 			  		 			     			  	 
        self.verbose = verbose  		  	   		  		 			  		 			     			  	 
        self.impact = impact  		  	   		  		 			  		 			     			  	 
        self.commission = commission
        random.seed(903862212)
        self.learner = ql.QLearner(num_states=96, num_actions=3, alpha=0.2, gamma=0.9, rar=0.9, radr=0.99, dyna=100,
                                   verbose=False)

    def add_evidence(
        self,  		  	   		  		 			  		 			     			  	 
        symbol="IBM",  		  	   		  		 			  		 			     			  	 
        sd=dt.datetime(2008, 1, 1),  		  	   		  		 			  		 			     			  	 
        ed=dt.datetime(2009, 1, 1),  		  	   		  		 			  		 			     			  	 
        sv=10000,  		  	   		  		 			  		 			     			  	 
    ):
        # add your code to do learning here
        ema_20, ema_30, ema_50, macd, tsi = get_standardized_indicators(sd, ed, symbol)
        df_prices, df_trades = get_prices(sd, ed, symbol)
        df_trades = df_trades.rename(columns={'SPY': symbol}).astype({symbol: 'int32'})
        df_trades[:] = 0
        current_position = 0
        current_cash = sv
        previous_position = 0
        previous_cash = sv
        for price in range(1, len(df_prices.index)):
            current_day = df_prices.index[price]
            yesterday = df_prices.index[price - 1]
            s_prime = get_current_state(current_position, ema_20.loc[current_day],
                                        ema_30.loc[current_day], ema_50.loc[current_day], macd.loc[current_day], tsi.loc[current_day])
            calc = current_position * df_prices.loc[current_day].loc[symbol] + current_cash - previous_position * df_prices.loc[current_day].loc[
                symbol] - previous_cash
            next_step = self.learner.query(s_prime, calc)
            if next_step == 0:
                trade = -1000 - current_position
            elif next_step == 1:
                trade = -current_position
            else:
                trade = 1000 - current_position
            if self.verbose:
                print(current_day)
                print("Today's Position: {}".format(current_position))
                print("Today's Cash: {}".format(current_cash))
                print("Yesterday's Position: {}".format(previous_position))
                print("Yesterday's Cash: {}".format(previous_cash))
                print("Today's Price: " + str(df_prices.loc[current_day].loc[symbol]))
                print("Last Reward: " + str(calc))
                print("Trade: {}".format(trade))
                print()
            previous_position = current_position
            current_position += trade
            df_trades.loc[current_day].loc[symbol] = trade
            if trade > 0:
                impact = self.impact
            else:
                impact = -self.impact
            previous_cash = current_cash
            current_cash += -df_prices.loc[current_day].loc[symbol] * (1 + impact) * trade
        if self.verbose:
            print("--{} In Sample Benchmark--".format(symbol))
            print(get_benchmark(sd, ed, sv, self.impact).tail())
            print()
            print("--{} Performance During Training--".format(symbol))
            print(compute_portvals(df_trades, start_val=sv, commission=0, impact=0.000).tail())
            print()

    def testPolicy(  		  	   		  		 			  		 			     			  	 
        self,  		  	   		  		 			  		 			     			  	 
        symbol="IBM",  		  	   		  		 			  		 			     			  	 
        sd=dt.datetime(2009, 1, 1),  		  	   		  		 			  		 			     			  	 
        ed=dt.datetime(2010, 1, 1),  		  	   		  		 			  		 			     			  	 
        sv=10000,  		  	   		  		 			  		 			     			  	 
    ):
        ema_20, ema_30, ema_50, macd, tsi = get_standardized_indicators(sd, ed, symbol)
        df_prices, df_trades = get_prices(sd, ed, symbol)
        df_trades = df_trades.rename(columns={'SPY': symbol}).astype({symbol: 'int32'})
        df_trades[:] = 0
        position = 0
        for price in range(1, len(df_prices.index)):
            curr_day = df_prices.index[price]
            yesterday = df_prices.index[price - 1]
            s_prime = get_current_state(position, ema_20.loc[curr_day],
                                        ema_30.loc[curr_day], ema_50.loc[curr_day], macd.loc[curr_day], tsi.loc[curr_day])
            next_step = self.learner.querysetstate(s_prime)
            if next_step == 0:
                trade = -1000 - position
            elif next_step == 1:
                trade = -position
            else:
                trade = 1000 - position
            position += trade
            df_trades.loc[curr_day].loc[symbol] = trade
        if self.verbose:
            print("--{} Out Sample Benchmark]".format(symbol))
            print(get_benchmark(sd, ed, sv, self.impact).tail())
            print()
            print("--{} Performance During Testing--".format(symbol))
            print(compute_portvals(df_trades, start_val=sv, commission=0, impact=0.000).tail())
            print()
        return df_trades


def get_standardized_indicators(sd, ed, symbol):
    prices, sym = get_prices(sd, ed, symbol)
    ema_20 = indicators.ema(sd, ed, symbol, window=20)
    ema_30 = indicators.ema(sd, ed, symbol, window=30)
    ema_50 = indicators.ema(sd, ed, symbol, window=50)
    #binary signals
    ema_30 = (prices > ema_30) * 1
    ema_20 = (prices > ema_20) * 1
    ema_50 = (prices > ema_50) * 1
    macd_raw, macd_signal = indicators.macd(sd, ed, symbol)
    macd = (macd_raw > macd_signal) * 1
    tsi = indicators.tsi(sd, ed, symbol)
    tsi = (tsi > 0) * 1
    return ema_20, ema_30, ema_50, macd, tsi


def get_current_state(position, ema_20, ema_30, ema_50, macd, tsi):
    lock = 0
    if position == 0:
        lock += 32
    elif position == 1000:
        lock += 64
    lock += ema_20 * 16 + ema_30 * 8 + ema_50 * 4 + macd * 2 + tsi
    return int(lock)


def get_prices(sd, ed, symbol):
    symbol = [symbol]
    df = ut.get_data(symbol, pd.date_range(sd, ed))
    prices = df[symbol].ffill().bfill()
    spy = df[['SPY']]
    return prices, spy


def get_benchmark(sd, ed, sv, impact):
    df_trades = ut.get_data(['SPY'], pd.date_range(sd, ed)).rename(columns={'SPY': 'JPM'}).astype({'JPM': 'int32'})
    df_trades[:] = 0
    df_trades.loc[df_trades.index[0]] = 1000
    port_vals = compute_portvals(df_trades, sv, commission=0, impact= impact)
    return port_vals


def author():
    return 'aishwary'


if __name__ == "__main__":  		  	   		  		 			  		 			     			  	 
    print("Not Applicable")
