import pandas as pd
from util import get_data


def compute_portvals(
        orders_df,
        start_val=1000000,
        commission=9.95,
        impact=0.005,
):
    start_date = orders_df.index.min()
    end_date = orders_df.index.max()

    symbols = {}
    shares_delivery = {}
    cash = start_val

    portvals = get_data(['SPY'], pd.date_range(start_date, end_date), addSPY=True, colname='Adj Close')
    portvals = portvals.rename(columns={'SPY': 'value'})
    dates = portvals.index

    symbol = orders_df.columns[0]

    for date in dates:
        trade = orders_df.loc[date].loc[symbol]
        if trade != 0:
            if trade < 0:
                order = 'SELL'
                shares = abs(trade)
            else:
                order = 'BUY'
                shares = trade

            cash, shares_delivery, symbols = update_cash(symbol, order, shares, cash, shares_delivery, symbols, date, end_date,
                                  commission, impact)

        shares_value = 0
        for symbol in shares_delivery:
            shares_value += symbols[symbol].loc[date].loc[symbol] * shares_delivery[symbol]
        portvals.loc[date].loc['value'] = cash + shares_value
        # if np.isnan(portvals.iloc[0]):
        #     raise ValueError('Portfolio values cannot be NaNs!')
    return portvals


def update_cash(symbol, order, shares, cash, shares_delivery, symbols, date, end_date, commission,
                impact):
    if symbol not in symbols:
        symbol_df = get_data([symbol], pd.date_range(date, end_date), addSPY=True, colname='Adj Close')
        symbol_df.fillna(method='ffill', inplace=True)
        symbol_df.fillna(method='bfill', inplace=True)
        symbols[symbol] = symbol_df

    if order == 'BUY':
        share_diff = shares
        cash_diff = -symbols[symbol].loc[date].loc[symbol] * (1 + impact) * shares
    elif order == 'SELL':
        share_diff = -shares
        cash_diff = symbols[symbol].loc[date].loc[symbol] * (1 - impact) * shares
    else:
        print('Invalid Order')
    shares_delivery[symbol] = shares_delivery.get(symbol, 0) + share_diff
    cash += cash_diff - commission
    return cash, shares_delivery, symbols


def author():
    return 'aishwary'


if __name__ == "__main__":
    pass
