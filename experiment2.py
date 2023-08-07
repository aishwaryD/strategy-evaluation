import datetime as dt
import matplotlib.pyplot as plt
from marketsimcode import compute_portvals
from StrategyLearner import StrategyLearner


def experiment2():
    symbol = "JPM"
    sd = dt.datetime(2008, 1, 1)
    ed = dt.datetime(2009, 12, 31)
    sv = 100000
    learner_1 = StrategyLearner(verbose=False, impact=0.000)
    learner_1.add_evidence(symbol=symbol, sd=sd, ed=ed, sv=sv)
    df_trades_1 = learner_1.testPolicy(symbol=symbol, sd=sd, ed=ed, sv=sv)
    trades1_portvals = compute_portvals(df_trades_1, start_val=sv, commission=0, impact=0.000)
    learner_2 = StrategyLearner(verbose=False, impact=0.005)
    learner_2.add_evidence(symbol=symbol, sd=sd, ed=ed, sv=sv)
    df_trades_2 = learner_2.testPolicy(symbol=symbol, sd=sd, ed=ed, sv=sv)
    trades2_portvals = compute_portvals(df_trades_2, start_val=sv, commission=0, impact=0.005)
    learner_3 = StrategyLearner(verbose=False, impact=0.01)
    learner_3.add_evidence(symbol=symbol, sd=sd, ed=ed, sv=sv)
    df_trades_3 = learner_3.testPolicy(symbol=symbol, sd=sd, ed=ed, sv=sv)
    trades3_portvals = compute_portvals(df_trades_3, start_val=sv, commission=0, impact=0.01)
    plot_graph(trades1_portvals, trades2_portvals, trades3_portvals)
    long, short = get_trades(df_trades_1)
    plot_trades(trades1_portvals, long, short, "0", len(long) + len(short))
    long, short = get_trades(df_trades_3)
    plot_trades(trades3_portvals, long, short, "0.01", len(long) + len(short))


def plot_graph(trade_1, trade_2, trade_3):
    trade_1['value'] = trade_1['value'] / trade_1['value'][0]
    trade_2['value'] = trade_2['value'] / trade_2['value'][0]
    trade_3['value'] = trade_3['value'] / trade_3['value'][0]
    plt.figure(figsize=(14, 8))
    plt.xticks(rotation=30)
    plt.grid()
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return")
    plt.title("Experiment 2 | Impact Values")
    plt.plot(trade_1, label="impact: 0.000", color="green")
    plt.plot(trade_2, label="impact: 0.005", color="red")
    plt.plot(trade_3, label="impact: 0.01", color="blue")
    plt.legend()
    plt.savefig("images/experiment2.png", bbox_inches='tight')
    plt.clf()


def plot_trades(trade, long, short, label, total):
    #normalizing
    trade['value'] = trade['value'] / trade['value'][0]

    plt.figure(figsize=(14, 8))
    plt.xticks(rotation=30)
    plt.grid()
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return")
    plt.title("Experiment 2 Impact Value {}, {} Trades In Total".format(label, str(total)))
    plt.plot(trade, color="green")

    for date in long:
        plt.axvline(date, color="blue")

    for date in short:
        plt.axvline(date, color="black")

    plt.legend()
    plt.savefig("images/experiment2_{}_impact.png".format(label), bbox_inches='tight')
    plt.clf()


def get_trades(df_trades):
    current = 0
    short = []
    long = []
    step = 'OUT'
    for trade in df_trades.index:
        current += df_trades.loc[trade].loc['JPM']
        if current < 0:
            if step == 'OUT' or step == 'LONG':
                step = 'SHORT'
                short.append(trade)
        elif current > 0:
            if step == 'OUT' or step == 'SHORT':
                step = 'LONG'
                long.append(trade)
        else:
            step = 'OUT'
    return long, short


def author():
    return 'aishwary'


if __name__ == "__main__":
    experiment2()
