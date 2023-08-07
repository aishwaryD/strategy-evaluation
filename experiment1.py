import datetime as dt
import matplotlib.pyplot as plt
from ManualStrategy import ManualStrategy
from StrategyLearner import StrategyLearner
from marketsimcode import compute_portvals


def experiment1():
    symbol = 'JPM'
    sd = dt.datetime(2008, 1, 1)
    ed = dt.datetime(2009, 12, 31)
    sv = 100000
    manual_trades = ManualStrategy().testPolicy([symbol], sd=sd, ed=ed, sv=sv)
    manual_portvals = compute_portvals(manual_trades, start_val=sv, commission=0, impact=0.000)
    print_stats(manual_portvals, "Manual Trader")
    learner = StrategyLearner(verbose=False, impact=0.000)
    learner.add_evidence(symbol=symbol, sd=sd, ed=ed, sv = sv)
    learner_trades = learner.testPolicy(symbol = symbol, sd=sd, ed=ed, sv = sv)
    learner_portvals = compute_portvals(learner_trades, start_val = sv, commission=0, impact=0.000)
    print_stats(learner_portvals, "Strategy Learner")
    plot_graph(manual_portvals, learner_portvals)


def plot_graph(manual_portvals, strategy_portvals):
    strategy_portvals['value'] = strategy_portvals['value'] / strategy_portvals['value'][0]
    manual_portvals['value'] = manual_portvals['value'] / manual_portvals['value'][0]
    plt.figure(figsize=(14, 8))
    plt.xticks(rotation=30)
    plt.grid()
    plt.title("Manual VS Q Learning Bot")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return")
    plt.plot(manual_portvals, label="manual", color="green")
    plt.plot(strategy_portvals, label="q learning bot", color="red")
    plt.legend()
    plt.savefig("images/experiment1.png", bbox_inches='tight')
    plt.clf()


def print_stats(port_val, name):
    port_val = port_val['value']
    cr = port_val[-1] / port_val[0] - 1
    dr = (port_val / port_val.shift(1) - 1).iloc[1:]
    adr = dr.mean()
    sddr = dr.std()
    print("--" + name + "--")
    print("Cumulative Return: " + str(cr))
    print("Standard Deviation of Daily Returns: " + str(sddr))
    print("Mean of Daily Returns: " + str(adr))
    print()


def author():
    return 'aishwary'


if __name__=="__main__":
    experiment1()
