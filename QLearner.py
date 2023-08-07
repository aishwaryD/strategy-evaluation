import random as rand
import numpy as np


class QLearner(object):
    """
    This is a Q learner object.

    :param num_states: The number of states to consider.
    :type num_states: int
    :param num_actions: The number of actions available..
    :type num_actions: int
    :param alpha: The learning rate used in the update rule. Should range between 0.0 and 1.0 with 0.2 as a typical value.
    :type alpha: float
    :param gamma: The discount rate used in the update rule. Should range between 0.0 and 1.0 with 0.9 as a typical value.
    :type gamma: float
    :param rar: Random action rate: the probability of selecting a random action at each step. Should range between 0.0 (no random actions) to 1.0 (always random action) with 0.5 as a typical value.
    :type rar: float
    :param radr: Random action decay rate, after each update, rar = rar * radr. Ranges between 0.0 (immediate decay to 0) and 1.0 (no decay). Typically 0.99.
    :type radr: float
    :param dyna: The number of dyna updates for each regular update. When Dyna is used, 200 is a typical value.
    :type dyna: int
    :param verbose: If “verbose” is True, your code can print out information for debugging.
    :type verbose: bool
    """

    def __init__(self, num_states=100, num_actions=4, alpha=0.2, gamma=0.9, rar=0.5, radr=0.99, dyna=0, verbose=False):
        self.verbose = verbose
        self.num_states = num_states
        self.num_actions = num_actions
        self.alpha = alpha
        self.gamma = gamma
        self.rar = rar
        self.radr = radr
        self.dyna = dyna
        self.q = np.zeros((num_states, num_actions))
        self.s = 0
        self.a = 0
        if dyna > 0:
            self.r = np.zeros((num_states, num_actions))
            self.t = np.full((num_states, num_actions, num_states), 0.00001)
            self.records = {}
            self.optimal_state = {}

    def author(self):
        return 'aishwary'

    def querysetstate(self, s):
        self.s = s
        self.a = np.argmax(self.q[s])
        return self.a

    def query(self, s_prime, r):
        self.q[self.s, self.a] = (1 - self.alpha) * self.q[self.s, self.a] + self.alpha * (
                    r + self.gamma * np.max(self.q[s_prime]))
        if self.dyna > 0:
            self.records[self.s] = list(set().union(self.records.get(self.s, []), [self.a]))
            self.t[self.s, self.a, s_prime] += 1
            self.r[self.s, self.a] = (1 - self.alpha) * self.r[self.s, self.a] + self.alpha * r
            self.optimal_state = {}
            for _ in range(self.dyna):
                self.hallucinate_experience()
        action = rand.randint(0, self.num_actions - 1) if rand.random() < self.rar else np.argmax(self.q[s_prime])
        self.rar = self.rar * self.radr
        self.a = action
        self.s = s_prime
        if self.verbose:
            print(f"Last Action: {self.a}, Last State: {self.s}, New State: {s_prime}, Action: {action}, Reward: {r}")
        return self.a

    def hallucinate_experience(self):
        s = rand.choice(list(self.records.keys()))
        a = rand.choice(self.records[s])
        s_prime = self.optimal_state[(s, a)] if (s, a) in self.optimal_state else np.argmax(self.t[s, a])
        self.optimal_state[(s, a)] = s_prime if (s, a) not in self.optimal_state else self.optimal_state[(s, a)]
        r = self.r[s, a]
        self.q[s, a] = (1 - self.alpha) * self.q[s, a] + self.alpha * (r + self.gamma * np.max(self.q[s_prime]))


if __name__ == "__main__":
    print("Not Applicable")
