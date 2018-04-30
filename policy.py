import numpy as np


class Policy(object):
    def __init__(self):
        self.bandit = None
        self.agent = None

    def setBandit(self, bandit):
        self.bandit = bandit

    def setAgent(self, agent):
        self.agent = agent

    def select(self, state):
        self.qval = self.agent.getQValuesOfState(state)


class EpsilonGreedyPolicy(Policy):
    def __init__(self, epsilon):
        Policy.__init__(self)
        self.epsilon = epsilon

    def select(self, state):
        super(EpsilonGreedyPolicy, self).select(state)
        if np.random.rand() < self.epsilon:
            action = np.random.randint(self.bandit.num_actions)
        else:
            # best action
            action = np.argmax(self.qval)
        return action
