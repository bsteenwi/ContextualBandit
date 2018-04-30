import numpy as np


class ContextualBandit(object):
    def __init__(self, X, y):
        self.X = X
        self.y = y

        # number of different states
        self.num_state_features = X.shape[1]
        # number of different actions
        self.num_actions = max(y)+1
        # number of samples
        self.num_samples = X.shape[0]

        self.index = 0

    def getInputState(self):
        arr = np.array(self.X.iloc[self.index].tolist())
        return(arr.reshape(1, self.num_state_features), self.index)

    def pullArm(self, action):
        reward = self.y.iat[self.index, action]
        self.index = (self.index + 1) % self.num_samples

        # change freely
        if reward == 1:
            return 1
        else:
            return -1
