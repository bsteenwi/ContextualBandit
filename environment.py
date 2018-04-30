import numpy as np
import pandas as pd
from tqdm import tqdm

from bandit import ContextualBandit
from agent import KerasAgent
from utils import Mean_Log_Loss, High_Order_Iterative_Knockout


class Environment(object):
    def __init__(self, features, labels, policy):
        self.features = features
        self.labels = labels

        # init label rewards (adapt freely)
        m = np.zeros((len(self.labels), max(self.labels)+1))
        for i in range(0, len(self.labels)):
            m[i][self.labels[i]] = 1

        # label frame
        self.y = pd.DataFrame(m)

        # frame of features
        self.X = pd.DataFrame(features)
        self.X = self.X.reset_index().drop(columns=['index'])

        self.cBandit = ContextualBandit(self.X, self.y)
        self.myAgent = KerasAgent(lr=0.001,
                                  a_size=self.cBandit.num_actions,
                                  n_states=self.cBandit.num_state_features)

        self.policy = policy
        self.policy.setBandit(self.cBandit)
        self.policy.setAgent(self.myAgent)

    def iter(self):
        # classical bandit interaction
        # a) get state, b) perform action, c) get reward and update
        s, t = self.cBandit.getInputState()
        action = self.policy.select(s)

        reward = self.cBandit.pullArm(action)

        # Update the network.
        y = self.policy.qval[:]
        y[0][action] = reward
        self.myAgent.model.fit(s, y, batch_size=1, epochs=1, verbose=0)

        return t, action, reward

    def experiment(self, total_rounds=1000000):
        i = 0
        pbar = tqdm(total=total_rounds)
        while i < total_rounds:
            t, action, reward = self.iter()
            i += 1
            pbar.update(1)
        pbar.close()

        inputs = self.myAgent.model.predict(self.cBandit.X)
        probas = inputs.reshape(self.cBandit.num_samples, -1)
        predictions = np.argmax(probas, axis=1)

        accuracy = Mean_Log_Loss(predictions=predictions, labels=self.labels)
        self.output(accuracy, predictions)
        return predictions

    def output(self, accuracy, predictions):
        print("baseline accuracy: ", accuracy)
        print("predicitons: ", predictions)

        high_order_knockout, index = High_Order_Iterative_Knockout(
                                     features_knockout=np.array(self.features),
                                     model=self.myAgent.model,
                                     baseline=accuracy,
                                     labels=self.labels)

        print("high-order knockout accuracy change: ")
        Z = [(y, x) for y, x in sorted(zip(high_order_knockout, index),
             reverse=True, key=lambda l:(l[0], -len(l[1])))]
        for z in Z:
            print(z)
