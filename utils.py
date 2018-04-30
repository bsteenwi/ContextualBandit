import numpy as np
import operator
import math
import copy
import itertools

'''
Freely adapted from Pudkip's:
A Method for Finding Feature Importance and High-Order interactions
link: https://github.com/Pudkip/Iterative-Knockout
'''


def Mean_Log_Loss(predictions, labels, limit=1):
    lim = 10 ** -limit
    cost = list(map(operator.sub, labels, predictions))
    cost_adj = []
    for i in range(len(cost)):
        cost_adj.append(-math.log10(abs(cost[i] + lim)))
    accuracy = sum(cost_adj) / len(cost_adj)
    return accuracy


def Single_Iterative_Knockout(features_knockout, model, labels, baseline):
    inp = copy.copy(features_knockout)
    accuracies = []
    for i in range(features_knockout.shape[1]):
        for j in range(features_knockout.shape[0]):
            features_knockout[j][i] = 0
        predictions = model.predict(features_knockout)
        predictions = predictions.reshape(features_knockout.shape[0], -1)
        accuracy = Mean_Log_Loss(predictions=np.argmax(predictions, axis=1),
                                 labels=labels)
        accuracies.append(accuracy)
        features_knockout = copy.copy(inp)
    accuracies[:] = [abs(x - baseline) for x in accuracies]

    return accuracies


def High_Order_Iterative_Knockout(features_knockout, model, labels, baseline,):
    inp = copy.copy(features_knockout)
    accuracies = []
    iter_list = list(range(features_knockout.shape[1]))
    combinations = []
    for k in range(features_knockout.shape[1]):
        combinations.append(list(itertools.combinations(iter_list, k)))
    combinations = list(itertools.chain.from_iterable(combinations[1:]))
    combinations = [list(item) for item in combinations]
    for i in range(len(combinations)):
        for j in range(features_knockout.shape[0]):
                features_knockout[j][combinations[i]] = [0]
        predictions = model.predict(features_knockout)
        predictions = predictions.reshape(features_knockout.shape[0], -1)
        accuracy = Mean_Log_Loss(predictions=np.argmax(predictions, axis=1),
                                 labels=labels)
        accuracies.append(accuracy)
        features_knockout = copy.copy(inp)
    accuracies[:] = [abs(x - baseline) for x in accuracies]

    return accuracies, combinations
