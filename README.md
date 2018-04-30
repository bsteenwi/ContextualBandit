# ContextualBandit
Contextual bandit implementation using Keras [Python]

This repository contains an implementation of an online contextual bandit. Given a set of samples with multiple features, the agent will try to find the best corresponding label.

Additional functions are implemented to get the feature importance weights, given the current sample set.

## Dependencies
used packages: <br>
Keras==2.1.6 <br>
numpy==1.14.2 <br>
pandas==0.22.0 <br>
tqdm==4.19.5

## Example main file:
```python
labels = [3, 1, 0, 2, 2, 0]
features = [[0, 0, 1], [1, 0, 1], [2, 0, 1], [3, 0, 1], [3, 0, 1], [2, 0, 1]]

p = EpsilonGreedyPolicy(epsilon=0.1)
env = Environment(features, labels, p)
env.experiment(total_rounds=10000)
```

## Contact
You can contact me at bram.steenwinckel@ugent.be for any questions, proposals or if you wish to contribute.
