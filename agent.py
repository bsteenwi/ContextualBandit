from keras.models import Sequential, optimizers
from keras.layers.core import Dense


class KerasAgent(object):
    def __init__(self, lr, a_size, n_states):
        # These lines established the feed-forward part of the network.
        # The agent takes a state and produces an action.
        self.model = Sequential()
        self.model.add(Dense(10, input_shape=(n_states,), activation="relu"))

        self.model.add(Dense(a_size, activation="softmax"))
        sgd = optimizers.Adam(lr)
        self.model.compile(loss='categorical_crossentropy', optimizer=sgd)

    def getQValuesOfState(self, state):
        return self.model.predict(state, batch_size=1)
