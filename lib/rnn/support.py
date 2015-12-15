import itertools

from collections import namedtuple

State = namedtuple('State', ['loss', 'grads', 'hidden'])
Gradients = namedtuple('Gradients', ['dWhh', 'dbhh', 'dWxh', 'dbxh', 'dWs', 'dbs'])

Snapshot = namedtuple('State', ['xs', 'ys', 'hiddens', 'Whh', 'bhh', 'Wxh', 'bxh', 'Ws', 'bs', 'dWhh', 'dbhh', 'dWxh', 'dbxh', 'dWs', 'dbs', 'dhiddens', 'dhiddens_local', 'dhiddens_downstream', 'scores', 'loss'])

def input_generator(X, ys):
    """Keep yielding the next training example

    Loop around when you get to the end

    """
    N, T = X.shape

    while True:
        for t in range(T):
            yield X[:, [t]], ys[t]
