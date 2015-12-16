import itertools

import numpy as np
from collections import namedtuple


Model = namedtuple('Model', ['X', 'ys', 'params', 'gradients', 'loss'])
State = namedtuple('State', ['loss', 'dWh', 'dbh', 'dWs', 'dbs'])


def sigmoid(x):
    """Sigmoid function"""
    
    return 1 / (1 + np.exp(-x))

def sigmoid_grad(f):
    """Sigmoid gradient function
    
    Compute the gradient for the sigmoid function
    
    - f is the sigmoid function value of your original input x
    
    """
    return f * (1-f)

def sigmoid_inverse(z):
    """Computes the inverse of sigmoid

    z is the value that sigmoid produced

    """
    return -np.log(1/z - 1)

def random_Ws(layer_sizes):
    """Initialize list of random weight matrices

    layer_sizes : int
    the size of each layer in the network

    """
    num_layers = len(layer_sizes)

    for i in range(num_layers-1):
        n, m = layer_sizes[i:i+2]
        epsilon = np.sqrt(6) / np.sqrt(m+n)

        yield np.random.uniform(low=-epsilon, high=epsilon, size=(m, n))

def random_bs(layer_sizes):
    """Initialize list of random bias vectors

    layer_sizes : int
    the size of each layer in the network

    """
    num_layers = len(layer_sizes)
    for i in range(1, num_layers):
        n = layer_sizes[i]
        epsilon = np.sqrt(6) / np.sqrt(n+1)

        yield np.random.uniform(low=-epsilon, high=epsilon, size=(n, 1))

def get_minibatch_indices(max_index, start, num_desired):
    """Produces indices for a minibatch

    Wraps around the end to the beginning

    """
    index = itertools.cycle(range(max_index))
    for _ in range(start):
        next(index)

    return [next(index) for _ in range(num_desired)]

def minibatch_generator(X, ys, batch_size=None, random=False):
    """Yields minibatch after minibatch
    
    For the gradient checking minibatch to be the same as the minibatch used to
    compute analytic gradients, we explicitly tell the batch index to update or
    not. Gradient checking will always tell the batch index *not* to update,
    whereas learn() will always have it update.
    
    """
    yield # Dummy yield
    
    N, M = X.shape
    batch_size = M if not batch_size else batch_size
    batch_index = 0

    while True:
        low, high = batch_index, batch_index+batch_size
        indexes = get_minibatch_indices(M, batch_index, batch_size)
        X_mini, ys_mini = X[:, indexes], ys[indexes]

        freeze_batch_index = (yield X_mini, ys_mini)

        # Do *not* update the batch index if we were called during a gradient
        # check!
        if not freeze_batch_index:
            if random:
                batch_index = np.random.randint(0, M)
            else:
                batch_index = (batch_index+batch_size) % M
