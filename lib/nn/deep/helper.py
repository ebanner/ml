import numpy as np
from collections import namedtuple

# Model = namedtuple('Model',['x', 'y', 'wh', 'bh', 'z', 'w1', 'b1', 'score1', 'w2', 'b2', 'score2', 'prob1', 'prob2', 'dscore1', 'dscore2', 'db1', 'dw1', 'db2', 'dw2', 'loss'])
# State = namedtuple('State', ['loss', 'dwh', 'dbh', 'dws', 'dbs'])

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

def minibatch_generator(X, ys, batch_size=None):
    """Yields minibatch after minibatch"""
    
    N, M = X.shape
    batch_size = M if not batch_size else batch_size
    batch_index = 0

    while True:
        low, high = batch_index*batch_size, (batch_index+1)*batch_size
        X = X[:, low:high].reshape(N, batch_size)
        ys = ys[low:high]

        yield X, ys

        batch_index = (batch_index+1) % (M//batch_size)
