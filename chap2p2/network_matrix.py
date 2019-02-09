"""
network_matrix.py
~~~~~~~~~~

A module to implement the stochastic gradient descent learning
algorithm for a feedforward neural network.  Gradients are calculated
using backpropagation.  Note that I have focused on making the code
simple, easily readable, and easily modifiable.  It is not optimized,
and omits many desirable features.
"""

#### Libraries
# Standard library
import random
import time

# Third-party libraries
import numpy as np

class Network(object):

    def __init__(self, sizes):
        """The list ``sizes`` contains the number of neurons in the
        respective layers of the network.  For example, if the list
        was [2, 3, 1] then it would be a three-layer network, with the
        first layer containing 2 neurons, the second layer 3 neurons,
        and the third layer 1 neuron.  The biases and weights for the
        network are initialized randomly, using a Gaussian
        distribution with mean 0, and variance 1.  Note that the first
        layer is assumed to be an input layer, and by convention we
        won't set any biases for those neurons, since biases are only
        ever used in computing the outputs from later layers."""
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        """Return the output of the network if ``a`` is input."""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a    

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None):
        """Train the neural network using mini-batch stochastic
        gradient descent.  The ``training_data`` is a list of tuples
        ``(x, y)`` representing the training inputs and the desired
        outputs.  The other non-optional parameters are
        self-explanatory.  If ``test_data`` is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially."""
        if test_data: n_test = len(test_data)
        self.mini_batch_size = mini_batch_size  # to be used in other functions
        n = len(training_data)
        start = time.time()  # begin timer
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            mini_batches_X, mini_batches_Y = [], []
            for batch in mini_batches:
                mini_batches_X.append(np.column_stack(tuple([batch[k][0]
                    for k in range(mini_batch_size)])))
                mini_batches_Y.append(np.column_stack(tuple([batch[k][1]
                    for k in range(mini_batch_size)])))
            for X, Y in zip(mini_batches_X, mini_batches_Y):
                self.update_mini_batch(X, Y, eta)
            if test_data:
                print("Epoch {0}: {1} / {2}, elapsed time: {3:.2f}s".format(
                    j, self.evaluate(test_data), n_test, time.time()-start))
            else:
                print("Epoch {0} complete, elapsed time: {1:.2f}s".format(
                    j, time.time()-start))

    def update_mini_batch(self, X, Y, eta):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        ``eta`` is the learning rate."""
        nabla_b, nabla_w = self.backprop(X, Y)
        sum_nabla_b = [np.sum(nb, axis=1).reshape((nb.shape[0],1)) for nb in nabla_b]
        self.weights = [w-(eta/self.mini_batch_size)*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/self.mini_batch_size)*nb
                       for b, nb in zip(self.biases, sum_nabla_b)]


    def backprop(self, X, Y):
        """Return a tuple ``(nabla_B, nabla_W)`` representing the
        gradient for the cost function C_x.  ``nabla_B`` and
        ``nabla_W`` are layer-by-layer lists of numpy arrays of dimension 2,
        similar to ``self.biases`` and ``self.weights`` but nabla_B's columns
        are repeated over the training examples."""
        nabla_B = [np.tile(np.zeros(b.shape), (1, self.mini_batch_size))
                for b in self.biases]
        nabla_W = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = X
        activations = [X] # list to store all the activations, layer by layer
        zs = [] # list to store all the z matrices, layer by layer
        for b, w in zip(self.biases, self.weights):
            B = np.tile(b, (1,self.mini_batch_size))  # repeat column b for each training example
            Z = np.dot(w, activation)+B
            zs.append(Z)
            activation = sigmoid(Z)
            activations.append(activation)
        # backward pass
        delta = self.cost_derivative(activations[-1], Y) * \
            sigmoid_prime(zs[-1])
        nabla_B[-1] = delta
        nabla_W[-1] = np.dot(delta, activations[-2].transpose())
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in range(2, self.num_layers):
            Z = zs[-l]
            sp = sigmoid_prime(Z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_B[-l] = delta
            nabla_W[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_B, nabla_W)

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        """Return the matrix of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return (output_activations-y)

#### Miscellaneous functions
def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))
