from sklearn.neural_network import MLPClassifier
from sklearn.utils import check_random_state
# from itertools import chain
import numpy as np
import pandas as pd
from scipy.special import expit

class MLP(MLPClassifier):

    def get_n_weights(self):
        """Sets coefs_ and intercepts_ to x."""
        return np.sum([c.size for c in self.coefs_ + self.intercepts_])

    def set_weights(self, x):
        """Sets coefs_ and intercepts_ to x."""
        self._unpack(x)

    def fit(self, X, y):
        raise NotImplementedError('Dont call fit on this class!')
        return self
    
    def predict(self, X):
        # one-hot encode X
        # X = pd.get_dummies(X.astype('category'))
        return self.predict_proba(X)[:,1]

    def init(self, X, incremental=False):
        """Overload of MLP training. Just determines the dimensions of 
        the coefficients in the network. 
        Adapted from _multilayer_perceptron.py::_fit()

        """
        # one-hot encode X
        # X = pd.get_dummies(X.astype('category'))
        # print('Xohc shape:',X.shape)
        # import pdb
        # pdb.set_trace()
        # make a random y
        y = np.random.randint(0,1,size=len(X))
        # Make sure self.hidden_layer_sizes is a list
        hidden_layer_sizes = self.hidden_layer_sizes
        if not hasattr(hidden_layer_sizes, "__iter__"):
            hidden_layer_sizes = [hidden_layer_sizes]
        hidden_layer_sizes = list(hidden_layer_sizes)

        if np.any(np.array(hidden_layer_sizes) <= 0):
            raise ValueError(
                "hidden_layer_sizes must be > 0, got %s." % hidden_layer_sizes
            )
        first_pass = not hasattr(self, "coefs_") or (
            not self.warm_start and not incremental
        )

        X, y = self._validate_input(X, y, incremental, reset=first_pass)

        n_samples, n_features = X.shape

        # Ensure y is 2D
        if y.ndim == 1:
            y = y.reshape((-1, 1))

        self.n_outputs_ = y.shape[1]

        layer_units = [n_features] + hidden_layer_sizes + [self.n_outputs_]

        # check random state
        self._random_state = check_random_state(self.random_state)

        if first_pass:
            # First time training the model
            self._initialize(y, layer_units, X.dtype)

        # Initialize lists
        activations = [X] + [None] * (len(layer_units) - 1)
        deltas = [None] * (len(activations) - 1)

        coef_grads = [
            np.empty((n_fan_in_, n_fan_out_), dtype=X.dtype)
            for n_fan_in_, n_fan_out_ in zip(layer_units[:-1], layer_units[1:])
        ]

        intercept_grads = [
            np.empty(n_fan_out_, dtype=X.dtype) for n_fan_out_ in layer_units[1:]
        ]

        # from _fit_lbfgs:
        # Store meta information for the parameters
        self._coef_indptr = []
        self._intercept_indptr = []
        start = 0

        # Save sizes and indices of coefficients for faster unpacking
        for i in range(self.n_layers_ - 1):
            n_fan_in, n_fan_out = layer_units[i], layer_units[i + 1]

            end = start + (n_fan_in * n_fan_out)
            self._coef_indptr.append((start, end, (n_fan_in, n_fan_out)))
            start = end

        # Save sizes and indices of intercepts for faster unpacking
        for i in range(self.n_layers_ - 1):
            end = start + layer_units[i + 1]
            self._intercept_indptr.append((start, end))
            start = end

        return self

class Linear:

    def __init__(self, Xp):
        # Xohc = pd.get_dummies(Xp.astype('category'))
        # print('Xohc shape:',Xohc.shape)
        self.coefs_ = np.empty(Xp.shape[1] + 1) 

    def get_n_weights(self):
        """Sets coefs_ and intercepts_ to x."""
        return len(self.coefs_)

    def set_weights(self, x):
        """Sets coefs_ and intercepts_ to x."""
        self.coefs_ = x

    def fit(self, X, y):
        raise NotImplementedError('Dont call fit on this class!')
        return self

    def predict(self, X):
        # Xohc = pd.get_dummies(X.astype('category'))
        intercept = np.ones(X.shape[0])
        Xintercept = np.column_stack((intercept, X))
        return expit(np.dot(Xintercept,self.coefs_))