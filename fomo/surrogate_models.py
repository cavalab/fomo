"""
Fairness Oriented Multiobjective Optimization (Fomo)

BSD 3-Clause License

Copyright (c) 2023, William La Cava

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""
from sklearn.neural_network import MLPClassifier
from sklearn.utils import check_random_state
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
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
        return self.predict_proba(self._one_hot_encode(X))[:,1]

    def _one_hot_encode(self, X):
        if hasattr(self, 'ohc'):
            return self.ohc.transform(X)
        else:
            categorical_features = [c for c in X.columns if X[c].nunique() < 8]
            self.ohc = ColumnTransformer(
                [
                    (
                        "cat",
                        OneHotEncoder(
                            handle_unknown="ignore", 
                            sparse_output=False
                        ),
                        categorical_features,
                    ),
                ],
                verbose_feature_names_out=False,
                remainder='passthrough'
            )
            return self.ohc.fit_transform(X)

    def init(self, X, incremental=False):
        """Overload of MLP training. Just determines the dimensions of 
        the coefficients in the network. 
        Adapted from _multilayer_perceptron.py::_fit()

        """
        # one-hot encode X
        X = self._one_hot_encode(X)
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
        X = self._one_hot_encode(Xp)
        self.coefs_ = np.empty(X.shape[1] + 1) 

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
        X = self._one_hot_encode(X)
        intercept = np.ones(X.shape[0])
        Xintercept = np.column_stack((intercept, X))
        # return expit(np.dot(Xintercept,self.coefs_))
        return np.dot(Xintercept,self.coefs_)

    def _one_hot_encode(self, X):
        if hasattr(self, 'ohc'):
            return self.ohc.transform(X)
        else:
            binary_columns = [col for col in X.columns if X[col].isin([0, 1]).all()]
            categorical_features = [c for c in X.columns if (X[c].nunique() <= 8 and c not in binary_columns)] #Do not one-hot-encode binary columns and columns with more than 8 categories. 
            self.ohc = ColumnTransformer(
                [
                    (
                        "cat",
                        OneHotEncoder(
                            handle_unknown="ignore", 
                            sparse_output=False
                        ),
                        categorical_features,
                    ),
                ],
                verbose_feature_names_out=False,
                remainder='passthrough'
            )
            return self.ohc.fit_transform(X)

class InterLinear:
    """Make an interaction version of a linear model
    change one hot encode to capture all unique values of Xp
    """

    def __init__(self, Xp):
        # Xohc = pd.get_dummies(Xp.astype('category'))
        # print('Xohc shape:',Xohc.shape)
        # Xpu = Xp.unique()
        X = self._one_hot_encode(Xp)
        self.coefs_ = np.empty(X.shape[1] + 1) 

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
        X = self._one_hot_encode(X)
        intercept = np.ones(X.shape[0])
        Xintercept = np.column_stack((intercept, X))
        # return expit(np.dot(Xintercept,self.coefs_))
        return np.dot(Xintercept,self.coefs_)

    def _make_interaction(self, X):
        """add an interaction variable"""
        X = X.copy()
        groups = list(X.columns)
        categories = X.groupby(groups).groups
        categories = {k:i for i,k in enumerate(categories.keys())}
        def categorize(x):
            return categories[tuple(x[c] for c in x.index)]
        X['interaction'] = X.apply(lambda x: categorize(x), axis=1)
        return X

    def _one_hot_encode(self, X):

        X = self._make_interaction(X)

        if not hasattr(self, 'ohc'):
            self.ohc = ColumnTransformer(
                [
                    (
                        "cat",
                        OneHotEncoder(
                            handle_unknown="ignore", 
                            sparse_output=False
                        ),
                        X.columns,
                    ),
                ],
                verbose_feature_names_out=False,
                remainder='passthrough'
            )
            self.ohc.fit(X)

        return self.ohc.transform(X)