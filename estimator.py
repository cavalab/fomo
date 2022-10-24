"""
This is a module to be used as a reference for building other modules
"""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import make_scorer
from sklearn.linear_model import SGDClassifier
# pymoo
from pymoo.core.algorithm import Algorithm
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from .problem import FomoProblem

class FomoEstimator(BaseEstimator):
    """ The base estimator for training fair models.  
    This class should not be called directly. Use `FomoRegressor` or
    `FomoClassifier` instead. 

    Parameters
    ----------
    estimator : sklearn-like estimator, default=None
        The underlying ML model to be trained. 
        The ML model must accept `sample_weight` as an argument to :meth:`fit`. 
    fairness_metrics : list, default=None
        The fairness metrics to try to optimize during fitting. 
    accuracy_metrics : list, default=None
        The accuracy metrics to try to optimize during fitting. 
    algorithm: pymoo Algorithm, default=None
        The multi-objective optimizer to use. Should be compatible with 
        `pymoo.core.algorithm.Algorithm`. 

    Examples
    --------
    >>> from fomo import FomoEstimator
    >>> import numpy as np
    >>> X = np.arange(100).reshape(100, 1)
    >>> y = np.zeros((100, ))
    >>> estimator = FomoEstimator()
    >>> estimator.fit(X, y)
    FomoEstimator()
    """
    def __init__(self, 
                 estimator,
                 fairness_metrics,
                 accuracy_metrics,
                 algorithm,
                 random_state
                ):
         self.estimator=estimator
         self.fairness_metrics=fairness_metrics
         self.algorithm=algorithm
         self.random_state=random_state

    def fit(self, X, y, groups=None):
        """Train the model.

        1. Train a population of self.estimator models with random weights. 
        2. Update sample weights using self.algorithm. 
        3. Select a given model as best, but also save the set of models. 

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples,) or (n_samples, n_outputs)
            The target values (class labels in classification, real numbers in
            regression).

        Returns
        -------
        self : object
            Returns self.
        """
        X, y = check_X_y(X, y, accept_sparse=True)

        # define problem
        self.problem_ = FomoProblem(fomo_estimator=self)

        # define algorithm
        
        ########################################
        # minimize
        self.res_ = minimize(self.problem_,
                             self.algorithm,
                             seed=self.random_state
                            )
        
        self.is_fitted_ = True
        # `fit` should always return `self`
        return self

    def predict(self, X):
        """ A reference implementation of a predicting function.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.

        Returns
        -------
        y : ndarray, shape (n_samples,)
            Returns an array of ones.
        """
        X = check_array(X, accept_sparse=True)
        check_is_fitted(self, 'is_fitted_')
        return np.ones(X.shape[0], dtype=np.int64)

class FomoClassifier(FomoEstimator, ClassifierMixin, BaseEstimator):
    """ An example classifier which implements a 1-NN algorithm.

    For more information regarding how to build your own classifier, read more
    in the :ref:`User Guide <user_guide>`.

    Parameters
    ----------
    demo_param : str, default='demo'
        A parameter used for demonstation of how to pass and store paramters.

    Attributes
    ----------
    X_ : ndarray, shape (n_samples, n_features)
        The input passed during :meth:`fit`.
    y_ : ndarray, shape (n_samples,)
        The labels passed during :meth:`fit`.
    classes_ : ndarray, shape (n_classes,)
        The classes seen at :meth:`fit`.
    """
    def __init__(self, 
                 estimator=SGDClassifier(),
                 fairness_metrics=None,
                 accuracy_metrics=None,
                 algorithm=NSGA2(),
                 random_state=None
                ):
        super().__init__(
            estimator, 
            fairness_metrics, 
            accuracy_metrics,
            algorithm, 
            random_state
        )

    def fit(self, X, y, groups=None):
        """A reference implementation of a fitting function for a classifier.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples,)
            The target values. An array of int.

        Returns
        -------
        self : object
            Returns self.
        """
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        # Store the classes seen during fit
        self.classes_ = unique_labels(y)

        self.X_ = X
        self.y_ = y
        # Return the classifier
        return self

    def predict(self, X):
        """ A reference implementation of a prediction for a classifier.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        y : ndarray, shape (n_samples,)
            The label for each sample is the label of the closest sample
            seen during fit.
        """
        # Check is fit had been called
        check_is_fitted(self, ['X_', 'y_'])

        # Input validation
        X = check_array(X)

        y_pred = None
        return y_pred

    def _init_metrics(self):
        """ Check metric definitions and/or define when necessary. """
        if len(self.accuracy_metrics) == 0:
            self.metrics_.append(make_scorer(self.estimator.score))


class FomoRegressor(RegressorMixin, BaseEstimator):
    """ An example transformer that returns the element-wise square root.

    For more information regarding how to build your own transformer, read more
    in the :ref:`User Guide <user_guide>`.

    Parameters
    ----------
    demo_param : str, default='demo'
        A parameter used for demonstation of how to pass and store paramters.

    Attributes
    ----------
    n_features_ : int
        The number of features of the data passed to :meth:`fit`.
    """
    def __init__(self, demo_param='demo'):
        self.demo_param = demo_param

    def fit(self, X, y, groups=None):
        """A reference implementation of a fitting function for a classifier.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples,)
            The target values. An array of int.

        Returns
        -------
        self : object
            Returns self.
        """
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        # Store the classes seen during fit
        self.classes_ = unique_labels(y)

        self.X_ = X
        self.y_ = y
        # Return the classifier
        return self

    def predict(self, X):
        """ A reference implementation of a prediction for a classifier.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        y : ndarray, shape (n_samples,)
            The label for each sample is the label of the closest sample
            seen during fit.
        """
        # Check is fit had been called
        check_is_fitted(self, ['X_', 'y_'])

        # Input validation
        X = check_array(X)

        return self.y_
