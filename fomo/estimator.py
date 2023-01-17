"""
This is a module to be used as a reference for building other modules
"""
import pdb
import numpy as np
import copy
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import make_scorer, roc_auc_score, r2_score, mean_squared_error
from sklearn.linear_model import SGDClassifier, SGDRegressor
from sklearn.base import clone
import multiprocessing

# pymoo
from pymoo.core.algorithm import Algorithm
from pymoo.core.problem import StarmapParallelization
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from .problem import FomoProblem
import fomo.metrics as metrics
# from pymoo.decomposition.asf import ASF
from pymoo.mcdm.high_tradeoff import HighTradeoffPoints
from pymoo.visualization.scatter import Scatter
# plotting
import matplotlib.pyplot as plt


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
    random_state: int | None
        Random seed. 

    """
    def __init__(self, 
                 estimator: BaseEstimator,
                 fairness_metrics: list[str],
                 accuracy_metrics: list[str],
                 algorithm: Algorithm,
                 random_state: int,
                 verbose:bool,
                 n_jobs:int,
                 batch_size:int
                ):
         self.estimator=estimator
         self.fairness_metrics=fairness_metrics
         self.accuracy_metrics=accuracy_metrics
         self.algorithm=algorithm
         self.random_state=random_state
         self.verbose=verbose
         self.n_jobs=n_jobs
         self.batch_size=batch_size

    def fit(self, X, y, protected_features=None, Xp=None, **kwargs):
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
        protected_columns: list[str] | None
            The columns in DataFrame X used to assign fairness. 
        Xp : {array-like, sparse matrix}, shape (n_samples, n_protected_features)
            The input samples for measuring/optimizing fairness during training.
        Returns
        -------
        self : object
            Returns self.
        """
        self._init_model()
        # X, y = check_X_y(X, y, accept_sparse=True)
        self.n_obj_ = len(self.accuracy_metrics_)+len(self.fairness_metrics_)
        # define problem
        # initialize the thread pool and create the runner
        n_processes = self.n_jobs if self.n_jobs > 0 else multiprocessing.cpu_count()
        pool = multiprocessing.Pool(n_processes)
        runner = StarmapParallelization(pool.starmap)

        self.problem_ = FomoProblem(fomo_estimator=self, elementwise_runner=runner)

        # metric arguments
        self.metric_kwargs = dict(
            groups=protected_features, 
            X_protected=Xp
        )

        ########################################
        # minimize
        self.res_ = minimize(self.problem_,
                             self.algorithm,
                             seed=self.random_state,
                             verbose=self.verbose,
                             **kwargs
                            )
        pool.close()
        # choose "best" estimator
        self.best_estimator_ = self._pick_best() 
        self.is_fitted_ = True
        # `fit` should always return `self`
        return self

    def _pick_best(self):
        """Picks the best solution based on high tradeoff point. """
        # weights = np.array([0.5 for n in range(self.n_obj_)])
        # decomp = ASF()
        # I = decomp(F, weights).argmin()
        F = self.res_.F
        if len(F) == 1:
            I = 0
        else:
            dm = HighTradeoffPoints()
            I = dm(F)[0]
        print("Best regarding decomposition: Point %s - %s" % (I, F[I]))
        self.best_weights_ = self.res_.X[I]
        print(f'best_weights: {self.best_weights_}')
        self.I_ = I
        best_est = clone(self.estimator)
        best_est.fit(self.X_, self.y_, sample_weight=self.best_weights_)
        return best_est

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
        return self.best_estimator_.predict(X)

    def plot(self):
        check_is_fitted(self, 'is_fitted_')
        F = copy.copy(self.res_.F)
        I = self.I_
        axis_labels = (
            [ am._score_func.__name__ for am in self.accuracy_metrics_ ] 
            + [ fn.__name__ for fn in self.fairness_metrics_ ]
        )
        axis_labels = [al.replace('_',' ') for al in axis_labels]
        # reverse F for metrics where higher is better
        for i,m in enumerate(self.accuracy_metrics_ + self.fairness_metrics_): 
            if hasattr(m, '_sign'):
                F[:,i] = F[:,i]*m._sign
        plot = (
            Scatter()
            .add(F, alpha=0.2)
            .add(F[I], color="red", s=100)
        )
        plot.axis_labels = axis_labels
        return plot

    def _init_model(self):
        if hasattr(self.estimator, 'random_state'):
            self.estimator.random_state = self.random_state
        if hasattr(self.estimator, 'n_jobs'):
            self.estimator.n_jobs = 1

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
                 estimator: ClassifierMixin=SGDClassifier(),
                 fairness_metrics=None,
                 accuracy_metrics=None,
                 algorithm: Algorithm = NSGA2(),
                 random_state: int=None,
                 verbose: bool = False,
                 n_jobs: int = -1,
                 batch_size: int = 0
                ):
        super().__init__(
            estimator, 
            fairness_metrics, 
            accuracy_metrics,
            algorithm, 
            random_state,
            verbose,
            n_jobs,
            batch_size
        )

    def fit(self, X, y, protected_features=None, Xp=None, **kwargs):
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
        # X, y = check_X_y(X, y)
        # Store the classes seen during fit
        self.classes_ = unique_labels(y)

        self.X_ = X
        self.y_ = y

        self._init_metrics()

        super().fit(X, y, protected_features=protected_features, Xp=Xp, **kwargs)

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
        # X = check_array(X)

        return super().predict(X)

    def predict_proba(self, X):
        """ Return prediction probabilities."""
        return self.best_estimator_.predict_proba(X)

    def _init_metrics(self):
        """ Check metric definitions and/or define when necessary. """
        self.accuracy_metrics_ = self.accuracy_metrics
        self.fairness_metrics_ = self.fairness_metrics
        if self.accuracy_metrics is None:
            self.accuracy_metrics_ = [make_scorer(roc_auc_score, greater_is_better=False)]
        if self.fairness_metrics is None:
            self.fairness_metrics_ = [metrics.multicalibration_loss]


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
                 estimator: RegressorMixin=SGDRegressor(),
                 fairness_metrics: list[str]=None,
                 accuracy_metrics: list[str]=None,
                 algorithm: str ='NSGA2',
                 random_state: int=None,
                 verbose: bool = False,
                 n_jobs: int = -1,
                 batch_size: int = 0
                ):
        super().__init__(
            estimator, 
            fairness_metrics, 
            accuracy_metrics,
            algorithm, 
            random_state,
            verbose,
            n_jobs,
            batch_size
        )

    def fit(self, X, y, protected_features=None, Xp=None, **kwargs):
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
        super().fit(X, y, protected_features=protected_features, Xp=Xp, **kwargs)

        # Return the regressor
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

        return super().predict(X)

    def _init_metrics(self):
        """ Check metric definitions and/or define when necessary. """
        self.accuracy_metrics_ = []
        self.fairness_metrics_ = []
        if self.accuracy_metrics is None:
            # self.accuracy_metrics_.append(make_scorer(r2_score, greater_is_better=False))
            self.accuracy_metrics_.append(make_scorer(mean_squared_error))
        if len(self.fairness_metrics) == 0:
            self.fairness_metrics_.append(metrics.subgroup_MSE)
