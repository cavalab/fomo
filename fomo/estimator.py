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
import copy
import math
import uuid 
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import make_scorer, roc_auc_score, r2_score, mean_squared_error
from sklearn.linear_model import LogisticRegression, SGDRegressor
from sklearn.base import clone
from sklearn.pipeline import Pipeline
import multiprocessing
from multiprocessing.pool import ThreadPool
import dill

# pymoo
from pymoo.core.algorithm import Algorithm
from pymoo.core.problem import StarmapParallelization, ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
# MCDM imports
from pymoo.mcdm.high_tradeoff import HighTradeoffPoints
from pymoo.mcdm.pseudo_weights import PseudoWeights
# fomo
from fomo.utils import Compromise
import fomo.metrics as metrics
from .problem import BasicProblem
# plotting
from pymoo.visualization.scatter import Scatter
# types
from collections.abc import Callable


class FomoEstimator(BaseEstimator):
    """ The base estimator for training fair models.  
    This class should not be called directly. Use :class:`FomoRegressor` or
    :class:`FomoClassifier` instead. 

    Parameters
    ----------
    estimator : sklearn-like estimator
        The underlying ML model to be trained. 
        The ML model must accept `sample_weight` as an argument to :meth:`fit`. 
    fairness_metrics : list[Callable]
        The fairness metrics to try to optimize during fitting. 
    accuracy_metrics : list[Callable]
        The accuracy metrics to try to optimize during fitting. 
    algorithm: pymoo Algorithm
        The multi-objective optimizer to use. Should be compatible with 
        `pymoo.core.algorithm.Algorithm`. 
    random_state: int | None
        Random seed. 
    verbose: bool
        Whether to print progress.
    n_jobs: int
        Number of parallel processes to use. Parallelizes evaluation.
    store_final_models: bool
        If True, the final set of models will be stored in the estimator.
    problem_type: ElementwiseProblem
        Determines the evaluation class to be used. Options:
        - :class:`BasicProblem`
        - :class:`MLPProblem`
        - :class:`LinearProblem`
    """
    def __init__(self, 
                 estimator: BaseEstimator,
                 fairness_metrics: list[Callable],
                 accuracy_metrics: list[Callable],
                 algorithm: Algorithm,
                 random_state: int,
                 verbose:bool,
                 n_jobs:int,
                 store_final_models:bool,
                 problem_type:ElementwiseProblem,
                 checkpoint:bool,
                 picking_strategy: str = 'PseudoWeights'
                ):
         self.estimator=estimator
         self.fairness_metrics=fairness_metrics
         self.accuracy_metrics=accuracy_metrics
         self.algorithm=algorithm
         self.random_state=random_state
         self.verbose=verbose
         self.n_jobs=n_jobs
         self.store_final_models=store_final_models
         self.problem_type=problem_type
         self.checkpoint=checkpoint
         self.picking_strategy=picking_strategy

    def fit(self, X, y, grouping = 'intersectional', abs_val = False, gamma = True, protected_features=None, Xp=None, starting_point=None, **kwargs):
        """Train the model.


        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples,) or (n_samples, n_outputs)
            The target values (class labels in classification, real numbers in
            regression).
        protected_columns: list[str] | None
            The columns in DataFrame X used to assign fairness. 
        Xp : {array-like, sparse matrix}, shape (n_samples, n_protected_features)
            The input samples for measuring/optimizing fairness during training.
        starting_point : str | None
            Optionally start from a checkpoint file with this name
        **kwargs : keyword arguments that are passed to `pymoo.optimize.minimize`.

        Returns
        -------
        self : object
            Returns self.
        """
        self._init_model()
        self.n_obj_ = len(self.accuracy_metrics_)+len(self.fairness_metrics_)
        ########################################
        # define problem
        # metric arguments
        metric_kwargs = dict(
            groups=protected_features, 
            X_protected=Xp,
            grouping = grouping, 
            abs_val = abs_val,
            gamma = gamma
        )
        problem_kwargs=dict(fomo_estimator=self, metric_kwargs=metric_kwargs)
        # parallelization
        n_processes = self.n_jobs if self.n_jobs > 0 else multiprocessing.cpu_count()
        print('running',n_processes,'processes')
        if n_processes > 1:
            pool = multiprocessing.Pool(n_processes)
            runner = StarmapParallelization(pool.starmap)
            problem_kwargs.update(dict(elementwise_runner=runner))

        self.problem_ = self.problem_type(**problem_kwargs)
        print('number of variables:',self.problem_.n_var)
        print('number of objectives:',self.problem_.n_obj)

        # define algorithm
        if starting_point is not None:
            with open(starting_point, 'rb') as f:
                self.algorithm_ = dill.load(f)
                print("Loaded Checkpoint:", self.algorithm_)
        else:
            self.algorithm_ = self.algorithm
        ########################################
        # minimize
        if self.checkpoint:
            run_id = uuid.uuid4()
            checkpoint_file = f"checkpoint.{run_id}.pkl"
            print('checkpoint file:',checkpoint_file)
            self.algorithm_.setup(
                self.problem_,
                seed=self.random_state,
                verbose=self.verbose,
                **kwargs
            )
            while self.algorithm_.has_next():
                self.algorithm_.next()
                with open(checkpoint_file, "wb") as f:
                    dill.dump(self.algorithm_, f)
            self.res_ = self.algorithm_.result()
        else:
            self.res_ = minimize(
                self.problem_,
                self.algorithm_,
                seed=self.random_state,
                verbose=self.verbose,
                save_history=True,
                **kwargs
            )

        if n_processes > 1:
            pool.close()
        ########################################
        # choose "best" estimator
        self.best_estimator_ = self.pick_best(strategy=self.picking_strategy) 
        self.is_fitted_ = True
        # store archive of estimators
        if self.store_final_models:
            self.estimator_archive_ = self._store_final_models()
        return self

    def _store_final_models(self):
        """Store archive of fitted estimators using final weights."""
        estimator_archive_ = []
        for x in self.res_.X:
            sample_weight = self.problem_.get_sample_weight(x)
            est = clone(self.estimator)
            if isinstance(est, Pipeline):
                stepname = est.steps[-1][0]
                param_name = stepname + '__sample_weight'
                kwarg = {param_name:sample_weight}
                est.fit(self.X_, self.y_, **kwarg)
            else:
                est.fit(self.X_, self.y_, sample_weight=sample_weight)
            estimator_archive_.append(est)
        return estimator_archive_

    def _output_archive(self, fn, **kwargs):
        """Call a function on every estimator in the archive and return output."""
        return [getattr(est, fn)(**kwargs) for est in self.estimator_archive_]

    def predict_archive(self, X):
        """Return a list of predictions from the archive models. """
        check_is_fitted(self, 'is_fitted_')
        if not hasattr(self, 'estimator_archive_'):
            print("Need to fit archive models. You can set `store_final_models` to True to avoid this step.")
            self.estimator_archive_ = self._store_final_models()

        return self._output_archive('predict', X=X)

        
    def pick_best(self, strategy='PseudoWeights', weights=None):
        """Picks the best solution based on on a multi-criteria decision-making
        (MCDM) strategy. 

        A description of MCDM strategies is given in the `pymoo docs <https://pymoo.org/mcdm/index.html>`_. 

        Parameters
        ----------
        strategy : str
            Name of an MCDM strategy. Built-in support for the following:

            - 'HighTradeOffPoints' : return a point near a cutoff.
            - 'Compromise': equally weight both objectives.  
            - 'PseudoWeights' : normalized weighting of both objectives. 

        weights: np.ndarray|None
            Weights for each objective. Used for Compromise and PseudoWeights methods.
            Default is equal weighting.
        """

        if type(weights) == type(None):
            if strategy in ['PseudoWeights','Compromise']:
                weights = np.array([float(1.0/self.n_obj_) for n in range(self.n_obj_)])

        if strategy == 'PseudoWeights':
            picking_fn = PseudoWeights(weights).do
        elif strategy == 'Compromise':
            picking_fn = Compromise(weights).do
        else:
            picking_fn = HighTradeoffPoints()

        F = self.res_.F.copy() 

        if len(F) <= 1:
            print('Warning: only one point on pareto front')
            I = 0
        else:
            I = picking_fn(F)
            if isinstance(I, np.ndarray):
                if len(I) > 1:
                    I = I[math.floor(len(I)/2)]
                else:
                    I = I[0]
            elif I is None:
                print('warning: picking returned None')
                I = np.random.randint(len(F))
        self.best_weights_ = self.res_.X[I]
        self.I_ = I
        best_est = clone(self.estimator)
        sample_weight = self.problem_.get_sample_weight(self.best_weights_)

        if isinstance(best_est, Pipeline):
            stepname = best_est.steps[-1][0]
            param_name = stepname + '__sample_weight'
            kwarg = {param_name:sample_weight}
            best_est.fit(self.X_, self.y_, **kwarg)
        else:
            best_est.fit(self.X_, self.y_, sample_weight=sample_weight)
        return best_est

    def predict(self, X):
        """Predict from X.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.

        Returns
        -------
        y : ndarray, shape (n_samples,)
        """
        # X = check_array(X, accept_sparse=True)
        check_is_fitted(self, 'is_fitted_')
        return self.best_estimator_.predict(X)

    def plot(self):
        """Plots the Pareto set of models with a dot indicating the selected 
        model.
        Returns
        -------
        plot : matplotlib figure
            The figure object.
        """
        check_is_fitted(self, 'is_fitted_')
        I = self.I_
        F = self._get_signed_F()
        axis_labels = (
            [ am._score_func.__name__ for am in self.accuracy_metrics_ ] 
            + [ fn.__name__ for fn in self.fairness_metrics_ ]
        )
        axis_labels = [al.replace('_',' ') for al in axis_labels]
        plot = (
            Scatter()
            .add(F, alpha=0.2, label='Candidate models')
            .add(F[I], color="red", s=100, label='Chosen model')
        )
        plot.axis_labels = axis_labels
        return plot

    def _init_model(self):
        if hasattr(self.estimator, 'random_state'):
            self.estimator.random_state = self.random_state
        if hasattr(self.estimator, 'n_jobs'):
            self.estimator.n_jobs = 1
    
    def _get_signed_F(self, F=None):
        if F is None:
            F = copy.copy(self.res_.F)
        # reverse F for metrics where higher is better
        for i,m in enumerate(self.accuracy_metrics_ + self.fairness_metrics_): 
            if hasattr(m, '_sign'):
                F[:,i] = F[:,i]*m._sign
        return F

class FomoClassifier(FomoEstimator, ClassifierMixin, BaseEstimator):
    """FOMO Classifier. 

        1. Train a population of self.estimator models with random weights. 
        2. Update sample weights using self.algorithm. 
        3. Select a given model as best, but also save the set of models. 

    Parameters
    ----------
    estimator : sklearn-like estimator
        The underlying ML model to be trained. 
        The ML model must accept `sample_weight` as an argument to :meth:`fit`. 
    fairness_metrics : list[Callable]
        The fairness metrics to try to optimize during fitting. 
    accuracy_metrics : list[Callable]
        The accuracy metrics to try to optimize during fitting. 
    algorithm: pymoo Algorithm
        The multi-objective optimizer to use. Should be compatible with 
        `pymoo.core.algorithm.Algorithm`. 
    random_state: int | None
        Random seed. 
    verbose: bool
        Whether to print progress.
    n_jobs: int
        Number of parallel processes to use. Parallelizes evaluation.
    store_final_models: bool
        If True, the final set of models will be stored in the estimator.
    problem_type: ElementwiseProblem
        Determines the evaluation class to be used. Options:
        - :class:`BasicProblem`
        - :class:`MLPProblem`
        - :class:`LinearProblem`

    Examples
    --------
    >>> from fomo import FomoClassifier
    >>> from pmlb import pmlb
    >>> X,y = pmlb.fetch_data('adult', return_X_y=True)
    >>> groups = ['race','sex']
    >>> est = FomoClassifier()
    >>> est.fit(X,y, protected_features=groups)
    """
    def __init__(self, 
                 estimator: ClassifierMixin=LogisticRegression(),
                 fairness_metrics=None,
                 accuracy_metrics=None,
                 algorithm: Algorithm = NSGA2(),
                 random_state: int=None,
                 verbose: bool = False,
                 n_jobs: int = -1,
                 store_final_models: bool = False,
                 problem_type = BasicProblem,
                 checkpoint = False,
                 picking_strategy='PseudoWeights'
                ):
        super().__init__(
            estimator, 
            fairness_metrics, 
            accuracy_metrics,
            algorithm, 
            random_state,
            verbose,
            n_jobs,
            store_final_models,
            problem_type,
            checkpoint,
            picking_strategy
        )

    def fit(self, X, y, grouping = 'intersectional', abs_val = False, gamma = True, protected_features=None, Xp=None, **kwargs):
        """Train the model.

        1. Train a population of self.estimator models with random weights. 
        2. Update sample weights using self.algorithm. 
        3. Select a given model as best, but also save the set of models. 

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples,)
            The target values. An array of int.
        protected_features: list[str]|None, default = None
            The columns of X to calculate fairness over. If specifying columns,
            do not also specify `Xp`.
        Xp: pandas DataFrame, shape (n_samples, n_protected_features), default=None
            The protected feature values used to calculate fairness. If `Xp` is 
            specified, `protected_features` must be None. 
        **kwargs : passed to `pymo.optimize.minimize`. 

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

        super().fit(X, y, grouping = grouping, abs_val = abs_val, gamma = gamma, protected_features=protected_features, Xp=Xp, **kwargs)

        # Return the classifier
        return self

    def predict(self, X):
        """ Predict labels from X.

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
            self.accuracy_metrics_ = [make_scorer(roc_auc_score, greater_is_better=False, needs_proba=True)]
        if self.fairness_metrics is None:
            self.fairness_metrics_ = [metrics.multicalibration_loss]

    def predict_proba_archive(self, X):
        """Return a list of predictions from the archive models. """
        check_is_fitted(self, 'is_fitted_')
        if not hasattr(self, 'estimator_archive_'):
            print("Need to fit archive models. You can set `store_final_models` to True to avoid this step.")
            self.estimator_archive_ = self._store_final_models()
        return self._output_archive('predict_proba', X=X)


class FomoRegressor(RegressorMixin, BaseEstimator):
    """Fomo class for regression models. 

    Parameters
    ----------
    estimator : sklearn-like estimator
        The underlying ML model to be trained. 
        The ML model must accept `sample_weight` as an argument to :meth:`fit`. 
    fairness_metrics : list[Callable]
        The fairness metrics to try to optimize during fitting. 
    accuracy_metrics : list[Callable]
        The accuracy metrics to try to optimize during fitting. 
    algorithm: pymoo Algorithm
        The multi-objective optimizer to use. Should be compatible with 
        `pymoo.core.algorithm.Algorithm`. 
    random_state: int | None
        Random seed. 
    verbose: bool
        Whether to print progress.
    n_jobs: int
        Number of parallel processes to use. Parallelizes evaluation.
    store_final_models: bool
        If True, the final set of models will be stored in the estimator.
    problem_type: ElementwiseProblem
        Determines the evaluation class to be used. Options:

        - :class:`BasicProblem`: loss function weights are directly optimized.
        - :class:`MLPProblem`: weights of a multilayer perceptron are optimized to estimate loss function weights. 
        - :class:`LinearProblem`: weights of a logistic model are optimized to estimate loss function weights.

    Attributes
    ----------
    n_features_ : int
        The number of features of the data passed to :meth:`fit`.

    Examples
    --------
    >>> from fomo import FomoRegressor
    >>> from pmlb import pmlb
    >>> X,y = pmlb.fetch_data('adult', return_X_y=True)
    >>> groups = ['race','sex']
    >>> est = FomoRegressor()
    >>> est.fit(X,y, protected_features=groups)
    """
    def __init__(self, 
                 estimator: RegressorMixin=SGDRegressor(),
                 fairness_metrics: list[str]=None,
                 accuracy_metrics: list[str]=None,
                 algorithm: str ='NSGA2',
                 random_state: int=None,
                 verbose: bool = False,
                 n_jobs: int = -1,
                 store_final_models: bool = False,
                 problem_type = BasicProblem,
                 checkpoint:bool = False,
                 picking_strategy: str = 'PseudoWeights'
                ):
        super().__init__(
            estimator, 
            fairness_metrics, 
            accuracy_metrics,
            algorithm, 
            random_state,
            verbose,
            n_jobs,
            store_final_models,
            problem_type,
            checkpoint,
            picking_strategy
        )

    def fit(self, X, y, protected_features=None, Xp=None, **kwargs):
        """Train a set of regressors. 

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples,)
            The target values. An array of int.
        protected_features: list|None, default = None
            The columns of X to calculate fairness over. If specifying columns,
            do not also specify `Xp`.
        Xp: pandas DataFrame, shape (n_samples, n_protected_features), default=None
            The protected feature values used to calculate fairness. If `Xp` is 
            specified, `protected_features` must be None. 

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
        super().fit(X, y, protected_features=protected_features, Xp=Xp, **kwargs)

        # Return the regressor
        return self

    def predict(self, X):
        """Predict outcome. 

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

    def _init_metrics(self):
        """ Check metric definitions and/or define when necessary. """
        self.accuracy_metrics_ = []
        self.fairness_metrics_ = []
        if self.accuracy_metrics is None:
            # self.accuracy_metrics_.append(make_scorer(r2_score, greater_is_better=False))
            self.accuracy_metrics_.append(make_scorer(mean_squared_error))
        if len(self.fairness_metrics) == 0:
            self.fairness_metrics_.append(metrics.subgroup_MSE)
