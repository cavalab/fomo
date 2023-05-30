"""
Fairness Oriented Multiobjective Optimization (Fomo)
Copyright (C) {2023}  William La Cava

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
import numpy as np
from pymoo.core.problem import ElementwiseProblem
from sklearn.base import clone, ClassifierMixin
from sklearn.utils import resample
from sklearn.pipeline import Pipeline
import warnings
import inspect
from .surrogate_models import MLP, Linear
import random
import fomo.metrics as metrics

class BasicProblem(ElementwiseProblem):
    """ The evaluation function for each candidate sample weights. """

    def __init__(
        self, 
        fomo_estimator, 
        metric_kwargs={},
        **kwargs
        ):
        self.fomo_estimator=fomo_estimator
        self.metric_kwargs=metric_kwargs
        n_var = len(self.fomo_estimator.X_)
        n_obj = (len(self.fomo_estimator.accuracy_metrics_)
                  +len(self.fomo_estimator.fairness_metrics_)
        )
        
        super().__init__(
            n_var = n_var,
            n_obj = n_obj,
            xl = np.zeros(n_var),
            xu = np.ones(n_var),
            **kwargs
        )

    def get_sample_weight(self, x):
        return x

    def _evaluate(self, sample_weight, out, *args, **kwargs):
        """Evaluate the weights by fitting an estimator and evaluating."""

        X = self.fomo_estimator.X_
        y = self.fomo_estimator.y_
        est = clone(self.fomo_estimator.estimator)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if isinstance(est, Pipeline):
                stepname = est.steps[-1][0]
                param_name = stepname + '__sample_weight'
                kwarg = {param_name:sample_weight}
                est.fit(X, y, **kwarg)
            else:
                est.fit(X,y,sample_weight=sample_weight)
                
        f = np.empty(self.n_obj)
        j = 0
        for i, metric in enumerate(self.fomo_estimator.accuracy_metrics_):
            f[i] = metric(est, X, y)
            fn = f[0]
            j += 1
        for metric in self.fomo_estimator.fairness_metrics_:
            f[j] = metric(est, X, y, **self.metric_kwargs)
            j += 1
        
        out['F'] = np.asarray(f)
        out['fn'] = fn
        #out['fng'] = metrics.fng(est, X, y, self.metric_kwargs['groups'], 'FPR')
        out['fng'] = metrics.fng(est, X, y, 'FPR', **self.metric_kwargs)

class SurrogateProblem(ElementwiseProblem):
    """ The evaluation function for each candidate weights. 

    """

    def __init__( self, fomo_estimator, metric_kwargs={}, **kwargs):

        self.fomo_estimator=fomo_estimator
        self.metric_kwargs=metric_kwargs

        n_obj = (len(self.fomo_estimator.accuracy_metrics_)
                 +len(self.fomo_estimator.fairness_metrics_)
        )

        for k,v in self.metric_kwargs.items(): 
            print(k,v)
            if v is not None:
                if k == 'X_protected':
                    self.X_protected = v
                elif k == 'groups':
                    X = self.fomo_estimator.X_
                    self.X_protected = X[v]
                else:
                    print('no match for',k)
                break

        n_var = self._get_surrogate().get_n_weights()

        super().__init__(
            n_var = n_var,
            n_obj = n_obj,
            # set lower bound to -1
            xl = -np.ones(n_var),
            # set upper bound to 1
            xu = np.ones(n_var),
            **kwargs
        )

    def _get_surrogate(self):
        raise NotImplementedError('Dont call _get_surrogate on this base class!')

    def get_sample_weight(self, x):
        surrogate = self._get_surrogate()
        surrogate.set_weights(x)
        sample_weight = surrogate.predict(self.X_protected)
        return sample_weight

    def _evaluate(self, x, out, *args, **kwargs):
        """Evaluate the weights by fitting an estimator and evaluating."""
        X = self.fomo_estimator.X_
        y = self.fomo_estimator.y_

        sample_weight = self.get_sample_weight(x)
        # sample_weight = sample_weight.ravel()
        assert len(sample_weight) == len(X)

        est = clone(self.fomo_estimator.estimator)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if isinstance(est, Pipeline):
                stepname = est.steps[-1][0]
                param_name = stepname + '__sample_weight'
                kwarg = {param_name:sample_weight}
                est.fit(X, y, **kwarg)
            else:
                est.fit(X,y,sample_weight=sample_weight)

        f = np.empty(self.n_obj)
        j = 0
        for i, metric in enumerate(self.fomo_estimator.accuracy_metrics_):
            f[i] = metric(est, X, y)
            j += 1
        for metric in self.fomo_estimator.fairness_metrics_:
            f[j] = metric(est, X, y, **self.metric_kwargs)
            j += 1

        out['F'] = np.asarray(f)

class MLPProblem(SurrogateProblem):
    """ The evaluation function for each candidate weights. 

    """
    def _get_surrogate(self):
        return MLP(hidden_layer_sizes=(10,)).init(self.X_protected)

class LinearProblem(SurrogateProblem):
    """ The evaluation function for each candidate weights. 

    """
    def _get_surrogate(self):
        return Linear(self.X_protected)
