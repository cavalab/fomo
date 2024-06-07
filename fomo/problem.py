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
import numpy as np
from pymoo.core.problem import ElementwiseProblem
from sklearn.base import clone, ClassifierMixin
from sklearn.utils import resample
from sklearn.pipeline import Pipeline
import warnings
import inspect
import fomo.metrics as metrics
from .surrogate_models import MLP, Linear, InterLinear
from fomo.algorithm import Lexicase, Lexicase_NSGA2

class BasicProblem(ElementwiseProblem):
    """ The evaluation function for each candidate sample weights. """

    def __init__( self, fomo_estimator, metric_kwargs={}, **kwargs):
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
            j += 1
        for metric in self.fomo_estimator.fairness_metrics_:
            f[j] = metric(est, X, y, **self.metric_kwargs)
            j += 1
            
        out['F'] = np.asarray(f)

        if isinstance(self.fomo_estimator.algorithm, (Lexicase, Lexicase_NSGA2)):
            fn, fng, samples_fnr, gp_lens = metrics.flex_loss(est, X, y, 'FNR', **self.metric_kwargs)
            out['fn'] = fn #FNR of all samples to be used in Flex
            out['fng'] = fng #FNR of every group to be used in Flex
            out['samples_fnr'] = samples_fnr #FNR of each sample to be used in Flex with weighted coin flip
            out['gp_lens'] = gp_lens #Length of each protected group to be used in Flex with weighted coin flip


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

        if isinstance(self.fomo_estimator.algorithm, (Lexicase, Lexicase_NSGA2)):
            fn, fng, samples_fnr, gp_lens = metrics.flex_loss(est, X, y, 'FNR', **self.metric_kwargs)
            out['fn'] = fn #FNR of all samples to be used in Flex
            out['fng'] = fng #FNR of every group to be used in Flex
            out['samples_fnr'] = samples_fnr #FNR of each sample to be used in Flex with weighted coin flip
            out['gp_lens'] = gp_lens #Length of each protected group to be used in Flex with weighted coin flip

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

class InterLinearProblem(SurrogateProblem):
    """ The evaluation function for each candidate weights. 

    """
    def _get_surrogate(self):
        return InterLinear(self.X_protected)
