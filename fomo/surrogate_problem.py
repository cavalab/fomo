import numpy as np
from pymoo.core.problem import ElementwiseProblem
from sklearn.base import clone, ClassifierMixin
from sklearn.utils import resample
import warnings
import inspect
from .surrogate_models import MLP

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
        return MLP(hidden_layer_sizes=(10,)).init(self.X_protected)

    def get_sample_weight(self, x):
        surrogate = self._get_surrogate()
        surrogate.set_weights(x)
        sample_weight = surrogate.predict_proba(self.X_protected)[:,1]
        return sample_weight

    def _evaluate(self, x, out, *args, **kwargs):
        """Evaluate the weights by fitting an estimator and evaluating."""
        try:
            X = self.fomo_estimator.X_
            y = self.fomo_estimator.y_

            sample_weight = self.get_sample_weight(x)
            # sample_weight = sample_weight.ravel()
            assert len(sample_weight) == len(X)

            est = clone(self.fomo_estimator.estimator)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
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
        except Exception as e:
            print(e)
            import ipdb
            ipdb.set_trace()