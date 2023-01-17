import numpy as np
from pymoo.core.problem import ElementwiseProblem
from sklearn.base import clone, ClassifierMixin
from sklearn.utils import resample
import warnings

class FomoProblem(ElementwiseProblem):
    """ The evaluation function for each candidate sample weights. """

    def __init__(self, fomo_estimator=None, **kwargs):
        self.fomo_estimator=fomo_estimator
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

    def _evaluate(self, sample_weight, out, *args, **kwargs):
        """Evaluate the weights by fitting an estimator and evaluating."""

        X = self.fomo_estimator.X_
        y = self.fomo_estimator.y_
        # batch sampling
        if self.fomo_estimator.batch_size > 0:
            stratify = y if isinstance(self.fomo_estimator, ClassifierMixin) else None
            X, y, sample_weight = resample(
                X, 
                y,
                sample_weight, 
                n_samples=self.fomo_estimator.batch_size,
                random_state=self.fomo_estimator.random_state, 
                stratify=stratify
            )

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
            f[j] = metric(est, X, y, **self.fomo_estimator.metric_kwargs)
            j += 1

        out['F'] = np.asarray(f)
