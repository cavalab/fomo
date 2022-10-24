import numpy as np
from pymoo.core.problem import ElementwiseProblem

class FomoProblem(ElementwiseProblem):
    """ The evaluation function for each candidate sample weights. """

    def __init__(self, fomo_estimator=None):
        self.fomo_estimator=fomo_estimator
        super().__init__(
            n_var = len(self.fomo_estimator.X_),
            n_obj = len(self.fomo_estimator.metrics_),
            xl = np.zeros(n_var),
            xu = np.ones(n_var)
        )

    def _evaluate(self, x, out, *args, **kwargs):
        est = (self.fomo_estimator.estimator
              .clone()
              .fit(X,y,sample_weights=x)
             )
        f = np.empty(self.n_obj)
        for i, metric in enumerate(self.fomo_estimator.metrics_):
            f[i] = metric(est, X, y)

        # f1 = 100 * (x[:, 0]**2 + x[:, 1]**2)
        # f2 = (x[:, 0]-1)**2 + x[:, 1]**2
        # out["F"] = np.column_stack([f1, f2])
        out['F'] = np.asarray(f)
