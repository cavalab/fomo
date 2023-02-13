import copy
import pytest
import pandas as pd
from fomo import FomoClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.metrics import make_scorer, accuracy_score
from pmlb import pmlb   
import fomo.metrics as metrics
from pymoo.algorithms.moo.nsga2 import NSGA2

dataset = pmlb.fetch_data('adult', 
                          local_cache_dir='/home/bill/projects/pmlb'
)
# dataset = dataset.sample(n=1000)
X = dataset.drop('target',axis=1)
y = dataset['target']
Xtrain,Xtest, ytrain,ytest = train_test_split(X,y,
                                            stratify=y, 
                                            random_state=42,
                                            test_size=0.5
                                           )
# ss = StandardScaler()
# Xtrain = pd.DataFrame(ss.fit_transform(Xtrain), columns=Xtrain.columns, index=ytrain.index)
# Xtest = pd.DataFrame(ss.transform(Xtest), columns=Xtest.columns, index=ytest.index)
# groups = ['age','workclass','race','sex','native-country']
groups = ['race','sex']
# groups = ['race', 'sex']
# groups = list(X.columns)


# testdata = [
#     (metrics.multicalibration_loss,'marginal'), 
#     (metrics.multicalibration_loss,'intersectional'), 
#     (metrics.subgroup_FNR, 'marginal'),
#     (metrics.subgroup_FNR, 'intersectional')
#            ]
from xgboost import XGBClassifier
from sklearn.linear_model import SGDClassifier
from fomo.problem import BasicProblem, MLPProblem, LinearProblem

est = FomoClassifier(
#     estimator = XGBClassifier(),
    estimator = LogisticRegression(),
    algorithm = NSGA2(),
    # accuracy_metrics=[make_scorer(metrics.FPR)],
    # fairness_metrics=[metrics.subgroup_audit_FPR_loss],
    fairness_metrics=[metrics.subgroup_FNR],
    verbose=True,
    batch_size=0,
    problem_type=MLPProblem,
    # problem_type=BasicProblem,
    n_jobs=1
)

est.fit(Xtrain,ytrain,protected_features=groups, termination=('n_gen',5))