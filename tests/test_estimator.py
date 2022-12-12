import copy
import pytest
import pandas as pd
from fomo import FomoClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, average_precision_score
from pmlb import pmlb   
import fomo.metrics as metrics

dataset = pmlb.fetch_data('adult', 
                          local_cache_dir='/home/bill/projects/pmlb'
)
X = dataset.drop('target',axis=1)
y = dataset['target']
Xtrainval,Xtest, ytrainval,ytest = train_test_split(X,y,
                                                    stratify=y,
                                                    random_state=42,
                                                    test_size=0.2
                                                   )
Xtrain,Xval, ytrain,yval = train_test_split(Xtrainval,ytrainval,
                                            stratify=ytrainval, 
                                            random_state=42,
                                            test_size=0.5
                                           )

# groups = ['age','workclass','race','sex','native-country']
groups = ['race','sex','native-country']
# groups = ['race', 'sex']
# groups = list(X.columns)



testdata = [
    (metrics.multicalibration_loss,'marginal'), 
    (metrics.multicalibration_loss,'intersectional'), 
    (metrics.subgroup_FNR, 'marginal'),
    (metrics.subgroup_FNR, 'intersectional')
           ]

@pytest.mark.parametrize("metric,grouping", testdata)
def test_training(metric,grouping):
    """Test training"""
    params = dict(
        estimator = LogisticRegression(),
        fairness_metrics=[metric],
        # algorithm=NSGA2()
        # eta = 0.3,
        # gamma=0.1,
        # alpha=0.1,
        # rho=0.2,
        # max_iters=1000,
        # verbosity=2,
        # n_bins=5,
        # iter_sample='bootstrap'
    )

    est = FomoClassifier(**params)

    est.fit(Xval,yval)

    print('model\tfold\tAUROC\tAUPRC\tMC\tPMC')
    for x,y_true,fold in [(Xtrain, ytrain,'train'), 
                          (Xval, yval,'val'),
                          (Xtest, ytest,'test')]:
        y_pred = pd.Series(est.predict_proba(x)[:,1], index=x.index)
        print(metric,end='\t')
        print(fold,end='\t')
        for score in [roc_auc_score, average_precision_score]: 
            print(f'{score(y_true, y_pred):.3f}',end='\t')
        mc = metrics.multicalibration_loss(
            est,
            x, 
            y_true, 
            groups,
            grouping=grouping,
            alpha=params['alpha'], 
            gamma=params['gamma'],
            n_bins=params['n_bins'],
            rho=params['rho']
        )
        pmc = metrics.proportional_multicalibration_loss(
            est,
            x, 
            y_true, 
            groups,
            grouping=grouping,
            proportional=True,
            alpha=params['alpha'], 
            gamma=params['gamma'],
            n_bins=params['n_bins'],
            rho=params['rho']
        )
        print(f'{mc:.3f}',end='\t')
        print(f'{pmc:.3f}')
        # print('-----------')

