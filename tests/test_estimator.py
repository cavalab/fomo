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
import copy
import pytest
import pandas as pd
from fomo import FomoClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, average_precision_score
from pmlb import pmlb   
import fomo.metrics as metrics

dataset = pmlb.fetch_data('adult')
dataset = dataset.sample(n=2000)
X = dataset.drop('target',axis=1)
y = dataset['target']
Xtrain,Xtest, ytrain,ytest = train_test_split(X,y,
                                            stratify=y, 
                                            random_state=42,
                                            test_size=0.5
                                           )

# groups = ['age','workclass','race','sex','native-country']
GROUPS = ['race','sex','native-country']
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
    est = FomoClassifier(
        estimator = LogisticRegression(),
        fairness_metrics=[metric],
        verbose=True
    )

    est.fit(Xtrain,ytrain,protected_features=GROUPS, termination=('n_gen',1))

    print('model\tfold\tAUROC\tAUPRC\tMC\tPMC')
    for x,y_true,fold in [(Xtrain, ytrain,'train'), 
                          (Xtest, ytest,'test')]:
        y_pred = pd.Series(est.predict_proba(x)[:,1], index=x.index)
        print(metric,end='\t')
        print(fold,end='\t')
        score = metric(
            est,
            x, 
            y_true, 
            groups=GROUPS
        )
        print(f'Fold:{fold}\t{metric.__name__}:{score}',end='\t')
        for score in [roc_auc_score, average_precision_score]: 
            print(f'{score.__name__}: {score(y_true, y_pred):.3f}')

if __name__=='__main__':
    for td in testdata:
        test_training(*td)