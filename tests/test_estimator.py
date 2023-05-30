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
    (metrics.subgroup_FNR_scorer, 'marginal'),
    (metrics.subgroup_FNR_scorer, 'intersectional')
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