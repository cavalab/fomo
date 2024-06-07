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
import warnings
import ipdb
import numpy as np
import pandas as pd
import logging
import itertools as it
from fomo.utils import categorize 
from sklearn.metrics import mean_squared_error

logger = logging.getLogger(__name__)

def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = it.tee(iterable)
    next(b, None)
    return zip(a, b)

def stratify_groups(X, y, groups,
               n_bins=10,
               bins=None,
               alpha=0.0,
               gamma=0.0
              ):
    """Map data to an existing set of groups, stratified by risk interval."""
    assert isinstance(X, pd.DataFrame), "X should be a dataframe"


    if bins is None:
        bins = np.linspace(float(1.0/n_bins), 1.0, n_bins)
        bins[0] = 0.0
    else:
        n_bins=len(bins)


    df = X[groups].copy()
    df.loc[:,'interval'], retbins = pd.cut(y, bins, 
                                           include_lowest=True,
                                           retbins=True
                                          )
    stratified_categories = {}
    min_size = gamma*alpha*len(X)/n_bins
    for group, dfg in df.groupby(groups):
        # filter groups smaller than gamma*len(X)
        if len(dfg)/len(X) <= gamma:
            continue
        
        for interval, j in dfg.groupby('interval').groups.items():
            if len(j) > min_size:
                if interval not in stratified_categories.keys():
                    stratified_categories[interval] = {}

                stratified_categories[interval][group] = j
    # now we have categories where, for each interval, there is a dict of groups.
    return stratified_categories

def multicalibration_loss(
    estimator,
    X,
    y_true,
    groups=None,
    X_protected=None,
    grouping='intersectional',
    n_bins=None,
    bins=None,
    categories=None,
    proportional=False,
    alpha=0.01,
    gamma=0.01,
    rho=0.1,
    **kwargs
):
    """custom scoring function for multicalibration.
       calculate current loss in terms of (proportional) multicalibration
    """
    if not isinstance(y_true, pd.Series):
        y_true = pd.Series(y_true)

    y_pred = estimator.predict_proba(X)[:,1]
    y_pred = pd.Series(y_pred, index=y_true.index)


    assert isinstance(y_true, pd.Series)
    assert isinstance(y_pred, pd.Series)
    loss = 0.0

    assert groups is not None or X_protected is not None, "groups or X_protected must be defined."

    if categories is None:
        categories = categorize(X, y_pred, groups, grouping,
                                n_bins=n_bins,
                                bins=bins,
                                alpha=alpha, 
                                gamma=gamma
                               )

    for c, idx in categories.items():
        category_loss = np.abs(y_true.loc[idx].mean() 
                               - y_pred.loc[idx].mean()
                              )
        if proportional: 
            category_loss /= max(y_true.loc[idx].mean(), rho)

        if  category_loss > loss:
            loss = category_loss

    return loss

def multicalibration_score(estimator, X, y_true, **kwargs):
    return -multicalibration_loss(estimator, X, y_true, **kwargs)

def proportional_multicalibration_loss(estimator, X, y_true, **kwargs):
    kwargs['proportional'] = True
    return multicalibration_loss(estimator, X, y_true, **kwargs)

def proportional_multicalibration_score(estimator, X, y_true, groups, **kwargs):
    return -proportional_multicalibration_loss(estimator, X, y_true, groups,  **kwargs)

def differential_calibration_loss(
    estimator, 
    X, 
    y_true,
    groups=None,
    X_protected=None,
    n_bins=None,
    bins=None,
    stratified_categories=None,
    alpha=0.0,
    gamma=0.0,
    rho=0.0
):
    """Return the differential calibration of estimator on groups."""

    assert groups is not None or X_protected is not None, "groups or X_protected must be defined."
    assert isinstance(X, pd.DataFrame), "X needs to be a dataframe"
    assert all([g in X.columns for g in groups]), ("groups not found in"
                                                   " X.columns")
    if not isinstance(y_true, pd.Series):
        y_true = pd.Series(y_true)

    y_pred = estimator.predict_proba(X)[:,1]

    if stratified_categories is None:
        stratified_categories = stratify_groups(X, y_pred, groups,
                                n_bins=n_bins,
                                bins=bins,
                                alpha=alpha, 
                                gamma=gamma
                               )
    logger.info(f'# categories: {len(stratified_categories)}')
    dc_max = 0
    logger.info("calclating pairwise differential calibration...")
    for interval in stratified_categories.keys():
        for (ci,i),(cj,j) in pairwise(stratified_categories[interval].items()):

            yi = max(y_true.loc[i].mean(), rho)
            yj = max(y_true.loc[j].mean(), rho)

            dc = np.abs( np.log(yi) - np.log(yj) )

            if dc > dc_max:
                dc_max = dc

    return dc_max

def differential_calibration_score(estimator, X, y_true, **kwargs):
    return -differential_calibration_loss(estimator, X, y_true, **kwargs)

def TPR(y_true, y_pred):
    """Returns True Positive Rate.

    Parameters
    ----------
    y_true: array-like, bool 
        True labels. 
    y_pred: array-like, float or bool
        Predicted labels. 

    If y_pred is floats, this is the "soft" true positive rate 
    (i.e. the average probability estimate for the positive class)
    """
    return np.sum(y_pred[(y_true==1)])/np.sum(y_true)

def FPR(y_true, y_pred):
    """Returns False Positive Rate.

    Parameters
    ----------
    y_true: array-like, bool 
        True labels. 
    y_pred: array-like, float or bool
        Predicted labels. 

    If y_pred is floats, this is the "soft" false positive rate 
    (i.e. the average probability estimate for the negative class)
    """
    # if there are no negative labels, return zero
    if np.sum(y_true) == len(y_true):
        return 0
    yt = y_true.astype(bool)
    return np.sum(y_pred[~yt])/np.sum(~yt)

def FNR(y_true, y_pred):
    """Returns False Negative Rate.

    Parameters
    ----------
    y_true: array-like, bool 
        True labels. 
    y_pred: array-like, float or bool
        Predicted labels. 

    If y_pred is floats, this is the "soft" false negative rate 
    (i.e. the average probability estimate for the negative class)
    """
    # if there are no postive labels, return zero
    if np.sum(y_true) == 0:
        return 0
    yt = y_true.astype(bool)
    return np.sum(1-y_pred[yt])/np.sum(yt)


def subgroup_loss(y_true, y_pred, X_protected, metric, grouping = 'intersectional', abs_val = False, gamma = True):
    assert isinstance(X_protected, pd.DataFrame), "X should be a dataframe"
    if not isinstance(y_true, pd.Series):
        y_true = pd.Series(y_true, index=X_protected.index)
    else:
        y_true.index = X_protected.index

    y_pred = pd.Series(y_pred, index=X_protected.index)

    if (grouping == 'intersectional'):
        groups = list(X_protected.columns)
        categories = X_protected.groupby(groups).groups  
    else:
        categories = {}
        for col in X_protected.columns:
            unique_values = X_protected[col].unique()
            for val in unique_values:
                category_key = f'{col}_{val}'
                mask = X_protected[col] == val
                indices = X_protected[mask].index
                categories[category_key] = indices
    # print('#intersectional groups: ', len(categories))
    # singles = 0
    # gp_lens = [len(lst) for lst in categories.values()]
    # singles = gp_lens.count(1)
    # avg_len = sum(gp_lens) / len(gp_lens) if gp_lens else 0

    if isinstance(metric,str):
        loss_fn = FPR if metric=='FPR' else FNR
    elif callable(metric):
        loss_fn = metric
    else:
        raise ValueError(f'metric={metric} must be "FPR", "FNR", or a callable')

    base_loss = loss_fn(y_true, y_pred)
    max_loss = 0.0
    for c, idx in categories.items():
        # for FPR and FNR, gamma is also conditioned on the outcome probability
        if metric=='FPR' or loss_fn == FPR: 
            g = 1 - np.sum(y_true.loc[idx])/len(X_protected)
        elif metric=='FNR' or loss_fn == FNR: 
            g = np.sum(y_true.loc[idx])/len(X_protected)
        else:
            g = len(idx) / len(X_protected)

        category_loss = loss_fn(
            y_true.loc[idx].values, 
            y_pred.loc[idx].values
        )
        
        deviation = category_loss - base_loss

        if abs_val:
            deviation = np.abs(deviation)
        
        if gamma:
            deviation *= g

        if deviation > max_loss:
            max_loss = deviation

    return max_loss

def subgroup_FPR_loss(y_true, y_pred, X_protected, grouping = 'intersectional', abs_val = False, gamma = True):
    return subgroup_loss(y_true, y_pred, X_protected, 'FPR', grouping, abs_val, gamma)

def subgroup_FNR_loss(y_true, y_pred, X_protected, grouping = 'intersectional', abs_val = False, gamma = True):
    return subgroup_loss(y_true, y_pred, X_protected, 'FNR', grouping, abs_val, gamma)

def subgroup_MSE_loss(y_true, y_pred, X_protected, grouping = 'intersectional', abs_val = False, gamma = True):
    return subgroup_loss(y_true, y_pred, X_protected, mean_squared_error, grouping, abs_val, gamma)

def subgroup_scorer(
    estimator,
    X,
    y_true,
    metric,
    grouping,
    abs_val,
    gamma,
    groups=None,
    X_protected=None,
    weights=None, 
):
    """Calculate the subgroup fairness of estimator on X according to `metric'.
    TODO: handle use case when Xp is passed
    """
    assert isinstance(X, pd.DataFrame), "X should be a dataframe"
    assert groups is not None or X_protected is not None, "groups or X_protected must be defined."

    y_pred = estimator.predict_proba(X)[:,1]
    # y_pred = estimator.predict(X)

    # assert groups is not None, "groups must be defined."
    if groups is None:
        assert X_protected is not None, "cannot define both groups and X_protected"
    else:
        assert X_protected is None, "cannot define both groups and X_protected"
        X_protected = X[groups]

    return subgroup_loss(y_true, y_pred, X_protected, metric, grouping, abs_val, gamma)

def subgroup_FPR_scorer(estimator, X, y_true, **kwargs):
    return subgroup_scorer( estimator, X, y_true, 'FPR', **kwargs)

def subgroup_FNR_scorer(estimator, X, y_true, **kwargs):
    return subgroup_scorer( estimator, X, y_true, 'FNR', **kwargs)

def subgroup_MSE_scorer(estimator, X, y_true, **kwargs):
    return subgroup_scorer( estimator, X, y_true, mean_squared_error, **kwargs)


def flex_loss(estimator, X, y_true, metric, **kwargs):
    """
        returns 
        ----------
        fn: overall loss of all samples
        fng: loss over group for every group in the training data
        samples_fnr: False negative rate of every sample in the training data
        gp_lens: length of each protected group
        
        Parameters
        ----------
        estimator : sklearn-like estimator
            The underlying ML model to be trained.
        X : array-like, shape (n_samples, n_features)
            The training input samples.
        y_true: array-like, bool 
            True labels.
        metric: string or function
            The loss function. Could be FPR or FNR.
        flag: bool
            flag = 1 means marginal grouping and flag = 0 means intersectional grouping
    """
    
    groups = kwargs['groups']
    X_protected = X[groups]
    categories = {}
    fng = []
    samples_fnr = []
    gp_lens = []
    
    y_pred = estimator.predict_proba(X)[:,1]
    y_pred = pd.Series(y_pred, index=X_protected.index)

    if isinstance(metric,str):
        loss_fn = FPR if metric=='FPR' else FNR
    elif callable(metric):
        loss_fn = metric
    else:
        raise ValueError(f'metric={metric} must be "FPR", "FNR", or a callable')

    categories = {}
    for col in X_protected.columns:
        unique_values = X_protected[col].unique()
        for val in unique_values:
            category_key = f'{col}_{val}'
            mask = X_protected[col] == val
            indices = X_protected[mask].index
            categories[category_key] = indices
         
    for c, idx in categories.items():

        category_loss = loss_fn(
            y_true.loc[idx].values, 
            y_pred.loc[idx].values
        )
        fng.append(category_loss)
        gp_lens.append(len(y_true.loc[idx].values))

    # print('#marginal groups: ', len(categories))
    # singles = 0
    # singles = gp_lens.count(1)
    # avg_len = sum(gp_lens) / len(gp_lens) if gp_lens else 0

    #Calculate FNR of each sample
    for idx in y_true.index:
        fnr = loss_fn(y_true[idx], y_pred[idx])
        samples_fnr.append(fnr)

    fn = loss_fn(y_true, y_pred)    
    return fn, fng, samples_fnr, gp_lens


def mce(estimator, X, y_true, num_bins=10):
    """
        The metric to use if fairness is calibration-based.
        Returns maximum calibration error among the bins. 
        
        Parameters
        ----------
        estimator : sklearn-like estimator
            The underlying ML model to be trained.
        X : array-like, shape (n_samples, n_features)
            The training input samples.
        y_true: array-like, bool 
            True labels.
        num_bins: int
            Number of bins that the predictions are sorted and partitioned into. 
    """
    y_pred = estimator.predict_proba(X)[:,1]

    mce = 0
    for i in range(1, num_bins + 1):

        calibration_error = np.abs(y_true.mean() - y_pred.mean())
        mce = max(mce, calibration_error)

    return mce
