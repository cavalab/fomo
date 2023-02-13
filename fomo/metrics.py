"""Implementation of different fairness metrics.
TODO:
    - false positive fairness
    - false negative fairness
    - mean squared error
"""
import warnings
import ipdb
import numpy as np
import pandas as pd
from tqdm import tqdm
import logging
import itertools as it
from fomo.utils import categorize 
from sklearn.metrics import mean_squared_error
import gerryfair

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
    rho=0.1
):
    """custom scoring function for multicalibration.
       calculate current loss in terms of (proportional) multicalibration"""
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


def subgroup_loss(y_true, y_pred, X_protected, metric):
    assert isinstance(X_protected, pd.DataFrame), "X should be a dataframe"
    if not isinstance(y_true, pd.Series):
        y_true = pd.Series(y_true, index=X_protected.index)
    else:
        y_true.index = X_protected.index

    y_pred = pd.Series(y_pred, index=X_protected.index)

    groups = list(X_protected.columns)
    categories = X_protected.groupby(groups).groups

    if isinstance(metric,str):
        loss_fn = FPR if metric=='FPR' else FNR
    elif callable(metric):
        loss_fn = metric
    else:
        raise ValueError(f'metric={metric} must be "FPR", "FNR", or a callable')

    max_loss = 0.0
    for c, idx in categories.items():
        gamma = len(idx) / len(X_protected)
        category_loss = gamma*loss_fn(
            y_true.loc[idx].values, 
            y_pred.loc[idx].values
        )

        if  category_loss > max_loss:
            max_loss = category_loss

    return max_loss

def subgroup_FPR_loss(y_true, y_pred, X_protected):
    return subgroup_loss(y_true, y_pred, X_protected, 'FPR')

def subgroup_FNR_loss(y_true, y_pred, X_protected):
    return subgroup_loss(y_true, y_pred, X_protected, 'FNR')

def subgroup_fairness(
    estimator,
    X,
    y_true,
    metric,
    groups=None,
    X_protected=None,
    grouping='intersectional'
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

    return subgroup_loss(y_true, y_pred, X_protected, metric)

def subgroup_FPR(estimator, X, y_true, **kwargs):
    return subgroup_fairness( estimator, X, y_true, 'FPR', **kwargs)

def subgroup_FNR(estimator, X, y_true, **kwargs):
    return subgroup_fairness( estimator, X, y_true, 'FNR', **kwargs)

def subgroup_MSE(estimator, X, y_true, **kwargs):
    return subgroup_fairness( estimator, X, y_true, mean_squared_error, **kwargs)


def auditor_loss(y_true, y_pred, X_protected, metric):
    auditor = gerryfair.model.Auditor(X_protected, y_true, metric)
    _,violation = auditor.audit(y_pred)
    return violation

def auditor_FPR(y_true, y_pred, X_protected):
    return auditor_loss(y_true, y_pred, X_protected, 'FP')

def auditor_FNR(y_true, y_pred, X_protected):
    return auditor_loss(y_true, y_pred, X_protected, 'FN')

def audit(
    estimator,
    X,
    y_true,
    metric:str,
    groups=None,
    X_protected=None,
    grouping='intersectional'
):
    assert isinstance(X, pd.DataFrame), "X should be a dataframe"
    assert groups is not None or X_protected is not None, "groups or X_protected must be defined."
    if isinstance(y_true, pd.Series):
       y_true = y_true.values

    y_pred = estimator.predict(X)
    # y_pred = pd.Series(y_pred, index=y_true.index)

    assert metric in ('FP','FN'), f'metric must be "FP" or "FN", not {metric}'
    # assert isinstance(y_true, pd.Series)
    # assert isinstance(y_pred, pd.Series)

    if groups is not None:
        assert X_protected is None, "cannot define both groups and X_protected"
        X_protected = X[groups]

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return auditor_loss(y_true, y_pred, X_protected, metric)

def audit_FPR_loss(
    estimator,
    X,
    y_true,
    groups=None,
    X_protected=None,
    grouping='intersectional'
):
    return audit(
        estimator,
        X,
        y_true,
        'FP',
        groups,
        X_protected,
        grouping
    )

def audit_FNR_loss(
    estimator,
    X,
    y_true,
    groups=None,
    X_protected=None,
    grouping='intersectional'
):
    return audit(
        estimator,
        X,
        y_true,
        'FN',
        groups,
        X_protected,
        grouping
    )
