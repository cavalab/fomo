Fomo is a **F**\airness **O**\riented **M**\ultiobjective **O**\ptimization toolbox for training regression and classification models. 
Fomo focuses on optimizing the trade-offs between various metrics in ML fairness that are often in direct conflict. 
The goal is to assist decision makers in weighing multiple criteria by generating good solution *sets*, rather than singular models. 

Introduction 
============

Fomo is designed to be incredibly *general*. 
It works with any ML model that has a `scikit-learn <https://scikit-learn.org>`_ interface (i.e. `fit()` and `predict()` methods) and takes sample weights as part of its loss function. 
Specifically, the `fit()` method should optionally take an argument, `sample_weight`, that provides a weight to each observation in `X`,`y`. 
That covers nearly all estimators in `sklearn`, including linear models  (linear and logistic regression, lasso), SVMs, neural nets, decision trees, and ensemble methods like random forests, gradient boosting, and XGBoost. 

In addition, Fomo works with many different *metrics* of fairness and accuracy. 
It currently supports:

- Subgroup Fairness (False Positive, False Negative, and Demographic Parity)
- Differential Fairness (Demographic Parity and Calibration)
- Multicalibration

In addition, users can specify any callable function they would like to be optimized, as long as it matches the call signature of these functions. 
Users can specify the combination of performance metrics and fairness metrics that best suit the task they are studying. 
You can specify any number and combinatoin of these metrics. 

Finally, Fomo works with many different *optimization* methods available from `pymoo <https://pymoo.org/>`_, including NSGA-II, NSGA-III, MOEAD, and others. 

Installation
============

The requirements for `fomo` are listed in `environment.yml`.
To install, do the following:

.. code-block:: bash

    git clone https://github.com/cavalab/fomo
    cd fomo
    pip install . 

Quick Start
============


.. code-block:: python

    from fomo import FomoClassifier
    from pmlb import pmlb
    X,y = pmlb.fetch_data('adult', return_X_y=True)
    groups = ['race','sex']
    est = FomoClassifier()
    est.fit(X,y, protected_features=groups)

License
=======

Fomo is licensed under GNU Public License v. 3.0.  See `LICENSE <https://github.com/cavalab/fomo/blob/main/LICENSE>`_.

Contact
============

- William La Cava: william dot lacava at childrens dot harvard dot edu
