## Overview

<!-- start overview -->

Fomo is a **F**airness **O**riented **M**ultiobjective **O**ptimization toolbox for training regression and classification models. 
Fomo focuses on optimizing the trade-offs between various metrics in ML fairness that are often in direct conflict. 
The goal is to assist decision makers in weighing multiple criteria by generating good solution *sets*, rather than singular models. 

## Cite

If you use Fomo please cite the following: 

- W.G. La Cava (2023). Optimizing fairness tradeoffs in machine learning with multiobjective meta-models. GECCO '23. [arXiv:2304.12190](https://arxiv.org/abs/2304.12190)

<!-- end overview -->

## Introduction 

<!-- start introduction -->

Fomo is designed to be incredibly *general*. 
It works with any ML model that has a [scikit-learn](https://scikit-learn.org) interface (i.e. `fit()` and `predict()` methods) and takes sample weights as part of its loss function. 
Specifically, the `fit()` method should optionally take an argument, `sample_weight`, that provides a weight to each observation in `X`,`y`. 
That covers nearly all estimators in `sklearn`, including linear models  (linear and logistic regression, lasso), SVMs, neural nets, decision trees, and ensemble methods like random forests, gradient boosting, and XGBoost. 

In addition, Fomo works with many different *metrics* of fairness and accuracy. 
It currently supports:

- Subgroup Fairness (False Positive, False Negative, and Demographic Parity)
- Differential Fairness (Demographic Parity and Calibration)
- Multicalibration
- Proportional Multicalibration

In addition, users can specify any callable function they would like to be optimized, as long as it matches the call signature of these functions. 
Users can specify the combination of performance metrics and fairness metrics that best suit the task they are studying. 
You can specify any number and combinatoin of these metrics. 

Finally, Fomo works with many different *optimization* methods available from [pymoo](https://pymoo.org/), including NSGA-II, NSGA-III, MOEAD, and others. 

<!-- end introduction -->

## Quick Start

<!-- start quickstart -->

### Installation

<!-- start installation -->

```text
pip install pyfomo
```

### Dependencies

The requirements for `fomo` are listed in `environment.yml`.

**Note on pymoo** If you are working in linux and get a warning about pymoo, is recommended that you manually install it from the github repo rather than pip:

```bash

git clone https://github.com/anyoptimization/pymoo
cd pymoo
make compile
pip install .

```

#### Development 

To install a development version, do the following:

```bash

git clone https://github.com/cavalab/fomo
cd fomo
pip install . 

```

<!-- end installation -->

### Basic Usage

Here is an example of training a fair classifier on the adult income prediction dataset. 

```python

from fomo import FomoClassifier
from pmlb import pmlb
dataset = pmlb.fetch_data('adult')
X = dataset.drop('target',axis=1)
y = dataset['target']
groups = ['race','sex']
est = FomoClassifier()
est.fit(X,y, protected_features=groups)

```

<!-- end quickstart -->

## License

<!-- start license -->

See [LICENSE](https://github.com/cavalab/fomo/blob/main/LICENSE).

<!-- end license -->

## Contact

<!-- start contact -->


- William La Cava: william dot lacava at childrens dot harvard dot edu

<!-- end contact -->
