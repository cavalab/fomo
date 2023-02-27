# fomo
FOMO is a **f**airness-**o**riented **m**ultiobjective **o**ptimization method for training regression and classification models. 

# example usage

```python

from fomo import FomoClassifier
from pmlb import pmlb
X,y = pmlb.fetch_data('adult', return_X_y=True)
groups = ['race','sex']
est = FomoClassifier()

est.fit(X,y, protected_features=groups)
```