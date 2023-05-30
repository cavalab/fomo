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
import numpy as np
import pandas as pd

def squash_array(x):
    x[x<0.0] == 0.0
    x[x>1.0] == 1.0
    return x

def squash_series(x):
    return x.apply(lambda x: max(x,0.0)).apply(lambda x: min(x,1.0))

def category_diff(cat1, cat2):
    different=False
    for k1,v1 in cat1.items():
        if k1 not in cat2.keys():
            print(f'{k1} not in cat2')
            different=True
        else:
            if not v1.equals(cat2[k1]):
                print(f'indices for {k1} different in cat2')
                different=True
    for k2,v2 in cat2.items():
        if k2 not in cat1.keys():
            print(f'{k1} not in cat1')
            different=True
        else:
            if not v2.equals(cat1[k2]):
                print(f'indices for {k2} different in cat1')
                different=True
    if not different:
        # print('categories match.')
        return True
    else:
        return False

def categorize(X, y, groups, grouping,
               n_bins=10,
               bins=None,
               alpha=0.01,
               gamma=0.01
              ):
    """Map data to an existing set of categories."""
    assert isinstance(X, pd.DataFrame), "X should be a dataframe"

    categories = None 

    if bins is None:
        if n_bins is None:
            n_bins = 10
        bins = np.linspace(float(1.0/n_bins), 1.0, n_bins)
        bins[0] = 0.0
    else:
        n_bins=len(bins)


    df = X[groups].copy()
    df.loc[:,'interval'], retbins = pd.cut(y, bins, 
                                           include_lowest=True,
                                           retbins=True
                                          )
    categories = {}
    # group_ids = groupby(X, y, groups, grouping)
    group_ids = df.groupby(groups).groups

    min_grp_size = gamma*len(X) 
    min_cat_size = min_grp_size*alpha/n_bins
    for group, i in group_ids.items():
        # filter groups smaller than gamma*len(X)
        if len(i) <= min_grp_size:
            continue
        for interval, j in df.loc[i].groupby('interval').groups.items():
            # filter categories smaller than alpha*gamma*len(X)/n_bins
            if len(j) > min_cat_size:
                categories[group + (interval,)] = j
    return categories

from pymoo.decomposition.asf import ASF
class Compromise:
    def __init__(self, weights):
        self.weights=weights
    def do(self,F):
        decomp = ASF()
        I = decomp(F, self.weights).argmin()
        return I