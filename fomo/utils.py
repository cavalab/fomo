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

