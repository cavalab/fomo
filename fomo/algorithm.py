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

import random
#from pymoo.algorithms.base.genetic import GeneticAlgorithm
from pymoo.operators.selection.tournament import compare, TournamentSelection
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga2 import binary_tournament
from fomo.utils import categorize
from pymoo.core.survival import Survival
from pymoo.core.selection import Selection
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.survival.rank_and_crowding import RankAndCrowding
from pymoo.util.display.multi import MultiObjectiveOutput
from pymoo.termination.default import DefaultMultiObjectiveTermination
from pymoo.util.misc import has_feasible
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.algorithms.base.genetic import GeneticAlgorithm

def get_parent(pop):
    
    fng = pop.get("fng")
    fn = pop.get("fn")
    G = np.arange(fng.shape[1])
    S = np.arange(len(pop))
    loss = []

    while (len(G) > 0 and len(S) > 1):

        g = random.choice(G)
        loss = []
        
        if (random.random() < 0.5):
            #half the time look at accuracy
            loss = fng[:, g]
        else:
            #half the time look at fairness
            loss = np.abs(fng[:, g] - fn)

        L = min(loss) 
        epsilon = np.median(np.abs(loss - np.median(loss)))
        survivors = np.where(loss <= L + epsilon)
        S = S[survivors]
        fng = fng[survivors] 
        fn = fn[survivors]
        G = G[np.where(G != g)]
            
    S = S[:, None].astype(int, copy=False)     
    return random.choice(S)
                
def get_parent_noCoinFlip(pop):

    fng = pop.get("fng")
    fng = np.tile(fng, 2)
    fn = pop.get("fn")
    G = np.arange(fng.shape[1])
    S = np.arange(len(pop))
    loss = []

    while (len(G) > 0 and len(S) > 1):

        g = random.choice(G)
        loss = []
        
        if g < max(G)/2:
            #look at accuracy
            loss = fng[:, g]
        else:
            #look at fairness
            loss = np.abs(fng[:, g] - fn)

        L = min(loss) 
        epsilon = np.median(np.abs(loss - np.median(loss)))
        survivors = np.where(loss <= L + epsilon)
        S = S[survivors]
        fng = fng[survivors] 
        fn = fn[survivors]
        G = G[np.where(G != g)]

            
    S = S[:, None].astype(int, copy=False)     
    return random.choice(S)


class FLEX(Selection):
    
    def __init__(self,
                 **kwargs):

        #self.X_protected_ = X_protected
        #self.categories = categories
        #self.group_losses = group_losses
        super().__init__(**kwargs)
     
         
    def _do(self, _, pop, n_select, n_parents=1, flag = 0, **kwargs):

        # offss = super().n_offsprings
        # s = n_select * n_parents
        # S = np.random.randint(0, s, s)
        # S = S[:, None].astype(int, copy=False)
        
        parents = []
        
        for i in range(n_select * n_parents): 
            #get pop_size parents
            p = get_parent_noCoinFlip(pop)
            parents.append(p)
            
        return np.reshape(parents, (n_select, n_parents))

class LexSurvival(Survival):
    def __init__(self) -> None:
        super().__init__(filter_infeasible=False)

    def _do(self, problem, pop, n_survive=None, **kwargs):
        return pop[-n_survive:]


class Lexicase_NSGA2(GeneticAlgorithm):

    def __init__(self,
                 pop_size=100,
                 sampling=FloatRandomSampling(),
                 selection=FLEX(),
                 crossover=SBX(eta=15, prob=0.9),
                 mutation=PM(eta=20),
                 survival=RankAndCrowding(),
                 output=MultiObjectiveOutput(),
                 **kwargs):
        
        super().__init__(
            pop_size=pop_size,
            sampling=sampling,
            selection=selection,
            crossover=crossover,
            mutation=mutation,
            survival=survival,
            output=output,
            advance_after_initial_infill=True,
            **kwargs)
        
        self.termination = DefaultMultiObjectiveTermination()
        self.tournament_type = 'comp_by_dom_and_crowding'

    def _set_optimum(self, **kwargs):
        if not has_feasible(self.pop):
            self.opt = self.pop[[np.argmin(self.pop.get("CV"))]]
        else:
            self.opt = self.pop[self.pop.get("rank") == 0]

class Lexicase(GeneticAlgorithm):

    def __init__(self,
                 pop_size=100,
                 sampling=FloatRandomSampling(),
                 selection=FLEX(),
                 crossover=SBX(eta=15, prob=0.9),
                 mutation=PM(eta=20),
                 survival=LexSurvival(),
                 output=MultiObjectiveOutput(),
                 **kwargs):
        
        super().__init__(
            pop_size=pop_size,
            sampling=sampling,
            selection=selection,
            crossover=crossover,
            mutation=mutation,
            survival=survival,
            output=output,
            advance_after_initial_infill=True,
            **kwargs)