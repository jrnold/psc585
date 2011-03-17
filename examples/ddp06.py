"""Bioeconomic Model """
import sys
sys.path.append("..")

import scipy as sp

import psc585
from psc585 import dp

T = 10
## Energy capacity
emax = 8
## Model parameters
e = sp.array([2, 4, 4])
## Predation survival probabilities
p = sp.array([1.0, 0.7, 0.8])
## foraging success probabilities
q = sp.array([0.5, 0.8, 0.7])

## State space
S = sp.r_[:(emax + 1)]
## number of states
n = len(S)
# Number of actions
m = 3

## Reward matrix
f = sp.zeros((n, m))

def getindex(x, A):
    return (x == A).nonzero()[0][0]

P = sp.zeros((m, n, n))
for k in range(m):
    P[k, 0, 0] = 1
    for i in range(1, n):
        ## does not survive predation
        snext = 0
        j = getindex(snext, S)
        P[k, i, j] = P[k, i, j] + 1 - p[k]
        # survives and finds food
        snext = min(S[i] - 1 + e[k], emax)
        j = getindex(snext, S)
        P[k, i, j] = P[k, i, j] + p[k] * q[k]
        # survives, does not find food
        snext = S[i] - 1
        j = getindex(snext, S)
        P[k, i, j] = P[k, i, j] + p[k] * (1 - q[k])
        
# terminal values
vterm = sp.ones((n, 1))
vterm[0, 0] = 0


ddp06 = dp.Ddpsolve(discount=1,
                    reward = f,
                    P=P,
                    T=T,
                    vterm=vterm)

foo = ddp06.backsolve()


            
            

