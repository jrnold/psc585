""" Water Management Model

The demo is slightly different than the version in the book.
It includes evap and minlevel parameters
"""
import sys
sys.path.append("..")

import scipy as sp

import psc585
from psc585 import dp

## Producer benefit function parameters
alpha1 = 14
beta1 = 0.8
def F(x):
    return alpha1 * (x ** beta1)

## Consumer benefit function parameters
alpha2 = 10
beta2 = 0.4
def G(x):
    return alpha2 * (x ** beta2)

    
## Maximum dam capacity
maxcap = 30
## Rain levels
r = sp.array([0, 1, 2, 3, 4])
## Rain probabilities
p = sp.array([0.1, 0.2, 0.4, 0.2, 0.1])
## Discount factor
delta = 0.9

## Other
evap = 0
minlevel = 10

## Construct state space
S = sp.r_[:maxcap + 1]
n = S.shape[0]

## Action Space
X = sp.r_[:maxcap + 1]
m = X.shape[0]

## Reward function
f = sp.zeros((n, m))
for i in range(n):
    for k in range(m):
        ## Action greater than possible extraction
        if X[k] > S[i]:
            f[i, k] = -sp.Inf
        else:
            geq =  int(S[i] - X[k] >= minlevel)
            f[i, k] = F(X[k]) + G(max(0, (S[i] - X[k]) * geq))

# transition matrix
# Axis 0 : action
# axis 1 : state from
# axis 2 : state to
P = zeros((m, n, n))
for k in range(m):
    for i in range(n):
        for j in range(len(r)):
            snext = min(S[i] - X[k] + r[j], maxcap)
            ## Since states indexed from 0
            ## snext is also the next index
            P[k, i, snext] = P[k, i, snext] + p[j]

            
mod = dp.Ddpsolve.from_transprob(transprob = P, reward = f, discount = delta)    
res = mod.funcit()


