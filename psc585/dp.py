"""Dynamic Programming """
import scipy as sp
from scipy import linalg as la
from scipy import sparse
from scipy.sparse import linalg as spla

def _expandg(g):
    """ Expand transition function to a matrix
    """
    return sparse.coo_matrix((sp.ones(prod(g.shape)),
                             (sp.r_[0:prod(g.shape)], g.flatten())))

class Ddpsolve(object):
    """
    Attributes
    ------------
    discount: float
       Discount factor. Must be on (0, 1) for infinite horizon problems, 
       or (0, 1] for finite horizon problems.
    reward: array, shape (n, m)
       Reward values
    transfunc: array, shape (n, m)
       Deterministic transition function.
    T: int, optional
       Number of time periods. Set to `None` for
       infinite horizon problems.
    vterm: optional
       Terminal value function for finite horizon problems.

    """
    
    def __init__(self, discount, reward, transfunc, T=None, vterm=None):
        self.discount = discount
        self.reward = reward
        self.P = _expandg(transfunc)
        self.n = reward.shape[0]
        self.m = reward.shape[1]
        if T and not vterm:
            self.vterm = sp.zeros(reward.shape[0])

    def valmax(self, v):
        """ Belman Euqation

        Notes
        ------

        Solve the Bellman equation for a given v

        .. math::
            V(s) = max_{x \in X(x)}
            \left\{
                f(s, x) + \delta \sum_{s' \in S} P(s' | s, x) V(s')
             \right\}            
        """
        U = self.reward + sp.reshape(foo.discount * self.P.dot(v), (self.n, self.m))
        ## argmax by row
        x = U.argmax(1)
        ## maximum values for these actions
        v = U[sp.r_[0:self.m], x]
        return (v, x)

    def valpol(self, x):
        """Returns state function and state transition prob matrix induced by a policy
        """
        

    def backsolve(self):
        """ Solve Bellman equation via backward recursion"""
        x = sp.zeros(self.n, self.T)
        v = sp.concatenate(sp.zeros(self.n, self.T), 1)
        pstar = zeros(sp.n, sp.n, sp.T)
        for t in range(sp.T):
            print t
            
        

    

# Market price
price = 1
# Initial stock of ore
sbar = 10
# Discount 
delta = 0.9
# State Space
S = sp.r_[0:(sbar + 1)]
# Action space
X = sp.r_[0:(sbar + 1)]
n = len(S)
m = len(X)

## Cost of extraction
def cost(s, x):  return x**2 / (1. + s)

## Reward function
f = sp.zeros((n, m))
## I can use indices for states and actions
## Since python indexes on 0
## All states
for i in range(n):
    ## All actions
    for j in range(m):
        ## If enough ore to extract
        if j <= i:
            f[i, j] = price * j - cost(i, j)
        else:
            f[i, j] = -Inf

## Deterministic Transition
g = sp.zeros((n, m))
for i in range(n):
    for j in range(m):
        snext = sp.nonzero(S == i - j)[0]
        if len(snext) > 0:
            g[i, j] = snext
            

