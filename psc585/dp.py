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
    transprob: array, shape (n, m)
       Stochastic transition matrix
    T: int, optional
       Number of time periods. Set to `None` for
       infinite horizon problems.
    vterm: optional
       Terminal value function for finite horizon problems.

    """
    
    def __init__(self, discount, reward, P, T=None, vterm=None):
        self.discount = discount
        self.reward = reward
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

        Parameters
        -----------
        x : array or integer
            policy (an action for all states)

        Notes
        ---------

        `x` is (n, T+1) for finite horizon problems, and (n, 1) for infinite
        horizon problems.
        """
        ## Select indices of policy from reward function
        ## Not sure if this works with
        ## ddpsolve.m calculates the index value
        fstar = self.reward[sp.r_[0:self.], x]

    def backsolve(self):
        """ Solve Bellman equation via backward recursion"""
        x = sp.zeros(self.n, self.T)
        v = sp.concatenate(sp.zeros(self.n, self.T), 1)
        pstar = zeros(self.n, self.n, sp.T)
        for t in sp.arange(self.T, -1, -1):
            print t
            v, x = self.valmax(v[ : , t])
            v[ :, t] = v
            x[ :, t] = x
            ## TODO add pstar
        return (x, v)


