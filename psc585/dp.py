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
    """ Discrete Time, Discrete Choice Dynamic Programming Problems
    
    Attributes
    ------------
    discount: float
       Discount factor. Must be on (0, 1) for infinite horizon problems, 
       or (0, 1] for finite horizon problems.
    reward: array, shape (n, m)
       Reward values
    P: array, shape (n, m)
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
        self.P = P
        self.T = T
        self.n, self.m = self.reward.shape
        self.vterm = vterm
        if self.T and vterm is None:
            self.vterm = sp.zeros(self.n)

    def setReward(self, reward):
        self.reward = reward
        self.n, self.m = self.reward.shape

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
        U = self.reward + sp.reshape(self.discount * self.P.dot(v), (self.n, self.m))
        print U
        ## argmax by row
        x = U.argmax(1)
        ## maximum values for these actions
        v = U[sp.r_[0:self.n], x]
        print v, x
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
        fstar = self.reward[sp.r_[0:self.T], x]

    def backsolve(self):
        """Solve finite time model by Backward Recursion

        Returns
        ----------
        X : array, shape (n, T)
            Optimal controls. An optimal policy for each starting state
        V : array, shape (n, T + 1)
            Value function.             

        """
        X = sp.zeros((self.n, self.T))
        V = sp.column_stack((sp.zeros((self.n, self.T)), self.vterm))
        # pstar = sp.zeros(self.n, self.n, sp.T)
        for t in sp.arange(self.T - 1, -1, -1):
            v, x = self.valmax(V[ : , t + 1])
            V[ :, t] = v
            X[ :, t] = x
            ## TODO add pstar
        return (X, V)


