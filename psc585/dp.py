"""Dynamic Programming"""
import scipy as sp
from scipy import linalg as la
from scipy import sparse

EPS = sp.sqrt(sp.finfo(sp.float64).eps)

def eyeminus(x):
    """ 1 - x in place """
    x *= -1
    x[sp.diag_indices(x.shape[0])] = 1 + x.diagonal()

def _expandg(g):
    """ Expand transition function to a matrix
    """
    P = sparse.coo_matrix((sp.ones(sp.prod(g.shape)),
                             (sp.r_[0:sp.prod(g.shape)],
                              g.flatten(1))))
    return P.tocsr()

class Ddpsolve(object):
    """ Discrete Time, Discrete Choice Dynamic Programming Problems
    
    Attributes
    ------------
    discount: float
       Discount factor. Must be on (0, 1) for infinite horizon problems, 
       or (0, 1] for finite horizon problems.
    reward: array, shape (n, m)
       Reward values
    P: array, shape (m * n, n)
       Stochastic transition matrix. The axes correspond to action,
       initial state, and next state.
    T: int, optional
       Number of time periods. Set to `None` for
       infinite horizon problems.
    vterm: optional
       Terminal value function for finite horizon problems.

    """
    
    def __init__(self, discount, reward, P, T=None, vterm=None):
        self.discount = discount
        self.reward = reward
        self.T = T
        self.n, self.m = self.reward.shape
        self.P = P 
        self.vterm = vterm
        if self.T and vterm is None:
            self.vterm = sp.zeros(self.n)

    def setReward(self, reward):
        """Set reward"""
        self.reward = reward
        self.n, self.m = self.reward.shape

    def valmax(self, v):
        """ Solve single Bellman equation

        Returns
        ----------------
        v : array (n, )
            Optimal values
        x : array (n, )
            Optimal controls.
            
        Notes
        ------

        Solve the Bellman equation for a given v

        .. math::
            V(s) = max_{x \in X(x)}
            \left\{
                f(s, x) + \delta \sum_{s' \in S} P(s' | s, x) V(s')
             \right\}            
        """
        U = self.reward + sp.reshape(self.discount * sp.dot(self.P, v),
                                     (self.m, self.n)).T
        ## argmax by row
        x = U.argmax(1)
        ## max by row
        v = U[sp.r_[0:self.n], x]
        return (v, x)

    def valpol(self, x):
        """ Evaluation policy

        Parameters
        -----------
        x : array
            policy (an action for all states)

        Returns
        --------
        pstar : array, shape (n, n)
            Transition probability matrix
        fstar : array, shape (n, )
            Optimal rewards
        ind : array, shape (n,)
            Index of values

        Notes
        ---------

        `x` is (n, T+1) for finite horizon problems, and (n, 1) for infinite
        horizon problems.
        """
        ## Select indices of policy from reward function
        ## Not sure if this works with
        ## ddpsolve.m calculates the index value
        ind = self.n * x + sp.r_[:self.n]
        fstar = self.reward[ sp.r_[0:self.n] , x.astype(int)].copy()
        pstar = self.P[ind, ].copy()
        return pstar, fstar, ind

    def backsolve(self, T=None, vterm=None):
        """Solve finite system by backward recursion

        Parameters
        -------------
        T : int, optional
            Number of periods of time.

        Returns
        ----------
        X : array, shape (n, T)
            Optimal controls. An optimal policy for each starting state
        V : array, shape (n, T + 1)
            Value function.             

        """
        if T is None:
            if self.T is not None:
                T = self.T
            else:
                print ("Not a finite time model")
                return
        if vterm is None and self.vterm is None:
            vterm = sp.zeros(self.n)
        else:
            vterm = self.vterm
        x = sp.zeros((self.n, T), dtype=int)
        v = sp.column_stack((sp.zeros((self.n, T)), vterm))
        pstar = sp.zeros((self.n, self.n, T))
        for t in sp.arange(T - 1, -1, -1):
            v[ :, t] , x[ :, t]  = self.valmax(v[ : , t + 1])
            pstar[..., t] = self.valpol(x[:, t])[0]
        return (x, v, pstar)

    def funcit(self, v=None, maxit=100, tol=EPS, error_bounds=True):
        """ Solve Bellman equations by function iteration

        Parameters
        --------------
        v : array, shape (n, ), optional
           Initial guess
        maxit : int, optional
           Maximum number of iterations
        tol : float, optional
           Convergence tolerance
        error_bounds : bool, optional
           Use error bounds to determine convergence.

        Returns
        ------------
        info : int
            Exit status. 0 if converged. -1 if not.
        t : int
            Number of iterations
        relres : float
            Residual variance
        v : array, shape (n, )
        x : array, shape 
        pstar : array, shape
           
        """
        if v is None:
            v = sp.zeros(self.n)
        info = -1
        delta = (self.discount) / (1 - self.discount)
        t = 0
        relres = tol + 1
        for it in range(maxit):
            t += 1
            vold = v.copy()
            v, x = self.valmax(vold)
            if error_bounds:
                lbound = delta * (v - vold).min()
                ubound = delta * (v - vold).max()
                relres = (ubound - lbound)
                if relres < tol:
                    v += (ubound + lbound) / 2
                    info = 0
                    break
            else:
                relres = la.norm(v - vold)
                if relres < tol:
                    info = 0
                    break
        pstar = self.valpol(x)[0]
        return (info, t, relres, v, x, pstar)

    def newton(self, v=None, maxit=100, tol=EPS, verbose=False,
               gauss_seidel=False):
        """Solve Bellman equations via Newton method (policy iteration)

        Parameters
        --------------
        v : array, shape (n, ), optional
           Initial guess for values.
        x : array, shape (n, ), optional
           Initial guess for policy.
        maxit : int, optional
           Maximum number of iterations
        tol : float, optional
           Convergence tolerance
        gauss_seidel : bool, optional
           Use Gauss-Seidel to solve the linear equation.

        Returns
        ------------
        info : int
            Exit status. 0 if converged. -1 if not.
        t : int
            Number of iterations
        relres : float
            Residual variance
        v : array, shape (n, )
        x : array, shape 
        pstar : array, shape

        Notes
        --------

        Also called policy iteration.

        """
        if v is None:
            v = sp.zeros(self.n)
        ## Set initial values of x to such
        x = sp.zeros(self.n) 
        info = -1
        t = 0
        for it in range(maxit):
            t += 1
            xold = x.copy()
            v, x = self.valmax(v)
            pstar, fstar, ind = self.valpol(x)
            Q = pstar * self.discount
            eyeminus(Q)
            if not gauss_seidel:
                vold = v.copy()
                v = la.solve(Q, fstar)
                relres = la.norm(v - vold)
            else:
                ## Gauss Seidel
                L = sp.tril(Q)
                dv = la.solve(L, fstar - sp.dot(Q, v))
                relres = la.norm(dv)
                v += dv
            if verbose:
                print("%d, %f" % (it, relres))
            if sp.all(x == xold):
                info = 0
                break
        return (info, t, relres, v, x, pstar)

    @classmethod
    def from_transfunc(cls, transfunc, **kwargs):
        """Initialize with deterministic transition function

        Parameters
        -------------
        transfunc : array (n, m)
             Deterministic transition function. Axes are
             state, action.
        """
        kwargs['P'] = sp.asarray(_expandg(transfunc).todense())
        return cls(**kwargs)        

    @classmethod
    def from_transprob(cls, transprob, **kwargs):
        """Initialize with transaction probabilities

        Parameters
        -----------
        transprob : array, shape (m, n, n)
                  Stochastic transition matrix. The axes correspond to action,
                  initial state, and next state.

        """
        m = transprob.shape[0]
        n = transprob.shape[1]
        kwargs['P'] = sp.reshape(transprob, (m * n, n))
        return cls(**kwargs)

    
def ddpsimul(pstar, s, N, x):
    """ Monte-Carlo simulation of discrete-state/action controlled Markov process

    Parameters
    -------------
    pstar : array, shape (n, n) or (n, n, T)
      Optimal state transition matrix. Usually returned by one of the methods of
      `Dpsolve`. The array has shape (n, n) for infinite horizon processes,
      and (n, n, T) for finite horizon processes.
    s : array, shape (k, )
      Initial states
    N : int
      Number of simulations
    x : array, shape (n, ) or (n, T)
      Optimal controls

    Returns
    ---------
    spath : array, shape (k, N + 1)
       Simulated states
    
    """
    infinite = (len(pstar.shape) == 2)
    n = pstar.shape[1]
    k = len(s)
    spath = sp.zeros((k, N+1), int)
    if infinite:
        ## Row cumulative sum
        cp = pstar.cumsum(1)
        spath[:, 0] = s
        for t in range(1, N + 1):
            ## Draws the column from a categorical distribution
            rdraw = random.rand(k, 1)
            s = (sp.repeat(rdraw, n, 1) > cp[s, ]).sum(1)
            spath[:, t] = s
    else:
        T = pstar.shape[2]
        if N > T:
            print("Simulations greater than the time horizon are ignored.")
        N = min(N, T)
        spath[:, 0] = s
        for t in range(N + 1):
            cp = pstar[...,t].cumsum(1)
            rdraw = random.rand(k, 1)
            s = (sp.repeat(rdraw, n, 1) > cp[s, ]).sum(1)

    xpath = x[spath]
    return (spath, xpath)

