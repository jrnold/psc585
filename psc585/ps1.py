""" Problem Set 1: Markov Chains"""
import scipy as sp
from scipy import special
from scipy import sparse
from psc585 import *

def p2a_f(sprime, s):
    """Problem 2.a, function f

    Parameters
    -----------
    sprime : float
        State transitioning to. Between 0 and 1.
    s : float
        State transitioning from. Between 0 and 1.

    Returns
    ----------
    p : float
        Transition probability.

    Notes
    ----------

    Continuous Markov chain on the unit interval :math:`[0, 1]`, 

    .. math::

        f(s' | s) = B(a(s), b(s)) s'^{(a(s) - 1)} (1 - s')^{(b(s) - 1)}


    where :math:`a(s) = 2 + 50 s^{10}`  and :math:`b(s) = 2 + 50 (1 - s)^{10}`

    """
    def p2a_a(s):
        return 2. + 50. * pow(s, 10)

    def p2a_b(s):
        return 2. + 50. * pow(1 - s, 10)

    if sprime >= 0 and sprime <= 1: 
        a = p2a_a(s)
        b = p2a_b(s)
        beta = special.gamma(a + b) / (special.gamma(a) * special.gamma(b))
        y = sp.power(sprime, a - 1) * sp.power(1 - sprime, b - 1) * beta
    else:
        y = 0
    return y


def p2c_f_sample(s, size=1):
    """Sample from transition function in problem 2.c

    Parameters
    ----------
    s : (2,) ndarray 
        Current state
    size : int, optional
        Number of samples to draw

    Returns
    ---------
    sprime : (size, 2) ndarray

    See Also
    ----------
    p2c_f

    """
    alpha1 = 0.6
    alpha2 = 0.3
    mean = sp.array([alpha1 * s[0], alpha2 * (s[0] + s[1])])
    cov = sp.eye(2)
    sprime = random.multivariate_normal(mean, cov, size)
    return sprime

  
def p2c_f(sprime, s):
    """ Problem 2.c, function f

    Parameters
    ------------
    sprime : float.
        State transitioning to
    s : float
        State transitioning from

    Returns
    ---------
    p : float
        Transition probability

    Notes
    ---------

     .. math::

         (s' |s) = N((\\alpha_1 s_1 , \\alpha_2 (s_1 + s_2 )), I_2)

     where :math:`\\alpha_1 = 0.3` and :math:`\\alpha_2 = 0.6`.

    """
    alpha1 = 0.6
    alpha2 = 0.3
    mean = sp.array([alpha1 * s[0], alpha1 * s.sum()])
    ## Since independent can simply multiply pdfs
    p1 = stats.norm.pdf(sprime[0], loc=mean[0])
    p2 = stats.norm.pdf(sprime[1], loc=mean[1])
    p = p1 * p2
    return p

def quadratic_utility(x, y):
    """Quadratic utility function

    Notes
    ----------

    .. math::

       u(x, y) = - \sum_{i=1}^n (x_i - y_i)^2

    """
    u = - sp.square((x - y)).sum()
    return u

class Legislature(object):
    """Legislature in problem 3

    Parameters
    -----------
    x : (m, d) ndarray
         Ideal points of legislators
    ufunc : function, optional
         Function to calculate the utility of a policy relative to an ideal point

    Notes
    ---------
    
    This class contains a set of legislators with d-dimensional ideal points,
    and a utility function.  It includes methods to compare two policies given
    the legislators' ideal points. 
    
    """
    def __init__(self, x, ufunc=quadratic_utility):
        self.x = x
        self.ufunc = ufunc

    def votes(self, p, q):
        """Votes of legislature on p versus status quo q"""
        yeas = sp.array([ self.ufunc(p, i) >= self.ufunc(q, i) for i in self.x]).sum()
        return yeas

    def yeas(self, p, q):
        """Number of yeas on p compared to status quo q"""
        return self.votes(p, q).sum()

    def majpref(self, p, q):
        """Is policy p majority preferred to q"""
        majority = floor(self.x.shape[0] / 2) + 1
        ispref = self.yeas(p, q) >= majority
        return ispref

    def transition(self, S):
        """Return a transition matrix for a Markov Chain

        Parameters
        --------------
        S : (n, d) ndarray
          n points on a d-dimensional policy space

        Returns
        -------------
        P : (n, n) ndarray
           Transition matrix. There is an equal probability of
           transitioning to any point


        Notes
        --------------
        
        Right now this is VERY slow because this function
        finds the majority preferred policies by checking every
        other policy and returning a dense matrix.  There is
        probably a better way to implement this.

        """
        ## TOO SLOW! 
        ## Initialize matrix
        n = S.shape[0]
        P = sp.empty((n, n))
        ## For each state in state-space
        for i in range(n):
             q = S[i, ]
             preferred = sp.array([self.majpref(p, q) for p in S]).astype(float)
             preferred /= preferred.sum()
             P[i, ] = preferred
        return P

def _setdiff(x, y):
    print x.tolist()
    print y.tolist()
    print set(x.tolist()) - set(y.tolist())
    return sp.array(list(set(x.tolist()) - set(y.tolist())))

class Provinces(object):
    """Provinces in problem 4

    Parameters
    ------------
    D : (k, k) ndarray
        Matrix of distances between provinces

    Attributes
    -------------
    D : (k, k) ndarray
        Matrix of distances between provinces
    k : int
        Number of provinces
    dmax : int
        Maximum distance between provinces.
    maxstate : int
        Maximum state. States are numbered 0 to :math:`2^k`.
    P : (2^k, 2^k) sparse array
        Transition matrix of the Markov chain of revolt probabilities
        for each province.

    """

    def __init__(self, D):
        self.D = D
        self.k = D.shape[0]
        self.dmax = D.max()
        self.maxstate = 2**self.k
        self._mat()

    def _p(self, s):
        """Transition probability function

        Parameters
        -----------
        s : int
            Integer representing the province

        Returns
        -----------
        sprime : list
           Each element is a tuple with the transition state
           and the probability of transitioning to that state.
           
        """

        # I enumerate states such that
        # 0 = [0, ...., 0]
        # 1 = [1, 0, 0, 0, ... 0 ]
        # 2 = [0, 1, 0, 0, ... 0 ]
        # 2^k = [0, 0, ..., 0, 1]
        # 2^(k+1) - 1 = [1, 1, ..., 1, 1]
        sbinary = int2binary(s, width=self.k)[::-1]
        # indices of Revolting provinces
        R = sbinary.nonzero()[0].tolist()

        # indices of Non-revolting provinces
        C = sp.logical_not(sbinary).nonzero()[0]
        P = [1]
        newstate = [s]
        for i, si in enumerate(sbinary):
            ## Calculate
            Ci = list(set(C) - set([i]))
            if Ci:
                dminc = self.D[sp.ix_([i], Ci)].min()
            else:
                dminc = self.dmax

            Ri = list(set(R) - set([i]))
            if Ri: 
                dminr = sp.c_[self.D[sp.ix_([i], Ri)]].min()
            else:
                dminr = self.dmax
            # if i in revolt
            if si:
                Si = s - 2**i
                Pi = dminr / dminc
            # if i not in revolt
            else:
                Si = s + 2**i
                Pi = dminc / dminr
            P.append(Pi)
            newstate.append(Si)
        newstate = sp.array(newstate)
        P = sp.array(P)
        P /= P.sum()
        return (P.tolist(), newstate.tolist())

    def _mat(self):
        """ transition matrix

        Returns
        ------------
        None

        Notes
        ------------
        Sets `self.P`

        """
        n = 2**self.k
        self.P = sparse.lil_matrix((n, n))
        for i in range(self.maxstate):
            pi = self._p(i)
            for sj, j in zip(*pi):
                self.P[i, j] = sj
            
