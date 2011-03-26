"""Code written for PSC 585: Dynamic and Computational Models, Spring 2011"""
import itertools

import scipy as sp
from scipy import random
from scipy import linalg as la
from scipy import sparse
from scipy.sparse import linalg as spla
from scipy.special import orthogonal

def eigs(A, n=6):
    """Return the n largest eigenvalues of an array"""
    return sp.sort(la.eig(A)[0])[::-1][0:n]

def spectral_gap(A):
    """Spectral gap 
    
    The spectral gap is the difference between the two
    largest eigenvalues of matrix.
    
    """
    ev = eigs(A, 2)
    return (ev[0] - ev[1]).real

def dfs(P, order=None):
    """ Depth first search
    
    Parameters
    ------------

    P : array, shape (n, n)
        Stochastic transition matrix

    order : array, shape (n, ), optional
        Order in which to search the nodes.

    Returns
    ----------
    F : array, shape (n, )
        Time of first visit
    L : array, shape (n, )
        Time of last visit
    G : array, shape (n, n)
        Graph of minimum spanning tree. Entries have a
        value of 1 if there is an edge between i and j, and 0 if
        there is not.
        
    """

    n = P.shape[0]
    F = sp.zeros(n)
    L = sp.zeros(n)
    G = sp.zeros((n, n))
    T = 0

    if not order:
        order = range(n)

    # P, n in scope
    def visit(i, F, L, G, T):
        """ Visit children of each node
        """
        T += 1
        F[i] = T
        for j in range(n):
            if P[i, j] > 0 and F[j] < 1:
                G[i, j] = 1
                F, L, G, T = visit(j, F, L, G, T)

        T += 1
        L[i] = T
        return (F, L, G, T)
    
    for i in order:
        if F[i] < 1:
            F, L, G, T = visit(i, F, L, G, T)
    return (F, L, G)


def kosaraju(P):
    """ Kosaraju's Algorithm for Strongly Connected Components
    
    Arguments
    ----------
    
    P : array, shape (n, n)
        Must be bool or integer.

    Returns
    ------------

    list : list
      Each element of the list is a component of the graph. Each
      component is a list of length two.  The first element in the
      component is a list of the states in that component.  The second
      element in the component is a `boolean` indicating whether the
      component is an ergodic set.

    Notes
    -----------
    
    The typical Kosaraju algorithm is modified to return the ergodic
    sets and transient set of a Markov chain transition matrix.
    
    """

    # shape is the dimension of P
    n = P.shape[0]

    def visit(i, F, E):
        """ depth-first-search
        
        Parameters
        -----------
        
        i : int
           State index.
        F : list
           First visit time
        E : list
           Last visit time

        Returns
        ------------

        F : list
           First visit times
        E : list
           Last visit times

        P and n are contained in the enclosing environment.
        """
        # Mark i as visited
        F[i] = 1

        # For all states
        for j in range(n):
            # if edge from i to j
            # and j has not been visited
            if P[i, j] and not F[j]:
                ## Add j to list of nodes in component
                E[-1][0].append(j)
                F, E = visit(j, F, E)
        return (F, E)

    ## P.T is transpose of P
    L = dfs(P.T)[1]
    ## Get order of L (yes, it's kind of round about)
    order = sorted(range(L.shape[0]),
                   key=lambda i: L[i],
                   reverse=True)
    ## List of length 0 where all elements are 0
    F = [0] * n
    ## a python list like a matlab cell
    E = []
    # I don't use a counter. Instead I append to the list
    # and always work with the last element in the list.
    for i in order:
        if not F[i]:
            # add new component to E with root i
            E.append([[i], False])
            # Find descendents of i
            F, E = visit(i, F, E)
            # states not in current component
            not_in_comp = [j for j in range(n) if j not in E[-1][0]]
            # If all edges to states outside component
            # are 0, then it is a strongly connected component
            if not_in_comp:
                outside_edges = sp.any(P[sp.ix_(E[-1][0], not_in_comp)] > 0)
                if not outside_edges:
                    E[-1][1] = True
            else:
                E[-1][1] = True
    return E

def eyen(n, i=None):
    """1-D array of length `n` with zeros except in index `i`"""
    if not i:
        i = n - 1
    e = sp.zeros(n)
    e[i] = 1
    return e

def invariant_direct_solver(P):
    """ Calculate Invariant Distribution by direct methods

    Parameters
    -----------
    P : ndarray, shape (n, n)
        Transition matrix of a discrete Markov Chain

    Returns
    ---------
    x : ndarray, shape(n, )
        Invariant distribution of the Markov Chain

    Notes
    --------

    The invariant distribution :math:`\\pi` of a Markov Chain P satisfies
    
    .. math::

       \\pi P = \\pi \\iff \\pi(I - P) = 0

    Since the matrix I - P is singular, instead solve the linear system

    .. math::

       \\pi Q = e_n

    where

    .. math::
    
       Q =
       \\begin{pmatrix}
         1 - p_{11} & - p_{12} & \\cdots & -p_{1(n-1)} & 1 \\\\
         - p_{21} & 1 - p_{22} & \\cdots & -p_{2(n-1)} & 1 \\\\
         \\vdots & \\vdots & \\ddots  & \\vdots & \\vdots \\\\
         -p_{n1} & -p_{n2} & \\cdots & -p_{n(n-1)} & 1 \\\\
       \\end{pmatrix}

    and

    .. math ::

       e_n =
       \\begin{pmatrix}
       0 & 0 & \\cdots & 0 & 1
       \\end{pmatrix}


    """
    n = P.shape[0]
    Q = sp.eye(n) - P
    Q[ : , -1 ] = 1
    return la.solve(Q.T, eyen(n, n - 1))

def tvnorm(x, y=None):
    """ Total variation norm

    Parameters
    -----------
    x : ndarray, shape (N, )
        Vector
    y : ndarray, shape (N, ), optional
        Vector. If given calculates the norm of `x` - `y`

    Returns
    ------------
    z : float
        Total variation norm

    Notes
    ------------

    The total variation norm of two distributions :math:`x, y \\in \\Delta(S)` is

    .. math::

       \\| x - y \\|_{TV} = \sum_{i = 1}^{n} | x_i - y_i | 


    """
    if y is None:
        return la.norm(x)
    else:
        return la.norm(x - y, 1)

def power_iteration(P, x=None, tol=10e-16, T=1000):
    """ Solve for Invariant Distribution of a Markov Chain by Power Iteration

    Parameters
    ------------
    P : ndarray, shape (N, N)
        Transition matrix of a discrete Markov Chain
    x : ndarray, shape (N, )
        Initial guess for the invariant distribution
    tol : float, optional
        Convergence tolerance
    T : int, optional
        Maximum number of iterations

    Returns
    ---------
    x : ndarray, shape (N, )
        Invariant distribution
    t : int
        Number of iterations
    eps : float
        Final residual error 

    Notes
    ---------

    Solves for the invariant distribution of a Markov Chain using the
    iterative scheme

    .. math::

       \\pi^T_t = P^T \\pi_{t-1}^T = (\\pi_{t-1} P)^T

    """
    t = 0
    eps = tol + 1
    n = P.shape[0]
    if not x:
        x = sp.ones(n) / n
    while t < T and eps > tol:
        ## dot() is matrix multiplication
        x1 = sp.dot(P.T, x)
        eps = tvnorm(x1, x)
        x = x1
        t += 1
    return (x, t, eps)

def multinomial(u, pvals):
    """Draw from multinomial

    Parameters
    ----------------
    u : float
        Number in 0, 1 interval
    pvals : (k, ) ndarray
        Probability mass function of a discrete distribution

    Returns
    ----------------
    y : int

    """
    cdf = sp.cumsum(pvals)
    if u >= 1:
        y = (cdf >= 1).nonzero()[0].min()        
    else:
        y = (u <= sp.cumsum(pvals)).nonzero()[0].min()
    return y

def cftp(P, T=1, u=None):
    """Single draw with coupling from the past

    Parameters
    --------------
    P : (n, n) ndarray
         Transition matrix
    T : int
        Number of iterations
    u : (t <= T, ) ndarray
        Draws from a uniform distribution

    Returns
    -----------
    X : int
        State
    T : int
        Number of iterations until convergence

    """
    if u is not None:
        m = len(u)
        ## If u is shorter than T then add new u
        if T - m > 0:
            u = sp.concatenate((random.rand(T - m), u))
        ## If u is too long, then remove extra entries
        elif T - m < 0:
            u = u[:T]
        ## Otherwise T = m
        else:
            pass
    ## u is not defined, add new u
    else:
        u = random.rand(T)
        m = 0

    ## Define state space from P
    S = sp.r_[0:P.shape[0]]
    
    ## P is defined
    ## Define map from P matrix
    def phi(S, u):
        return sp.unique(sp.array([multinomial(u, P[i, ])
                                   for i in S]))

    ## Initialize X_T = S
    X = S
    ## I number them 0 to T because I append
    ## new u's to the front.
    for t in range(0, T):
        X = phi(X, u[t])

    # If more than one state in X, call cftp again
    # Double the length of the chain
    if len(X) > 1:
        X, T = cftp(P, T*2, u)

    return X, T

def cftp_sample(P, n=1):
    """Sample with coupling from the past

    Parameters
    ------------
    P : (N, N) ndarray
        Transition matrix
    n : int, optional
        Number of draws from the distribution

    Returns
    -----------
    X : (n, ) ndarray
        Draws from the invariant distribution of P

    Notes
    -----------
    
    Calls `cftp` `n` times to get `n` draws from the
    invariant distribution of `P` using the coupling
    from the past algorithm.

    """
    return sp.concatenate([cftp(P)[0] for x in range(0, n)])

class lookahead(object):
    """Look-ahead estimator of an invariant distribution

    Parameters
    ------------
    f : function
        Transition function for a continuous state Markov Chain
    fsample: function
        Function to draw directly from transition function given the current state.
    init : (m, ) ndarray
        Initial values for each chain. The number of chains to run is
        determined by the dimensions of this array.
    T : int
        Number of iterations to run each chain.

    Attributes
    ------------
    chains : ndarray, shape (m, d)
        Sample of points from running `m` chains for `T` iterations.

    """
    def __init__(self, f, fsample, init, T):
        self.f = f
        self.fsample = fsample
        self.T = T
        m = init.shape[0]
        chains = init.copy()
        for t in range(T):
            for i in range(m):
                x = fsample(chains[i, ])
                print(t, i, x)
                chains[i, ] =  x
        self.chains = chains

    def pdf(self, s):
        """pdf of invariant distribution

        Parameters
        -------------
        s : array, shape (d, )
            Point at which to calculate the density of the invariant distribution.

        Returns
        ---------
        p : float
            Probability density of the invariant distribution of `f` at `point`.

        """
        chains = [self.f(s, i) for i in self.chains]
        p = sp.array(chains).mean()
        return p

def newton_cotes(n, a=0., b=1.):
    """1-dimensional Newton-Cotes Quadrature Midpoint Rule

    Parameters
    ------------
    n : int
        Number of quadrature points.
    a : float
        Lower bound of interval to integrate over.
    b : float
        Upper bound of interval to integrate over.

    Returns
    ----------
    x : (n, ) ndarray
        Nodes
    w : (n, ) ndarray
        Weights for each node

    Quadrature points and weights derived using Newton-Cotes with
    the midpoint rule formula.  See Judd, p. 252-253.

    """
    ## Width of each interval
    h = (b - a) / n
    ## midpoints of each interval
    x = a + (sp.r_[0:n] + 0.5) * h
    ## weights of each interval
    w = sp.ones(n) / n
    return (x, w)

def newton_cotes_d(n, a=None, b=None):
    """d-dimensional Newton-Cotes Quadrature

    Parameters
    --------------
    n: (d, ) ndarray
        Number of quadrature nodes in each dimension
    a: (d, ) ndarray
        Lower bound in each dimension
    b: (d, ) ndarray
        Upper bound in each dimension

    Returns
    ------------
    x : (n.prod(), d) ndarray
        Quadrature nodes
    w : (n.prod(), ) ndarray
        Quadrature weights

    Notes
    ------------
    
    Generates quadrature nodes and weights in d-dimensions using the
    Newton-Cotes midpoint rule.

    """
    ## Generate tensor product of nodes and weights
    d = len(n)
    if a is None:
        a = sp.zeros(d)
    if b is None:
        b = sp.ones(d)
    x, w = zip(*[newton_cotes(ni, ai, bi) for ni, ai, bi in zip(n, a, b)])
    # Weights 
    w = sp.array([sp.array(y).prod() for y in itertools.product(*w)])
    ## X to Cartesian product of points
    x = sp.array([y for y in itertools.product(*x)])
    return x, w


def to_discrete(f, n, qw=newton_cotes):
    """Discretize Continuous Markov Chain

    Parameters
    -------------
    f: function
        Transition function.
    n: array, shape (d, )
        Number of discrete states.
    qw: function
        Function to use to calculate discrete states.

    Returns
    -------------
    P : array, (n.prod(), n.prod()) 
        Discrete Markov chain transition matrix. Where
        p_ij in the matrix means p(s_j | s_i).
    x : array, shape (n.prod(), d)
        quadrature nodes
    w : array, shape (n.prod(), )
        quadrature weights

    """

    ## Nodes and weights from a quadrature method
    x, w = qw(n)
    # Create empty matrix with zeros.
    m = x.shape[0]
    P = sp.zeros((m, m))
    for i in range(m):
        for j in range(m):
            si = x[i, ]
            sj = x[j, ]
            pij = f(sj, si) * w[j]
            P[i, j] = pij
    # normalize rows to get a stochastic matrix
    P = normalize_rows(P)
    return P, x, w


def normalize_rows(a):
    """Normalize array so rows sum to 1"""
    ## Sum by row
    b = a.copy()
    axis_sum = b.sum(1)
    for i in range(b.shape[0]):
        b[i, : ] = b[i, : ] / axis_sum[i]
    return b


def normalize_cols(a):
    """Normalize array so columns sum to 1"""

    ## Sum by row
    ## TODO: this function is crap. There must be a faster
    ## or more general way to do this.
    # If I don't use copy, then a would be changed in place.
    b = a.copy()
    axis_sum = b.sum(0)
    for i in range(b.shape[1]):
        b[:, i ] = b[ : , i ] / axis_sum[i]
    return b


def invariant_integral(f, n, qw=newton_cotes):
    """ Solve Integral equation by quadrature approximation

    Parameters
    ------------
    f : function 
        State transition function 
    n : int
        Number of quadrature nodes
    qw : function, optional
        Function to calculate the location and weights of the quadrature nodes


    Returns
    --------
    g : (n, ) ndarray
        Solutions to integral equation
    x : (n, ) ndarray
        Quadrature nodes 
    w : (n, ) ndarray
        Quadrature weights

    Notes
    --------
    
    Solves for g(s) where

    .. math::

        g(s') = \int g(s) f(s'|s) ds

    with the system of linear equations 

    .. math::

        g(s_i) = \sum_{j=1}^{n} \\frac{g(s_j) f(s_i | s_j) \omega_j}{w(s_j)}, i=1, \dots, n

    and

    .. math::

        \sum_{j=1}^{n} \\frac{g(s_j) \omega_j}{w(s_j)} = 1

    """
    ## Nodes and weights from a quadrature method
    x, w = qw(n)
    # Create empty matrix with zero
    # Fill in rows 1 to n - 1 
    P = sp.zeros((n, n))
    for i in range(n):
        for j in range(n):
            si = x[i]
            sj = x[j]
            pij = f(si, sj) * w[j]
            P[i, j] = pij
    ## If I don't normalize things are funky
        ## P = normalize_cols(P)
    P = sp.eye(n) - P
    P[ n - 1, : ] = w
    g = la.solve(P, eyen(n))
    return (g, x, w)

def qnwcheb1(n, a, b):
    """ Univariate Gauss-Chebyshev quadrature nodes and weights

    Parameters
    -----------
    n : int
        number of nodes
    a : float
        left endpoint
    b : float
        right endpoint

    Returns
    ---------
    x : array, shape (n,)
        nodes
    x : array, shape (n,)
        weights

    Notes
    ---------
    
    Port of the qnwcheb1 function in the compecon matlab toolbox.
    """
    x = ((b + a) / 2 - (b - a) / 2
         * sp.cos(sp.pi / n * sp.arange(0.5, n + 0.5, 1)))
    w2 =  sp.r_[1, -2. / (sp.r_[1:(n - 1):2] * sp.r_[3:(n + 1):2])]
    w1 = (sp.cos(sp.pi / n * sp.mat((sp.r_[0:n] + 0.5)).T *
                 sp.mat((sp.r_[0:n:2]))).A)
    w0 = (b - a) / n
    w = w0 * sp.dot(w1, w2)
    return x, w

def qnwnorm1(n):
    """Gauss-Hermite normal quadrature nodes and weights in 1 dimension

    Parameters
    ------------
    n : int
        Number of quadrature nodes

    Returns
    ----------
    x : (n, ) ndarray
        Quadrature nodes
    w : (n, ) ndarray
        Quadrature weights
    
    """
    x, w = orthogonal.he_roots(n)
    ## normalize weights to sum to 1
    w /= sum(w)
    return (x.real, w)

def qnwnorm(n, mu=None, var=None):
    """ Compute nodes and weights for multivariate normal distribution

    Parameters
    --------------
    n : (d, ) ndarray
        Array of the number of quadrature nodes for each dimension.
    mu : (d, ) ndarray, optional
        Distribution mean
    var : (d, d) ndarray, optional
        Distribution covariance matrix

    Returns
    --------------
    x : (n.prod(), d) ndarray
        Quadrature nodes.
    x : (n.prod(), ) ndarray
        Quadrature weights

    Notes
    ---------
    Port of `qnwnorm` function in the compecon matlab toolbox.
    """
    d = len(n)
    if mu is None:
        mu = sp.zeros(d)
    if var is None:
        var = sp.eye(d)
    ## Generate x and w for each dimension 
    x, w = zip(*[qnwnorm1(ni) for ni in n])
    ## Generate tensor product of nodes and weights
    # Weights to product
    w = sp.array([sp.array(y).prod() for y in itertools.product(*w)])
    ## x to Cartesian product of points
    x = sp.array([y for y in itertools.product(*x)])
    # scipy cholesky decomposition is opposite triangle of matlab
    x = sp.dot(x, la.cholesky(var).T) + mu
    return (x.real, w.real)

def int2binary(x, width=32):
    """Convert integer to binary array
    
    Parameters
    -----------
    x : int
    width : int
        Width of binary representation.
    
    Returns
    ----------
    y : (width, ) ndarray
         Vector of boolean values for binary representation.
    
    """
    conv = sp.power(2, sp.arange(1 - width, 1))
    y = sp.fmod(sp.floor(x * conv), 2).astype(bool)
    return y
  
def binary2int(x):
    """Convert binary array to integer 

    Parameters
    -----------
    x : ndarray
        Array with the binary representation of a number

    Returns
    ------------
    y : int
        Decimal integer
    """
    conv = sp.power(2, sp.arange(len(x) - 1, -1, -1))
    return (x.astype(bool) * conv).sum()


def ilu0_factor(a):
    """Incomplete LU Factorization

    Parameters
    -----------
    a: array, shape (M, M)
       Matrix to decompose

    Returns
    -----------
    lu : array, shape(M, M)
       Matrix containing U in its upper triangle and L in its lower triangle.

    Notes
    ------------
    
    Uses the ILU(0) algorithm. Algorithm 2.3 in PSC 585 class notes.n

    There exist other preconditioners in scipy.linalg and scipy.linalg.sparse
    but I could not find which one corresponded to ilu(0).

    This is too slow to be useful.
    """
    if any(a.diagonal() == 0) :
        print("Diagonal must be non-zero for zero-level factorization to work")
        return None
    n = a.shape[0]
    lu = sparse.lil_matrix((n, n))
    for i, j, v in itertools.izip(a.row, a.col, a.data):
        print (i, j, v)
        k = min(i, j) - 1
        s = v
        if k > 0:
            s = v - (lu[i, 0:k] * lu[0:k, j]).sum()
        if (i >= j):
            lu[i, j] = s
        else: # i < j
            lu[i, j] = s / lu[i, i]
    return lu


def sparse_power_iteration(P, x, tol=10e-16, maxiter=200):
    """Preconditioned power iteration for a sparse stochastic matrix

    Paramters
    ---------------
    P : array, shape (n, n), sparse
        transition matrix of a Markov Chain
    x : array, shape (n, )
        On entry, the initial guess. On exit, the final solution.

    """
    t = 0
    eps = tol + 1
    n = P.shape[0]
    # ILU factorization 
    LU = ilu0_factor(P)
    L = sparse.tril(LU)
    U = sparse.triu(LU)
    # New matrix Q
    Q = P.copy()
    Q.setdiag(1 - Q.diagonal())
    Q *= -1
    Q = Q.T
    info = -1
    t = -1
    for t in range(maxiter):
        ## dot() is matrix multiplication
        dx = spla.spsolve(U, spla.spsolve(L, Q.matvec(x)))
        x -= dx
        relres = tvnorm(dx)
        if relres < tol:
            info = 0
            break
    t += 1
    return (info, t, relres)


def gjacobi(A, b, x,  maxit=1000, tol=10e-12, normalizer=None):
    """ Gauss-Jacobi iterative linear solver for sparse matrices
    
    Parameters
    ------------
    A : sparse matrix, shape (n, n)
        Left hand side of linear system.
    b : array, array (n, )
        Right hand side of linear system.
    x : array, array (n, )
        On entry, `x` holds the initial guess. On exit `x` holds the final solution.
    tol : float
        Requested error tolerance for convergence.
    maxit :
        Maximum number of iterations.

    Returns
    -----------
    info : int
        Exit status. 0 if converged. -1 if it did not.
    iter : int
        Number of iterations
    relres : float
        total variance norm of the final solution.

    Notes
    -----------
    Code based on gjacobi in the compecon Matlab toolbox.
    
    """
    d = A.diagonal()
    info = -1
    for iter in range(maxit):
        print iter
        dx = (b - A.dot(x)) / d
        x += dx
        if normalizer:
            normalizer(x)
        relres = tvnorm(dx)
        print relres, tol
        if relres < tol:
            info = 0
            break
    iter += 1
    return (info, iter, relres)


def gseidel(A, b, x, maxit=1000, tol=10e-13, relax=1., normalizer=None):
    """ Gauss-Jacobi iterative linear solver for sparse matrices

    Parameters
    ------------
    A : sparse matrix, shape (n, n)
        Left hand side of linear system.
    b : array, array (n, )
        Right hand side of linear system.
    x : array, array (n, )
        On entry, `x` holds the initial guess. On exit `x` holds the final solution.
    tol : float, optional
        Requested error tolerance for convergence.
    maxit : int, optional 
        Maximum number of iterations.
    relax : float, optional
        Relaxation parameter. Default is 1 in Gauss-Seidel. Set
        to values of less than or greater to 1 for under or over relaxation.

    Returns
    -----------
    info : int
        Exit status. 0 if converged. -1 if it did not.
    iter : int
        Number of iterations
    relres : float
        total variance norm of the final solution.

    See Also
    ---------
    gjacobi, sparse_power_iteration


    Notes
    --------

    Code based on gseidel in the compecon Matlab toolbox.

    """
    Q = sparse.tril(A).tocsr()
    info = -1
    for iter in range(maxit):
        print iter
        dx = spla.spsolve(Q, b - A.dot(x))
        x += dx * relax
        if normalizer:
            normalizer(x)
        relres = tvnorm(dx)
        print relres, tol
        if relres < tol:
            info = 0
            break
    iter += 1
    return (info, iter, relres)

# def GTH(P):
#     """ Grassman-Taskar-H? Invariant Distribution

#     Parameters
#     -------------
#     P : ndarray, shape (n, n)
#         Stochastic matrix representing an ergodic Markov Chain.

#     Returns
#     --------------
#     p : ndarray, shape (n,)
#         Invariant distribution of `P`.

#     """

#     n = P.shape[0]
#     ## Storage Matrix
#     PA = sp.zeros((n, n))
#     ## Matrix reduction
#     PA[ :-1, -1] = P[:-1, -1] / P[-1, :-1].sum()
#     for i in range(n):
#         print(i)
#         # The Shape of P is now (n - i - 1, n - i - 1)
#         P = (P[:-(i + 1), :-(i + 1)]
#              + (asarray(mat(P)[ :-(i + 1), -(i + 1)]
#                 * mat(P)[ -(i + 1), :-(i + 1)]))
#              * (1 / P[-(i + 1), :-(i + 1)].sum()))
#         print(P)
#         ## TODO: not working
#         PA[:-(i + 2), -(i + 2)] = (P[ :-(i + 1), -(i + 1)] /
#                       P[ -(i + 1), :-(i+1)].sum())
#         print(PA)
    

class TestKosaraju(object):
    """ Test algorithms against known results

    Use on command line as py.test dfs.py
    """

    P1 = sp.array([[ 0. , 1. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ],
                   [ 0. , 0. , 1. , 0. , 0. , 0. , 0. , 0. , 0. ],
                   [ 0.5 , 0. , 0. , 0.5 , 0. , 0. , 0. , 0. , 0. ],
                   [ 0. , 0. , 0. , 0. , 1. , 0. , 0. , 0. , 0. ],
                   [ 1. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ],
                   [ 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 1. ],
                   [ 0. , 0. , 0. , 0. , 0. , 0. , 0.5 , 0.5 , 0. ],
                   [ 0. , 0. , 0. , 0. , 0. , 0. , 2./3 , 0. , 1./3 ],
                   [ 0. , 0. , 0. , 0. , 0. , 1. , 0. , 0. , 0. ]])
    P2 = sp.array([[ 0.2 ,  0.1 ,  0.05,  0.  ,  0.  ,  0.3 ,  0.25,  0.1 ,  0.  ],
                   [ 0.35,  0.  ,  0.  ,  0.2 ,  0.4 ,  0.05,  0.  ,  0.  ,  0.  ],
                   [ 0.  ,  0.  ,  0.  ,  0.4 ,  0.05,  0.2 ,  0.1 ,  0.  ,  0.25],
                   [ 0.  ,  0.  ,  0.  ,  0.  ,  0.2 ,  0.2 ,  0.2 ,  0.2 ,  0.2 ],
                   [ 0.2 ,  0.2 ,  0.2 ,  0.  ,  0.2 ,  0.2 ,  0.  ,  0.  ,  0.  ],
                   [ 0.1 ,  0.2 ,  0.1 ,  0.1 ,  0.1 ,  0.1 ,  0.1 ,  0.1 ,  0.1 ],
                   [ 0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.4 ,  0.1 ,  0.  ,  0.5 ],
                   [ 0.1 ,  0.  ,  0.  ,  0.  ,  0.4 ,  0.1 ,  0.  ,  0.  ,  0.4 ],
                   [ 0.05,  0.  ,  0.  ,  0.4 ,  0.1 ,  0.05,  0.  ,  0.4 ,  0.  ]])
    
    def test_P1(self):
        """ Test results against matrix P1"""
        E1 = [[[5, 8], True], [[7, 6], False], [[0, 1, 2, 3, 4], True]]
        assert kosaraju(self.P1) == E1

    def test_P2(self):
        """ Test results against matrix P2"""
        E2 = [[[0, 1, 3, 4, 2, 5, 6, 8, 7], True]]
        assert kosaraju(self.P2) == E2



