"""Functions to solve nonlinear equations and complementarity problems"""

import scipy as sp
from scipy import linalg as la
from scipy import sparse

def fixpoint(f, x, tol=None, maxit=100, **kwargs):
    """ Fixed point function iteration

    Parameters
    -------------
    f : function
      Function for which to find the fixed point.
    x : array 
      Initial guess.
    tol : float, optional
      Tolerance for convergence.
    maxit : int, optional
      Maximum number of iterations.

    Returns
    -----------
    info : integer
       Did function converge? -1 if it did not, and 0 if it did.
    relres : float
       Relative residual at last iteration.
    t : int
       Number of iterations
    gval : array
       Fixed point, where f(x) = x.
       
    """
    if tol is None:
        tol = sqrt(sp.finfo(float).eps)
    info = -1
    t = 0
    for it in range(maxit):
        t += 1
        gval = f(x, **kwargs)
        relres = la.norm(gval - x)
        if relres < tol:
            info = 0
            break
        x = gval
    return (info, relres, t, gval)



def bisect(f, a, b, tol=1e-4, **kwargs):
    """Find roots by bisection

    Parameters
    -------------
    f : function
    a : array
      Lower bounds
    b : array
      Upper bounds
    tol : float
      Tolerance criteria

    Returns
    -------------
    x : array
      Roots of f

    """

    if sp.any(a > b):
        print("Lower bound greater than upper bound")
        return 
    sa = sp.sign(f(a, **kwargs))
    sb = sp.sign(f(b, **kwargs))
    if sp.any(sa == sb):
        print("Root not bracketed")
        return
    ## Initializations
    dx = 0.5 * (b - a)
    tol = dx * tol
    x = a + dx
    dx = sb * dx

    while sp.any(sp.absolute(dx) > tol):
        print dx
        dx *= 0.5
        x -= sp.sign(f(x, **kwargs)) * dx

    return x

def smooth(f, x, a, b):
    """ Fischer's function
    
    Parameters
    ----------
    f : function
        should return (fx, J) where fx are the function
        values evaluated at x, and J is the Jacobian.
    x : ndarray, shape (n, )
    a : ndarray, shape (n, )
        lower bound
    b : ndarray, shape (n, )
        upper bound

    Returns
    --------
    fxnew : ndarray, shape (n, )
        value of function
    jnew : ndarray, shape (n, )
        Jacobian of function 

    Notes
    --------

    Fischer's function is defined as 
    
    .. math::

       f(x) = \phi^-(\phi^+(f(x), a - x), b - x)

    where

    .. math::

       \phi_i^{\pm}(u, v) = u_i + v_i \pm \sqrt{u_i^2 + v_i^2}

    It is commonly used in rootfinding for complementarity problems.
    It has the same roots as

    .. math::

       f(x) = \min (\max(f(x), a - x), b - x)

    but is smoother, and thus has fewer numerical problems. 
    
    """

    n = x.shape[0]
    if len(a) == 1:
        a = sp.zeros(n) + a
    if len(b) == 1:
        b = sp.zeros(n) + b

    dainf = sp.nonzero(a == -inf)[0]
    dbinf = sp.nonzero(b == inf)[0]
    da = a - x
    db = b - x

    fx, J = f(x)
    sq1 = sp.sqrt(fx ** 2 +  da ** 2)
    pval = fx + da + sq1
    pval[dainf] = pval[dainf]
    sq2 = sp.sqrt(pval ** 2 + db ** 2)
    fxnew = pval + db - sq2
    fxnew[dbinf] = pval[dbinf]

    dpdy = 1 + fx / sq1
    dpdy[dainf] = 1
    dpdz = 1 + da / sq1
    dpdz[dainf] = 0
    dmdy = 1 - pval / sq2
    dmdy[dbinf] = 1
    dmdz = 1 - db / sq2
    dmdz[dbinf] = 0
    ff = dmdy * dpdy          # ff = ds / df
    xx = dmdy * dpdz + dmdz   # xx = -ds / dx
    jnew = sparse.spdiags(ff, 0, n, n).dot(J) - sparse.spdiags(xx, 0, n, n)

    return fxnew, jnew


def minmax(f, x, a, b):
    """ Max-min transformation

    Parameters
    -----------
    f : function
        Returns a tuple of the function value and Jacobian.
    x : ndarray, shape (n, )
    a : ndarray, shape (n, )
    b : ndarray, shape (n, )


    Returns
    -----------
    fhatval : ndarray, shape (n, )
       function values
    fhatjac : ndarray
       Jacobian

    Notes
    -----------

    Function is 

    .. math::

       f(x) = \min (\max(f(x), a - x), b - x)

    This function is used in complementarity problems.

    """
    n = x.shape[0]
    da = a - x
    db = b - x
    fval, fjac = f(x)
    fhatval = sp.minimum(sp.maximum(fval, da), db)
    fhatjac = -1 * sp.eye(n)
    i = sp.nonzero((fval > da) & (fval < db))[0]
    fhatjac[i, :] = fjac[i, :]
    return fhatval, fhatjac

def ncpsolve(f, a, b, x, tol=10e-13, maxsteps=10, maxit=100, usesmooth=True, **kwargs):
    """ Solve nonlinear complementarity problem

    Parameters
    -----------
    f : function
    a : ndarray, shape (n, )
    b : ndarray, shape (n, )
    x : ndarray, shape (n, )
        initial guess
    tol : float
        convergence tolerance
    maxit : int
        maximum number of iterations
    maxsteps : int
        maximum number of backsteps
        

    Returns
    ----------

    x : ndarray, shape (n, )
        solution to ncp
    fval : ndarray, shape (n, )
        function value at x
    
    
    Notes
    ---------

    A nonlinear complementarity problem has two $n$-vectors
    a and b with a < b and a function f : \R^n \to \R^n.
    The CP is to find an n-vector $x \in [a, b]$ that satisfies
    the following constraints

    .. math::
       a \leq x \leq b
       x_i > a_i \to f_i(x) \geq 0 \forall i = 1, \dots, n
       x_i < b_i \to f_i(x) \leq 0 \forall i = 1, \dots, n
        

    """
    if usesmooth:
        _smooth = smooth
    else:
        _smooth = minmax
    n = x.shape[0]
    for i in range(maxit):
        fval, fjac = f(x, **kwargs)
        print i, fval
        ftmp, fjac = _smooth(f, x, a, b)
        ## infinity norm
        dx = - (la.solve(fjac, ftmp))
        fnorm = la.norm(ftmp, sp.inf)
        if fnorm < tol:
            break
        fnormold = sp.inf
        for backsteps in range(maxsteps):
            xnew = x + dx
            fnew = f(xnew, **kwargs)
            fnew = _smooth(f, xnew, a, b)[0]
            fnormnew = la.norm(fnew, inf)
            if fnormnew < fnorm:
                break
            if fnormold < fnormnew:
                dx *= 2
                break
            fnormold = fnormnew
            dx /= 2
            print(backsteps)
        ## No backstepping
        x += dx

    return x, fval

