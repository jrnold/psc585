"""Functions to solve nonlinear equations and complementarity problems"""

import scipy as sp
from scipy import linalg as la

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
    
