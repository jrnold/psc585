import scipy as sp
from scipy import special
from scipy.special import orthogonal

import numpy as np

def cartesian(arrays, out=None):
    """
    Generate a cartesian product of input arrays.

    Parameters
    ----------
    arrays : list of array-like
        1-D arrays to form the cartesian product of.
    out : ndarray
        Array to place the cartesian product in.

    Returns
    -------
    out : ndarray
        2-D array of shape (M, len(arrays)) containing cartesian products
        formed of input arrays.

    Examples
    --------
    >>> cartesian(([1, 2, 3], [4, 5], [6, 7]))
    array([[1, 4, 6],
           [1, 4, 7],
           [1, 5, 6],
           [1, 5, 7],
           [2, 4, 6],
           [2, 4, 7],
           [2, 5, 6],
           [2, 5, 7],
           [3, 4, 6],
           [3, 4, 7],
           [3, 5, 6],
           [3, 5, 7]])

    """

    arrays = [np.asarray(x) for x in arrays]
    dtype = arrays[0].dtype

    n = np.prod([x.size for x in arrays])
    if out is None:
        out = np.zeros([n, len(arrays)], dtype=dtype)

    m = n / arrays[0].size
    out[:,0] = np.repeat(arrays[0], m)
    if arrays[1:]:
        cartesian(arrays[1:], out=out[0:m,1:])
        for j in xrange(1, arrays[0].size):
            out[j*m:(j+1)*m,1:] = out[0:m,1:]
    return out

class Chebyshev(object):
    """ Chebyshev polynomial interpolation

    Parameters
    -------------
    n : ndarray, shape (n, )
        number of interpolation nodes
    a : array, shape (n, )
        Lower bound of interpolation region
    b : int, shape (n, )
        Upper bound of interpolation region

    Notes
    ---------

    Currently this only supports one dimension interpolation.  I need
    to expand this to d-dimensions with tensor products.

    In order to allow for rescaling the interpolation region to
    arbitrary intervals [a, b], I found it easier to use linear
    transformations of the shifted Chebyshev polynomials,
    i.e. orthogonal on [0, 1].

    """
    
    def __init__(self, n, a, b):
        self.d = n.shape[0]
        self.n = n
        self.a = a
        self.b = b
        ## Basis is on [0,1] so values 
        ## need to be transformed first
        ## Basis is a list of lists
        self.basis = [special.sh_chebyt(x)
                      for x in range(self.n)]
        self.x = self.funnode()

    def funbas(self, x):
        """Return basis matrix evaluated at x """
        n = self.n
        m = len(x)
        # rows = points
        # cols = basis polynomials
        phi = sp.zeros((m, n))
        for i in range(m):
            for j in range(n):
                ## transform x to from [a, b] to 0, 1
                phi[i, j] = self.basis[j](self._trans(self.x[i]))
        return phi

    def funfitf(self, f):
        """Approximate function at standard nodes"""
        vec_f = sp.vectorize(f)
        y = vec_f(self.x)
        c = self.funfitxy(self.x, y)
        return c

    def funfitxy(self, x, y):
        """ Fit function with arbitrary x and y"""
        phi = self.funbas(x)
        if len(x) == len(y):
            c = solve(phi, y)
        else:
            print("least squares not implemented")
        ## Else solve with least squares
        return c

    def funeval(self, c, x):
        """Evaluate function """
        def _funeval(x):
            return sum([ cj * phi(self._trans(x))
                         for phi, cj in zip(self.basis, c)])
        return vectorize(_funeval)(x)

    def _trans(self, x):
        """Transform [a, b] to [0, 1]"""
        return (x - self.a) / (self.b - self.a)

    def _invtr(self, x):
        """Transform from [0, 1] to [a, b]"""
        return self.a + (self.b - self.a) * x

    def funapprox(self, f):
        """Polynomial approximation of a function"""
        c = self.funfitf(f)
        basis = self.basis
        def fhat(x):
            return sum([ cj * phi(self._trans(x))
                         for phi, cj in zip(basis, c)])
        return sp.vectorize(fhat)
        
    def funnode(self):
        """Chebyshev nodes"""
        return self.a + (self.b - self.a) * orthogonal.ts_roots(self.n)[0]
        

def f(x):
    return sp.exp(-x)

def g(x):
    return abs(x)

foo = Chebyshev(sp.array([5]), sp.array([1]), sp.array([2]))
fhat = foo.funapprox(f)
print fhat(sp.linspace(1, 2, 10))
print f(sp.linspace(1, 2, 10))

foo = Chebyshev(sp.array([5]), sp.array([-1]), sp.array([2]))
ghat = foo.funapprox(g)
space = sp.linspace(-1, 1, 20)
print ghat(space)
print g(space)

