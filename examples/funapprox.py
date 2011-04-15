import scipy as sp
from scipy import special
from scipy.special import orthogonal


class Chebyshev(object):
    """ Chebyshev polynomial interpolation

    Parameters
    -------------
    n : int
        number of interpolation nodes
    a : int, optional
        Lower bound of interpolation region
    b : int, optional
        Upper bound of interpolation region

    Notes
    ---------

    In order to allow for rescaling the interpolation region to
    arbitrary intervals [a, b], I found it easier to use linear
    transformations of the shifted Chebyshev polynomials,
    i.e. orthogonal on [0, 1].

    """
    
    def __init__(self, n, a=-1, b=1):
        self.n = n
        self.a = a
        self.b = b
        self.d = len(self.n.shape)
        self.basis = [ a + (b - a) * special.sh_chebyt(x)
                       for x in range(self.n)]
        self.x = self.nodes()
        self.phi = sp.zeros((n, n))
        for i in range(self.n):
            for j in range(self.n):
                self.phi[i, j] = self.basis[j](self.x[i])

    def funfitf(self, f):
        vec_f = sp.vectorize(f)
        y = vec_f(self.x)
        c = solve(self.phi, y)
        return c

    def funapprox(self, f):
        c = self.funfitf(f)
        basis = self.basis
        def fhat(x):
            return sum([ cj * phi(x) for phi, cj in zip(basis, c)])
        return fhat
        
    def nodes(self):
        """Chebyshev nodes"""
        return self.a + (self.b - self.a) * orthogonal.ts_roots(self.n)[0]
        

def f(x):
    return sp.exp(-x)


