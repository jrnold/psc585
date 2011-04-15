""" Problem Set 2 """

## Probabilistic transition function
def p1p(sprime, a):
    """Probabilistic transition function in Assignment 2, Problem 1

    Parameters
    -----------
    sprime : int
      State transitioning to 
    a : int
      Action

    Returns
    ----------
    p : float
      Probability of transitioning to state `sprime` given action `a`.

    Notes
    -----------

    Stochastic transition function from problem 1.
    
    """
    def K(a):
        """"""
        return sp.array([sp.exp(-0.5 * sp.dot(sp.dot((x - a), V), (x - a)))
                         for x in X]).sum()
    V = sp.array([[15, 0], [0, 15]])
    p = sp.exp(-0.5 * sp.dot(sp.dot((sprime - a), V), (sprime - a))) / K(a)
    return p
