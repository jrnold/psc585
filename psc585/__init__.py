"""Code for PSC 585"""

########
def timethis(f, *args, **kwargs):
    """
    Parameters
    ------------
    f : function
        Function to time.

    Returns
    ------------
    t : float
        Execution time, results from time.clock()
    ret : 
        Return value of f

    Notes
    ------

    The module timeit is generally preferred for timing coding
    fragments, but it does not return the results. 
    """ 
    t0 = time.clock()
    res = f(*args, **kwargs)
    t = time.clock() - t0
    return (t, res)


