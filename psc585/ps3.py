""" Problem Set 3 specific code"""

import numpy as np
import scipy as sp
from scipy import linalg as la

class BargModel(object):
    """ Bargaining model

    Attributes
    ----------
    
    c : int
       Coalition specific payoff (constant)
    p : ndarray, shape (n, )
       Proposal probabilities
    ideals : ndarray, shape (n, 2)
       Ideal points
    d : float
       Discount parameter
    Cs1 :
       Final coalition specific payoffs
    Cs0 :
       Initial coalition specific payoffs
    prop : ndarray, shape (30, )
    part1 : ndarray, shape (30, )
    part2 : ndarray, snape (30, )
    parts : ndarray, shape (60, )
       Uknown
    pols : ndarray, shape (60, 1)
       Policies
    COAL : ndarray, shape (60, 5)
       Coalitions
    K : float
       Constant term in the utility function of players

    Notes
    ---------

    *COAL* is a array with shape (60, 5). Each row is
    a participant in a coalition. Rows 0 and 1 are
    the two participants in coalition $C_{1} \in D_1$,
    rows 2 and 3 correspond to the participants in coalition
    $C_2 \in D_1$, and so on.

    *parts* is an (60, ) array with the number of each voter of the
    corresponding row in *COAL*, i.e. the column number + 1 which is
    nonzero.  *parts1* contains the odd rows, i.e.  the first
    participant in each of the 30 coalitions. *parts2* list the
    participants in the even rows of *COAL*, i.e.  the second
    participant in each of the 30 coalitions.
    

    """

    def __init__(self, ideals, d, p, Cs1, c, K):
        """
        Parameters
        -----------
        Ideals : ndarray, shape (n, 2)
        d : float
            Discounting parameter
        p : ndarray, shape (n, )
            Proposal distribution for each legislator
        Cs1 :
            Something
        c : int
            Coalition specific payoff
        K : float
            Constant term in the utility function of players


        Notes
        -------------

        Function translated from the BargModel.m file provided by
        Tasos.


        """
        self.c = c
        self.p = p
        self.ideals = ideals
        self.d = d
        self.Cs1 = Cs1

        parts = np.zeros((60, ), int)
        pols = np.zeros((60, ), int)
        ## Coalitions
        COAL = np.zeros((60, 5), int)
        
        j = 0
        ## For all voters
        for i in np.arange(1, 6):
            m = 0
            ## For voters 1 to 4
            for j1 in np.arange(1, 5):
                ## that are not the initial voter
                if j1 != i:
                    ## for voters greater than the 1st other
                    for j2 in np.arange(j1 + 1, 6):
                        if j2 != i:
                            m = m + 1
                            j = j + 1
                            parts[int(j) - 1] = j1
                            COAL[int(j) - 1, int(j1) - 1] = 1
                            pols[int(j) - 1 ] = (i - 1) * 6 + m
                            
                            j = j + 1
                            parts[int(j) - 1 ] = j2
                            COAL[int(j) - 1, int(j2) - 1] = 1
                            pols[int(j) - 1] = (i - 1) * 6 + m

        parts -= 1
        self.prop = np.repeat(np.arange(0, 5), 6)
        self.part1 = parts[sp.r_[0:60:2]]
        self.part2 = parts[sp.r_[0:60:2] + 1]
        self.parts = parts
        self.pols = pols - 1
        self.COAL = COAL
        self.K = K
                ## Initial Values 
        ## Set all entries to c / (1 - p_j) by default
        Cs0 = sp.ones((30, 5)) * self.c / (1 - self.p)
        ## Fill in entries when 
        for i in sp.arange(0, 5):
            for j in sp.arange(0, 6):
                if i == j:
                    Cs0[i * 6 + j, i] = 0
                else:
                    Cs0[i * 6 + j, i] = - self.c
        self.Cs0 = Cs0


    def func(self, y, t=1):
        """ Function values

        Parameters
        --------------
        y : ndarray, shape (155, )
            Parameter vector
        t : float, optional
            Value of t between 0 and 1

        Returns
        ----------
        F : ndarray, shape (155, )
            Function values at y
        JF : ndarray, shape (155, 155)
            Jacobian of the function evaluated at y
        JFt : ndarray
            Derivative of the function as an implicit function of t
        
        """
        ## Coalition specific payoffs
        Cs = t * self.Cs1 + (1 - t) * self.Cs0

        # number of parameters
        npars = {'v' : 5, 'lam' : 30, 'x' : 60, 'mu' : 60 }
        # Indices for each parameter
        idxs = {}
        idxs['v'] = sp.arange(0, npars['v'])
        idxs['lam'] = sp.arange(idxs['v'].max() + 1,
                                idxs['v'].max() + 1 + npars['lam'])
        idxs['x'] = sp.arange(idxs['lam'].max() + 1,
                              idxs['lam'].max() + 1 + npars['x'])
        idxs['mu'] = sp.arange(idxs['x'].max() + 1,
                              idxs['x'].max() + 1 + npars['mu'])

        ## Split parameters 
        v = y[idxs['v']]
        lam = y[idxs['lam']]
        x = y[idxs['x']]
        mu = y[idxs['mu']]

        ## Function values
        F = sp.zeros(155, )
        ## Lambda
        sig = sp.maximum(0, lam) ** 2
        ## Mu values as a matrix
        ## TODO: check that rows and values are correct
        ## Matlab reshapes by Fortran (column order)
        mmu = sp.reshape(sp.maximum(0, mu), (30, 2), order='C')
        ## Lambda as a matrix
        mlam = sp.maximum(0, -lam) ** 2
        ## proposals to a 30 x 2 matrix
        X = sp.reshape(x, (30, 2), order='C')

        ## utilities for proposal 
        U = sp.zeros((30, 5))
        for i in range(U.shape[0]):
            u1 = - (X[i, 0] - self.ideals[:, 0])**2
            u2 = - (X[i, 1] - self.ideals[:, 1])**2
            U[i, : ] = u1 + u2 + Cs[i, :] + self.K

        ## \bar{u}.
        ## Utility to each player for status quo of (0,0)
        Usq = (-(self.ideals[self.parts, 0])**2
               - (self.ideals[self.parts, 1])**2
               + self.K)
        ## Derivatives of utilities
        DU1 = sp.zeros((30, 5))
        for i in range(DU1.shape[0]):
            DU1[i, ] = - 2 * (X[i, 0] - self.ideals[:, 0])
        DU2 = sp.zeros((30, 5))
        for i in range(DU2.shape[0]):
            DU2[i, ] = - 2 * (X[i, 1] - self.ideals[:, 1])
        ## Equation 7
        F[idxs['v']] = v - (U.T.dot(sig * sp.kron(self.p, sp.ones(6))))

        ## Equation 8
        ## For all players
        for i in range(5):
            ## Indices of player i
            ii = i * 6 + sp.arange(0, 6)
            u = sig[ii].T.dot(U[ii, i]) - U[ii, i] - mlam[ii]
            F[idxs['lam'].min() + ii] = u

        ## Equation 9
        Fx1 = (DU1[sp.r_[0:30], self.prop]
              + DU1[sp.r_[0:30], self.part1] * (mmu[:, 0] ** 2)
              + DU1[sp.r_[0:30], self.part2] * (mmu[:, 1] ** 2))
        Fx2 = (DU2[sp.r_[0:30], self.prop]
              + DU2[sp.r_[0:30], self.part1] * (mmu[:, 0] ** 2)
              + DU2[sp.r_[0:30], self.part2] * (mmu[:, 1] ** 2))
        F[idxs['x']] = sp.reshape(sp.column_stack((Fx1, Fx2)), (60, 1))

        # Equation 10
        F[idxs['mu']] = (U[self.pols, self.parts] - (1 - self.d) * Usq
                         - self.d * v[self.parts] - sp.maximum(0, -mu)**2)

        # Jacobian
        JF = sp.zeros((155, 155))
        # Equation 7
        # with respect to v
        JF[sp.r_[:5], sp.r_[:5]] = sp.ones(5)
        # with respect to lambda
        JF[sp.ix_(idxs['v'], idxs['lam'])] = \
        (-U * (2 * sp.maximum(0, lam) * self.p.repeat(6))[: , sp.newaxis]).T
        # with respect to x
        JF[:5, 35:95:2] = -(DU1 * (sig * self.p.repeat(6))[:, sp.newaxis]).T
        JF[:5, 36:96:2] = -(DU2 * (sig * self.p.repeat(6))[:, sp.newaxis]).T

        # Equation 8
        # with respect to lambda
        for i in range(5):
            ii = i * 6 + sp.r_[0:6]
            foo = (sp.tile(2 * sp.maximum(0, lam[ii]) * U[ii, i], (6, 1)) + 
                   sp.eye(6) * (2 * sp.maximum(0, -lam[ii])))
            JF[sp.ix_(npars['v'] + ii, npars['v'] + ii)] = foo
        # with respect to x
        for i in range(5):
            for m in range(6):
                minlam = idxs['lam'].min()
                minx = idxs['x'].min()
                # range of lambda pars for player i
                ii = i * 6 + sp.r_[0:6]
                # Indices
                JFi0 = minlam + (i * 6) + m
                JFi10 = minx + i * 12 + sp.r_[0:12:2]
                JFi11 = minx + i * 12 + sp.r_[0:12:2] + 1
                #
                JF[JFi0, JFi10] = DU1[ii, i] * sig[ii]
                JF[JFi0, JFi11] = DU2[ii, i] * sig[ii]
                JF[JFi0, minx + i * 12 + m * 2] -= DU1[i * 6 + m, i]
                JF[JFi0, minx + i * 12 + m * 2 + 1] -= DU2[i * 6 + m, i]

        # Equation 9
        # with respect to x
        JF[idxs['x'], idxs['x']] = -2 * (sp.ones(60) + sp.kron((mmu ** 2), sp.ones((2, 1))).sum(1))
        # with respect to mu
        JF[sp.r_[35:95:2], sp.r_[95:155:2]] = 2 * DU1[sp.r_[0:30], self.part1] * mmu[:, 0]
        JF[sp.r_[35:95:2] + 1, sp.r_[95:155:2]] = 2 * DU2[sp.r_[0:30], self.part1] * mmu[:, 0]
        JF[sp.r_[35:95:2], sp.r_[95:155:2] + 1] = 2 * DU1[sp.r_[0:30], self.part2] * mmu[:, 1]
        JF[sp.r_[35:95:2] + 1, sp.r_[95:155:2] + 1] = 2 * DU2[sp.r_[0:30], self.part2] * mmu[:, 1]

        # Equation 10
        # with respect to v
        JF[sp.ix_(idxs['mu'], idxs['v'])] = -self.d * self.COAL
        # with respect to x
        JF[sp.r_[95:155:2], sp.r_[35:95:2]] = DU1[sp.r_[0:30], self.part1]
        JF[sp.r_[95:155:2], sp.r_[35:95:2] + 1] = DU2[sp.r_[0:30], self.part1]
        JF[sp.r_[95:155:2] + 1, sp.r_[35:95:2]] = DU1[sp.r_[0:30], self.part2]
        JF[sp.r_[95:155:2] + 1, sp.r_[35:95:2] + 1] = DU2[sp.r_[0:30], self.part2]
        # with respect to mu
        JF[idxs['mu'], idxs['mu']] = 2 * sp.maximum(0, -mu)

        # Derivatives of H(y, y(t))
        JFt = sp.zeros(155, )
        JFt[idxs['v']] =  -(self.Cs1 - self.Cs0).T.dot(sig * self.p.repeat(6))

        for i in range(5):
            ii = i * 6 + sp.arange(0, 6)
            cv = self.Cs1[ii, i] - self.Cs0[ii, i]
            JFt[idxs['lam'].min() + ii] = sig[ii].T.dot(cv) - cv
        jft1 = self.d * (self.Cs1[sp.arange(0, 30), self.part1]
                          - self.Cs0[sp.arange(0, 30), self.part1])
        jft2 = self.d * (self.Cs1[sp.arange(0, 30), self.part2]
                          - self.Cs0[sp.arange(0, 30), self.part2])
        JFt[idxs['mu']] = sp.reshape(sp.column_stack((jft1, jft2)), (60, 1))

        return (F, JF, JFt)

    def newton(self, y, t=1, tol=None, maxit=100, verbose=False):
        """ Newton iteration

        Parameters
        ------------
        y : ndarray, shape (155, )
           Initial parameter guess. Changed in place.
        tol : float
           Tolerance for convergence
        maxit : int
           Maximum number of iterations
        verbose : bool
            Whether to print messages during iterations

        Returns
        -------
        info: integer
           Did function converge? True if it did.
        relres : float
           Relative residual of last iteration
        i : int
           Number of iterations


        """
        if tol is None:
            tol = sp.sqrt(sp.finfo(float).eps)
        info = False
        for i in range(maxit):
            f, df, dft = self.func(y, t)
            dy = la.solve(df, f)
            y -= dy
            relres = la.norm(dy)
            if verbose:
                print("%d: %f" % (i, relres))
            if relres  < tol:
                info = True
                break
        return (info, relres, i)

    def predcorr(self, y, glb, gub, t = 0, verbose=False):
        """ Predictor-corrector method

        Parameters
        -----------
        y : ndarray, shape (155, )
            Initial parameters. Updated in place.
        glb : float
            Gamma-step lower bound, between 0 and 1.
        gub : float
            Gamma-step upper bound, between 0 and 1.
        t : float, optional
            t between 0 and 1.
        verbose : bool
            Whether to print messages during iterations

        Returns
        ----------
        t : float
            Final t value. If less than 1, then it did not converge.
        i : int
            Number of iterations

        """
        gamma = gub
        i = 0
        while t < 1:
            i += 1
            t_old = t
            y_old = y.copy()
            # Update t
            t = min(1, t + gamma)
            if verbose:
                print("%d: %f %f" % (i, t, gamma))
            # Update y(t) in direction of t
            y += gamma * self.func(y, t)[2]
            success = False
            try:
                # y is updated in place
                nwtn = self.newton(y, t)
                success = nwtn[0]
                if not success:
                    print "Newton did not converge"
            except la.LinAlgError:
                print "LinAlgError"
                pass
            # If newton works
            if success:
                print ("success")
                gamma = min(gub, 2 * gamma)
            # If newton fails
            else:
                print ("fail")
                # reset y
                y = y_old
                # reset t
                t = t_old
                # halve gamma
                gamma = gamma / 2.
                # if gamma too small, end iterations
                if gamma < glb:
                    print "Warning: Did not converge"
                    print "Gamma less than lower bound"
                    break
        return (t, i)
            
def make_model(Cs1):
    """ Make model in assignment 3"""
    ideals = sp.array([[-1./4, -1./4],
                       [-1./4, 3./4],
                       [7./16, 7./16],
                       [7./16, -3./16],
                       [-1./3, -1./2]])
    
    # Discount
    d = 2./3
    # Proposal probability
    p = sp.array([1./5] * 5)
    # Coalition specific utility (constant)
    c = 45.
    # Utility constant
    K = 10
    model = BargModel.BargModel(points, d, p, Cs, c, K)
    return model


def make_y0(model):
    """ Make y0 """
    def mu_ij(i, j):
        return -sp.sqrt(uij[j, i] + (model.c / (1 - model.p[j]))
                        - (1 - model.d) * ubar[j]
                        - model.d * v0[j])

    # \bar{u} : status quo payoffs
    ubar = -(model.ideals ** 2).sum(1) + model.K
    # TODO: where did plus 10 come from?
    uij = (-(model.ideals[:, 0] - model.ideals[:, 0][:, sp.newaxis])**2 +
           -(model.ideals[:, 1] - model.ideals[:, 1][:, sp.newaxis])**2 + model.K)
    # v_0
    v0 = (uij * model.p[:, sp.newaxis]).sum(1) + model.c
        ## \lambda_0
    lam0 = sp.ones((5, 6)) * -sp.sqrt(model.c)
    # if m_i = i
    lam0[sp.r_[0:5], sp.r_[0:5]] = 1
    lam0 = reshape(lam0, (lam0.size, ))
    # x_0
    x0 = sp.reshape(sp.repeat(model.ideals, 6, axis=0), (60, ))
    # \mu_0
    mu0 = sp.zeros((5, 6, 2))
    # For players
    for i in range(5):
        # For coalitions
        for mi in range(6):
            # for each other player in the coalition
            ii = i * 6 + mi
            mu0[i, mi, 0] = mu_ij(i, model.part1[ii])
            mu0[i, mi, 1] = mu_ij(i, model.part2[ii])
    mu0 = sp.ravel(mu0)
    # y_0
    y0 = sp.concatenate((v0, lam0, x0, mu0))
    return y0


    
