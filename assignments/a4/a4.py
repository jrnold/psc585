#!/usr/bin/env python
import sys
sys.path.append("../..")

import scipy as sp
from scipy import io

import psc585
from psc585 import ps4

final_data = io.loadmat("FinalData.mat")['data']
    
foo = ps4.FinalModel.from_mat("FinalModel.mat", "FinalData.mat")

# Initial Conditional Choice probabilities
Pg = sp.ones((foo.n, foo.k))
Pg /= Pg.sum(1)[:, newaxis]

Pp = sp.ones((foo.n, 2 * foo.k)) * 0.5

theta = sp.zeros((5, 1))

## Y_d matrix
## Actions of provinces in the data
## (T * k ) x 1 matrix

foo.new_p(Pp, Pg, theta)
foo.phigprov(Pp, Pg, theta)
P = foo.ptilde(Pp, Pg)
Pi = foo.ptilde_i(Pp, Pg, 0, 1)
Eiai = foo.Ei_ai(Pp, 0, 1)
Ei = foo.Ei(Pp, 0)
Z1 = foo.Zia(Pg, 0, 1)
Z0 = foo.Zia(Pg, 0, 0)
Z = foo.Zi(Pg, Pp, 0)






