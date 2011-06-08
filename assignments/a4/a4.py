#!/usr/bin/env python
import sys
sys.path.append("../..")

import scipy as sp
from scipy import io

import psc585
from psc585 import ps4

final_data = io.loadmat("FinalData.mat")['data']
    
foo = ps4.FinalModel.from_mat("FinalModel.mat", "FinalData.mat")
print foo.model()

# Initial Conditional Choice probabilities
Pg = sp.ones((foo.n, foo.k))
Pg /= Pg.sum(1)[:, newaxis]

Pp = sp.ones((foo.n, 2 * foo.k)) * 0.5

theta = sp.zeros((5, 1))

## Y_d matrix
## Actions of provinces in the data
## (T * k ) x 1 matrix
Yd = foo.data[:, 1:-1].ravel("F")


foo.new_p(Pp, Pg, theta)
foo.phigprov(Pp, Pg, theta)
P = foo.ptilde(Pp, Pg)
Pi = foo.ptilde_i(Pp, Pg, 0, 1)

