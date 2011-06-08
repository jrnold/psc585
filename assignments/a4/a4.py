#!/usr/bin/env python
import sys
sys.path.append("../..")

import scipy as sp
from scipy import io

import psc585
from psc585 import ps4

final_data = io.loadmat("FinalData.mat")['data']
    
foo = ps4.FinalModel.from_mat("FinalModel.mat")
print foo.model()

# Initial Conditional Choice probabilities
Pg = sp.ones((foo.n, foo.k))
Pg /= Pg.sum(1)[:, newaxis]

Pp = sp.ones((foo.n, 2 * foo.k)) * 0.5

theta = sp.zeros((5, 1))

foo.new_p(Pp, Pg, theta)
foo.phigprov(Pp, Pg, theta)
foo.ptilde(Pp, Pg)

