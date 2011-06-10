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

theta0 = sp.ones((5, 1))

results = foo.argmax_theta(Pp, Pg)
theta = foo.npl(Pp, Pg)
