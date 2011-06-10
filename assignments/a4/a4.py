#!/usr/bin/env python
import sys
sys.path.append("../..")

import scipy as sp
from scipy import io
from scipy import random

import psc585
from psc585 import ps4

final_data = io.loadmat("FinalData.mat")['data']
    
foo = ps4.FinalModel.from_mat("FinalModel.mat", "FinalData.mat")

random.seed(4403205)

# Initial Conditional Choice probabilities
# Government
Pg = random.uniform(0, 1, (foo.n, foo.k))
Pg /= Pg.sum(1)[:, newaxis]

# Provinces
Pp0 = random.uniform(0, 1, (foo.n, foo.k))
Pp = sp.concatenate([sp.vstack((Pp0[:, i], 1 - Pp0[:, i])).T
                     for i in range(Pp0.shape[1])], 1)

# Initial Parameters
theta0 = random.normal(0, 1, (5, 1))

results = foo.argmax_theta(Pp, Pg)
theta = foo.npl(Pp, Pg, verbose=True)
