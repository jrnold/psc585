#!/usr/bin/env python

import scipy as sp
from scipy import io

final_data = io.loadmat("FinalData.mat")['data']

    
foo = FinalModel.from_mat("FinalModel.mat")
print foo.model()

# Initial Conditional Choice probabilities
Pg = sp.ones((foo.n, foo.k))
Pg /= Pg.sum(1)[:, newaxis]

Pp = sp.ones((foo.n, 2 * foo.k)) * 0.5

theta = sp.zeros((5, 1))

pytave.feval(2, "NewP", Pp, Pg, theta, foo.model())
pytave.feval(1, "Phigprov", Pp, Pg, theta, foo.model())
p
