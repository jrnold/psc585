#!/usr/bin/env python
import scipy as sp
from scipy import linalg as la
from scipy import io

import psc585.ps3

Cs = io.loadmat("Cs.mat")["Cs"]
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
model = psc585.ps3.BargModel(ideals, d, p, Cs, c, K)

y0 = make_y0(model)
print la.norm(model.func(y0, t=0)[0], inf)

y = y0.copy()
ret = model.predcorr(y, 0.0001, 0.25, verbose=True)
