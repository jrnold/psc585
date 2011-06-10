import sys
sys.path.append("../..")

import unittest

import scipy as sp
from scipy import io

import psc585
from psc585 import ps4

class FinalModelTestCase(unittest.TestCase):

    def setUp(self):
        Pg = sp.ones((foo.n, foo.k))
        Pg /= Pg.sum(1)[:, newaxis]
        Pp = sp.ones((foo.n, 2 * foo.k)) * 0.5
        theta0 = sp.zeros((5, 1))
        theta1 = sp.zeros((5, 1))
        foo = ps4.FinalModel.from_mat("FinalModel.mat", "FinalData.mat")

    def test_newP(self):
        self.foo.new_p(Pp, Pg, theta)

    def test_phigprov(self):
        self.foo.phigprov(Pp, Pg, theta)

    def test_ptilde(self):
        self.foo.ptilde(Pp, Pg)

    def test_ptilde_i(self):
        self.foo.ptilde_i(Pp, Pg, 0, 1)



if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(FinalModelTestCase)

