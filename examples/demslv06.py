from scipy import random

def fslv06(x):
    fval = sp.zeros(2, )
    fjac = sp.zeros((2, 2))
    fval[0] = -sp.exp(-x[0]) + x[1] ** 2
    fval[1] = 2 * x[0] * x[1] ** 2
    fjac[0, 0] = sp.exp(x[0])
    fjac[0, 1] = 2 * x[1]
    fjac[1, 0] = 2 * (x[1] ** 2)
    fjac[1, 1] = 2 * x[0] * x[1]
    return fval, fjac

z = random.uniform(0.0, 10.0, size=(2, 2))
a = z.min(1)
b = z.max(1)
# xinit = (b + a) / 2
xinit = rand(2, )

xinit = sp.array([0.77253, 0.33278])
x = xinit.copy()
a = sp.array([1.4146, 4.4497])
b = sp.array([7.1966, 7.9277])
#foo, bar = ncpsolve(fslv06, a, b, xinit)
foo, bar = ncpsolve(fslv06, a, b, x, usesmooth=False)
foo, bar = ncpsolve(fslv06, a, b, x, usesmooth=False)

def f(x):
    fval = 1.01 - (1 - x) ** 2
    fjac = sp.array([2 * (1 - x)])
    return fval, fjac

# x = sp.array([0])
# a = sp.array([0])
# b = sp.array([inf])
# fxnew, jnew = smooth(f, x, a, b)
# x, fval = ncpsolve(f,  sp.array([0]),  sp.array([inf]), sp.array([0]))
#x, fval = ncpsolve(f,  sp.array([0]),  sp.array([inf]), sp.array([0]), usesmooth=False)

#x, fval = ncpsolve(fslv06, a, b, xinit)
