# Market price
price = 1
# Initial stock of ore
sbar = 10
# Discount 
delta = 0.9
# State Space
S = sp.r_[0:(sbar + 1)]
# Action space
X = sp.r_[0:(sbar + 1)]
n = len(S)
m = len(X)

## Cost of extraction
def cost(s, x):  return x**2 / (1. + s)

## Reward function
f = sp.zeros((n, m))
## I can use indices for states and actions
## Since python indexes on 0
## All states
for i in range(n):
    ## All actions
    for j in range(m):
        ## If enough ore to extract
        if j <= i:
            f[i, j] = price * j - cost(i, j)
        else:
            f[i, j] = -Inf

## Deterministic Transition
g = sp.zeros((n, m))
for i in range(n):
    for j in range(m):
        snext = sp.nonzero(S == i - j)[0]
        if len(snext) > 0:
            g[i, j] = snext
            

model = dp.Ddpsolve.from_transfunc(transfunc=g, reward=f,
                                   discount=delta)
v, x, pstar = model.funcit(sp.zeros(n))

v1, x1, pstar1 = model.newton(sp.zeros(n))

