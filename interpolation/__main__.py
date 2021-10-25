import numpy as np
from numpy import random
from matplotlib import pyplot as plt

x = np.linspace(-np.pi, np.pi, 32)  # type: ignore
y = random.random(x.shape)

fig, ax = plt.subplots()

def d(x, y):
    y2 = np.zeros(y.shape)

    y2[1:] = (y[1] - y[0]) / (x[1] - x[0])
    y2[1:-1] = (y[2:] - y[:-2]) / (x[2:] - x[:-2])
    y2[-1] = (y[-1] - y[-2]) / (x[-1] - x[-2])

    return y2

def A(X):
    out = []

    for x1, x2 in zip(X, X[1:]):
        a_i = np.array([
            [x1 * x1 * x1, x1 * x1, x1, 1],
            [x2 * x2 * x2, x2 * x2, x2, 1],
            [3  * x1 * x1, 2  * x1, 1,  0],
            [3  * x2 * x2, 2  * x2, 1,  0],
        ])

        out.append(a_i)

    return out

def B(X, Y):
    out = []
    dX = d(X, Y)

    for y1, y2, m1, m2 in zip(Y, Y[1:], dX, dX[1:]):
        b_i = np.array([y1, y2, m1, m2])
        out.append(b_i)

    return out


def poly_factory(x, y):
    a = A(x)
    b = B(x, y)

    out = []

    for a_i, b_i in zip(a, b):
        x = np.linalg.solve(a_i, b_i)

        out.append(x)

    return out

def P(coeff, x): 
    a, b, c, d = coeff
    return a * x**3 + b * x**2 + c * x + d

def traceInterpol(ax, x, y): 
    polys = poly_factory(x, y)

    for poly, p1, p2 in zip(polys, x, x[1:]):
        
        lin = np.linspace(p1, p2, 10)
        ax.plot(lin, P(poly, lin))


        
traceInterpol(ax, x, y)
ax.scatter(x, y)

plt.show()
