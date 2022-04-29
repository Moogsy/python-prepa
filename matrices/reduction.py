import numpy as np 
import sympy as sp

X = sp.Symbol('X')


m = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9],
],
dtype='O')

p = np.add(X * np.identity(3, dtype='O'), -m, casting="unsafe")

print(p.dtype)

from det_naif import det

print(det(p))

