"""
Comparaison entre la factorielle exacte, la fonction gamma et
la formule de Stirling
"""

import numpy as np
import numba as nb
import pandas as pd

from scipy import special
from matplotlib import pyplot as plt


@nb.vectorize(target="cpu")
def stirling(n: int) -> int:
    return np.sqrt(2 * np.pi * n) * (n / np.e) ** n


X = np.linspace(1, 10, 10, dtype=np.uint64)
Y1 = special.factorial(X, exact=True)
Y2 = special.factorial(X, exact=False)
Y3 = stirling(X)

df = pd.DataFrame(data=dict(exact=Y1, gamma=Y2, stirling=Y3))

fig, ax = plt.subplots()
table = ax.table(cellText=df.values, colLabels=df.columns, loc="center")

plt.show()
