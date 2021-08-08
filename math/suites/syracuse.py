"""
Etude de la suite de Syracuse, recherche de la plus
grande duree de vol possible
"""

from __future__ import annotations

import numpy as np
import numba as nb
from matplotlib import pyplot as plt


@nb.vectorize(target="cpu")
def syracuse(n: int) -> int:
    if n % 2 == 0:
        return n // 2

    return 3 * n + 1


@nb.vectorize(target="cpu")
def duree_vol(n: int) -> int:
    count = 0
    while n != 1:
        n = syracuse(n)
        count += 1

    return count


n = 1_000_000
nums = np.linspace(1, n, n, dtype=np.uint64)
duree_vols = duree_vol(nums)

y_max = duree_vols.max()
x_max = np.where(duree_vols == y_max)[0][0] + 1

fig, ax = plt.subplots()

ax.annotate(
    f"Plus grand temps de vol \n x={x_max}, y={y_max}",
    xy=(x_max, y_max),
    xytext=(x_max, y_max),
    arrowprops=dict(facecolor="black", shrink=0.05),
)

ax.plot(nums, duree_vols)
plt.show()
