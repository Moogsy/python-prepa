from cmath import exp, pi
from time import perf_counter
from typing import Tuple, List, Literal, Generator

from matplotlib import pyplot as plt

Point = complex
Charge = Tuple[float, Point]

fig, ax = plt.subplots()
ax.set_aspect("equal")


def E(M: Point, charges: List[Charge]) -> Point:
    S = complex(0)
    for Qi, Ri in charges:
        T = Ri - M
        S += Qi * T / abs(T) ** 3

    return S


def ok(M: Point, charges: List[Charge], Xmax: float, Ymax: float, Dmin: float) -> bool:
    if abs(M.real) >= Xmax or abs(M.imag) >= Ymax:
        return False

    for _, Ri in charges:
        if abs(M - Ri) < Dmin:
            return False

    return True


def ligne(
    M: Point,
    charges: List[Charge],
    Xmax: float,
    Ymax: float,
    dt: float,
    Dmin: float,
    signe: Literal[-1, 1],
) -> Generator[tuple[float, float], None, None]:
    Xmax = M.real + 2 * Xmax
    Ymax = M.imag + 2 * Ymax

    while ok(M, charges, Xmax, Ymax, Dmin):
        M += signe * dt * E(M, charges)
        yield M.real, M.imag


def voisinage(M: Point, R: float, n: int) -> Generator[Point, None, None]:
    for k in range(n):
        yield M + R * exp(k * 2j * pi / n)


def lignes(
    charges: List[Charge], n: int, Xmax: float, Ymax: float, dt: float, Dmin: float
):

    debut = perf_counter()

    for i, (Qi, Ri) in enumerate(charges, 1):
        debut_c = perf_counter()

        Rx, Ry = Ri.real, Ri.imag

        if Qi > 0:
            color = "r"
            signe = -1

        else:
            color = "b"
            signe = 1

        txt = plt.Text(
            Rx,
            Ry,
            round(abs(Qi), 2),
            ha="center",
            va="center",
            color="w",
            bbox={"boxstyle": "circle", "color": color},
        )
        ax.add_artist(txt)

        for point_couronne in voisinage(Ri, Dmin, n):
            X, Y = zip(
                (Rx, Ry), *ligne(point_couronne, charges, Xmax, Ymax, dt, Dmin, signe)  # type: ignore
            )
            ax.plot(X, Y, zorder=1)

        fin_c = perf_counter()

        print(f"Charge {i}/{len(charges)}, {(fin_c - debut_c):.3f}s")

    fin = perf_counter()
    print(f"Total: {(fin - debut):.3f}s")
