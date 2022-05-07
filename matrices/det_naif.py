import numpy as np


def extraction(A: np.ndarray, i: int, j: int) -> np.ndarray:
    n, p = A.shape
    B = np.zeros((n - 1, p - 1))

    line_A = 0
    line_B = 0

    while line_A < n:
        if line_A != i:
            B[line_B, :j] = A[line_A, :j]
            B[line_B, j:] = A[line_A, j+1:]

            line_B += 1

        line_A += 1

    return B


def det(A: np.ndarray) -> complex:
    """
    Calcul naif d'un determinant en le
    developpant selon la premiere ligne.
    """
    if A.shape == (1, 1):
        return A[0, 0]

    somme = 0

    for j in range(len(A)):
        # Developpement selon la premiere ligne
        N = extraction(A, 0, j)

        num = (-1) ** j * A[0, j] * det(N)
        somme += num

    return somme


if __name__ == "__main__":
    from numpy import linalg
    from numpy import random

    for _ in range(100):
        i = random.randint(1, 5)
        A = 10 * random.rand(i, i)
        B = A.round().astype(int)

        ref = round(linalg.det(B))
        calc = det(B)

        assert ref == calc, f"{B=}\n{ref=}\n{calc=}"
