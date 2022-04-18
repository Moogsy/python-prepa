import numpy as np


def zeros_gauche(ligne: np.ndarray) -> int:
    """
    Compte le nombre de zeros a gauche sur une seule ligne
    """
    zeros = 0

    for case in ligne:
        if case == 0:
            zeros += 1
        else:
            return zeros

    return zeros  # Cas limite d'une matrice vide ...


def compte_decalage(A: np.ndarray) -> list:
    """
    Renvoie le nombre de zeros à gauche de chaque ligne de la matrice.
    """
    marges = []

    for num_ligne in range(len(A)):

        marge = zeros_gauche(A[num_ligne])
        marges.append(
            (marge, num_ligne)
        )  # Ordre peu naturel, mais necessaire pour trier

    return marges


def arrange_systeme(A: np.ndarray, Y: np.ndarray) -> tuple:
    """
    Cree une matrice B en permuttant les lignes de A,
    et une matrice Z en permuttant celles Y
    pour se rapprocher le plus possible d'un systeme triangulaire superieure
    """
    marges = compte_decalage(A)
    marges.sort()

    B = np.zeros(A.shape, dtype=A.dtype)
    Z = np.zeros(Y.shape, dtype=Y.dtype)

    for num_ligne in range(len(A)):

        _, num_ligne_cible = marges[num_ligne]

        B[num_ligne] = A[num_ligne_cible]
        Z[num_ligne] = Y[num_ligne_cible]

    return B, Z


def transvection(
    B: np.ndarray, Z: np.ndarray, recipient: int, arrivant: int, coeff: float
) -> tuple:
    """
    Effectue l'operation
    L_{recipient} <- L_{recipient} + coeff * L_{arrivant}
    """
    B[recipient] += coeff * B[arrivant]
    Z[recipient] += coeff * Z[arrivant]

    return B, Z


def dilatation(B: np.ndarray, Z: np.ndarray, recipient: int, coeff: float) -> tuple:
    """
    Effectue l'operation
    L_{recipient} <- coeff * L_{recipient}
    """
    B[recipient] *= coeff
    Z[recipient] *= coeff

    return B, Z


def elimination_var(num_var_cible: int, B: np.ndarray, Z: np.ndarray) -> tuple:
    """
    Amorce la resolution en eliminant la variable i du systeme
    """
    num_ligne_pivot = num_var_cible
    coeff_pivot = B[num_ligne_pivot, num_var_cible]

    for num_ligne in range(num_var_cible + 1, len(B)):
        coeff_cible_B = B[num_ligne, num_var_cible]
        coeff_transvection = coeff_cible_B / coeff_pivot

        B, Z = transvection(B, Z, num_ligne, num_ligne_pivot, -coeff_transvection)

    return B, Z


def remontee(B: np.ndarray, Z: np.ndarray) -> tuple:
    """
    Diagonalise un systeme deja trigonalise
    """
    for i in range(len(B)):  # Lignes de B
        num_ligne_cible = len(B) - i - 1  # On traverse de bas en haut
        num_colonne_cible = num_ligne_cible

        # On exprime les variables "a droite" de la diagonale
        # A ce stade, les lignes en dessous de num_ligne_cible
        # ne contiennent que la variable sur la diagonale

        for num_var_remplacee in range(num_ligne_cible + 1, len(B)):
            coeff_transvection = B[num_ligne_cible, num_var_remplacee]

            B, Z = transvection(
                B, Z, num_ligne_cible, num_var_remplacee, -coeff_transvection
            )

        # Ici, il ne reste plus qu'a normaliser la ligne de travail
        coeff_dilatation = 1 / B[num_ligne_cible, num_colonne_cible]
        B, Z = dilatation(B, Z, num_ligne_cible, coeff_dilatation)

    return B, Z


def resolution(A: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """
    Resolution de AX = Y en utilisant un pivot de Gauss
    Sous reserve d'unicite de la solution
    Renvoie une colonne de NaN sinon
    """
    B, Z = arrange_systeme(A, Y)

    for num_var_cible in range(len(A)):
        B, Z = elimination_var(num_var_cible, B, Z)

    B, Z = remontee(B, Z)

    return Z


if __name__ == "__main__":
    A = np.array(
        [
            [1, 2, 3],
            [5, 2, 1],
            [-4, 2, -1],
        ]
    )

    Y = np.array(
        [
            [2],
            [-1],
            [8],
        ]
    )

    Z = resolution(A, Y)
    print(Z)
