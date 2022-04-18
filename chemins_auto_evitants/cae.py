from math import cos, sin 

Point = tuple[int, int]
Matrice = list[list[float]]

def positions_possibles(p: Point, atteints: list[Point]) -> list[Point]:
    possibles = []

    x, y = p

    candidats = [
        [x + 1, y],
        [x - 1, y],
        [x, y + 1],
        [x, y - 1],
    ]

    for candidat in candidats:
        if candidat not in atteints:
            possibles.append(candidat)

    return possibles


def est_cae(chemin: list[Point]) -> bool:
    """
    Verifie si un chemin est auto-evitant
    """
    points = list(sorted(chemin))

    for p1, p2 in zip(points, points[1:]):
        if p1 == p2:
            return False

    return True


def mat_rot(theta: str) -> list[list]:
    """
    Renvoie a matrice de rotation d'angle theta (en radians)
    """
    return [
        [cos(theta), sin(theta)], 
        [sin(theta), -cos(theta),
    ]


def prod_mat2(mat1: Matrice, mat2: Matrice) -> Matrice:
    pass
