"""
Issu de la partie III du sujet Mines-Pont 2021 d'informatique commune 
"""
from random import randrange

Point = tuple[int, int]
Chemin = list[Point]
Matrice = list[list[int]]


def positions_possibles(p: Point, atteints: list[Point]) -> Chemin:
    px, py = p

    admis = []
    candidats = [(px + 1, py), (px - 1, py), (px, py + 1), (px, py - 1)]

    for candidat in candidats:
        if candidat not in atteints:
            admis.append(candidats)

    return admis


def genere_chemin_naif(n: int) -> Chemin | None:
    depart = (0, 0)

    atteints = [depart]
    position_actuelle = depart

    for _ in range(n):
        possibles = positions_possibles(position_actuelle, atteints)

        if not possibles:
            return None

        i = randrange(len(possibles))
        position_actuelle = possibles[i]

        atteints.append(position_actuelle)

    return atteints


def est_CAE(chemin: Chemin) -> bool:
    """
    Verifie si un chemin est auto-evitant
    """
    trie = list(sorted(chemin))

    for p1, p2 in zip(trie, trie[1:]):
        if p1 == p2:
            return False

    return True


def mat_rot(a: int) -> Matrice:
    """ 
    Renvoie la matrice de rotation d'angle 
    pi      si a = 0
    pi / 2  si a = 1
    -pi / 2 si a = 2
    """
    if a == 0:
        return [[-1, 0], [0, -1]] 
    elif a == 1:
        return [[0, -1], [1, 0]]
    else:
        return [[0, 1], [-1, 0]]


def img_mat(mat: Matrice, p: Point) -> Point:
    """
    Image d'un point par une matrice 2x2
    """
    l1, l2 = mat
    a, b, c, d = l1 + l2

    x, y = p

    return (a * x + b * y, c * x + d * y)
    

def sub(p1: Point, p2: Point) -> Point:
    """
    Soustraction de deux points
    Dans la base canonique, cela revient a considerer les coordonnees de p1 lorsque l'origine 
    est remplacee par p2
    """
    x1, y1 = p1 
    x2, y2 = p2

    return (x1 - x2, y1 - y2)


def add(p1: Point, p2: Point) -> Point:
    """
    Addition de deux points
    Dans la base canonique, cela revient a considerer les coordonnees de p1 lorsque l'origine 
    est remplacee par -p2
    """
    x1, y1 = p1 
    x2, y2 = p2

    return (x1 + x2, y1 + y2)


def rot(p: Point, q: Point, a: int) -> Point:
    """
    Renvoie l'image du point q par la rotation de centre p de type a.
    """
    r = sub(q, p)
    mat = mat_rot(a)

    s = img_mat(mat, r) 

    return add(s, p)


def rotation(chemin: Chemin, i_piv: int, a: int) -> Chemin:
    """
    Renvoie un nouveau chemin en gardant inchange les points d'indices inferieurs a i_piv. 
    Les autres subissent une rotation de type a autour du point d'indice i_piv
    """
    nv_chemin = chemin[:i_piv + 1]
    p = chemin[i_piv + 1]

    for q in chemin[i_piv + 1:]:
        r = rot(p, q, a)
        nv_chemin.append(r)

    return nv_chemin


def ajustement(os: Point, op: Point) -> tuple[Point, Point]:
    """
    Le vecteur op est le vecteur pivot->precedent
    Le vecteur os est le vecteur pivot->suivant

    On les fait tourner afin de se retrouver dans l'une des situations suivantes 

    ------------------------------------------------------
    | Configuration A | Configuration B | Configuration C |
    |                 |                 |                 |
    | S               | O -> P          | O -> P          |
    | |               |   -> S          | |               |
    | O -> P          |                 | S               |
    -------------------------------------------------------
    """
    rot = mat_rot(1)
    while op[1] != 0 and op[1] < 0:
        op = img_mat(rot, op) 
        os = img_mat(rot, os)

    return os, op


def tire_a(os: Point) -> int:
    """
    Apres ajustement, tire un angle qui n'est pas possible au vu des points precedents
    et suivants
    """ 
    # Configuration A, pas de rotations d'angle pi/2
    x, y = os
    if y > 0:
        return 2 * randrange(2)

    # Configuration B, pas de rotations d'angle pi
    elif y < 0:
        return randrange(2)

    # Configuration C, pas de rotation d'angle pi
    else:
        return 1 + randrange(2)


def genere_a(chemin: Chemin, i_piv: int) -> int:
    """
    Selectionne aleatoirement un type de rotation en evitant celles qui sont a premiere vue 
    impossibles en observant les points alentours
    """
    # Si le pivot est le premier point, tout le chemin tourne, pas de soucis
    # Si le pivot est le dernier point, rien ne change
    if i_piv == 0 or i_piv == (len(chemin) - 1):
        return randrange(3)

    pivot = chemin[i_piv]
    precedent = chemin[i_piv - 1]
    suivant = chemin[i_piv + 1]

    # On considere les vecteurs pivot->precedent et pivot->suivant
    op = sub(precedent, pivot) 
    os = sub(suivant, pivot)

    os, op = ajustement(os, op)       
        
    return tire_a(os)


def initialisation(n: int):
    """
    Initialise un chemin deja auto-evitant
    """
    return [(i, 0) for i in range(n + 1)]


def genere_chemin_pivot(n: int, n_rot: int) -> Chemin:
    """
    Genere un chemin en utilisant la methode du pivot
    """
    chemin = initialisation(n)

    for _ in range(n_rot):

        # Utiliser le dernier point comme pivot ne change rien
        i_piv = randrange(n - 1)

        # A quand une boucle do while ...
        a = genere_a(chemin, i_piv)
        nv_chemin = rotation(chemin, i_piv, a)

        while not est_CAE(nv_chemin):
            a = genere_a(chemin, i_piv)
            nv_chemin = rotation(nv_chemin, i_piv, a)

        chemin = nv_chemin

    return chemin

from matplotlib import pyplot as plt

fig, ax = plt.subplots() 

X = [] 
Y = []

chemin = genere_chemin_pivot(100, 15)

for x, y in chemin:
    X.append(x)
    Y.append(y)

ax.plot(X, Y, marker='o')
plt.show()
