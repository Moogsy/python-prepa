"""
Issu de la partie III du sujet Mines-Pont 2021 d'informatique commune 
"""
from random import randrange

Vecteur = tuple[int, ...]
Vecteur3D = tuple[float, float, float]
Point = Vecteur2D = tuple[int, int]
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

        atteints.append(position_actuelle)  # type: ignore

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
    

def sub(p1: Vecteur2D, p2: Vecteur2D) -> Vecteur2D:
    """
    Soustraction de deux points
    Dans la base canonique, cela revient a considerer les coordonnees de p1 lorsque l'origine 
    est remplacee par p2
    """
    x1, y1 = p1 
    x2, y2 = p2

    return (x1 - x2, y1 - y2)


def add(p1: Vecteur2D, p2: Vecteur2D) -> Vecteur2D:
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


def produit_vectoriel(v1: Vecteur3D, v2: Vecteur3D) -> Vecteur3D:
    """
    Produit vectoriel entre deux vecteurs
    """
    a, b, c = v1 
    d, e, f = v2

    return (
        b * f - c * e,
        c * d - a * f, 
        a * e - b * d
    )


def angle_exclu(op: Vecteur2D, os: Vecteur2D) -> int:
    """
    Trouve l'angle qui doit etre exclu en utilisant la definition du produit 
    vectoriel de deux vecteurs unitaires

    os ^ op = sin(os, op) * u

    On a trois configurations possibles:
    
    -------
    S
    | 
    O -> P 

    Leur produit vectoriel donne un vecteur selon +Uz
    ------

    O -> P 
      -> S 

    Leur produit vectoriel est nul
    ------

    O -> P 
    |
    S

    Dont le produit vectoriel est selon -Uz
    """
    xs, ys = os 
    os3 = (xs, ys, 0)

    xp, yp = op 
    op3 = (xp, yp, 0)

    _, _, z = produit_vectoriel(os3, op3)

    if z == 0:
        return 0

    elif z == 1:
        return 1
    
    else:
        return 2


def tire_a(angles_exclus: list[int]) -> int:
    """
    Tire un angle parmi ceux qui sont encore utilisables
    """
    possibles = [x for x in range(3) if x not in angles_exclus]
    return possibles[randrange(len(possibles))]


def genere_a(chemin: Chemin, i_piv: int, angles_exclus: list) -> int:
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

    # On cherche a determiner l'angle entre les vecteurs 
    # pivot->precedent et pivot->suivant
    op = sub(precedent, pivot) 
    os = sub(suivant, pivot)
        
    exclu = angle_exclu(op, os)
    angles_exclus.append(exclu)

    return tire_a(angles_exclus)

def initialisation(n: int):
    """
    Initialise un chemin deja auto-evitant
    """
    return [(i, 0) for i in range(n + 1)]


def maj_chemin(chemin: Chemin, pivots_utilisables: list[int]) -> tuple[Chemin, list[int]]:
    while True:
        angles_exclus = []
        i_piv = pivots_utilisables[randrange(len(pivots_utilisables))]

        a = genere_a(chemin, i_piv, angles_exclus)
        nv_chemin = rotation(chemin, i_piv, a)

        if est_CAE(nv_chemin):
            return nv_chemin, pivots_utilisables

        angles_exclus.append(a)

        # Cas ou le chemin se replie sur lui-meme, on change de pivot
        if len(angles_exclus) == 3:
            pivots_utilisables.remove(i_piv)
            i_piv = pivots_utilisables[randrange(len(pivots_utilisables))]


def genere_chemin_pivot(n: int, n_rot: int) -> Chemin:
    """
    Genere un chemin en utilisant la methode du pivot
    """
    chemin = initialisation(n)
    pivots_utilisables = list(range(1, n))

    for _ in range(n_rot):
        nv_chemin, pivots_utilisables = maj_chemin(chemin, pivots_utilisables)

        assert nv_chemin != chemin

        chemin = nv_chemin


    return chemin

from matplotlib import pyplot as plt

fig, ax = plt.subplots() 

X = [] 
Y = []

for _ in range(100):
    chemin = genere_chemin_pivot(100, 10)
    print("ok")

exit()
for x, y in chemin:
    X.append(x)
    Y.append(y)

ax.plot(X, Y, marker='o')
plt.show()
