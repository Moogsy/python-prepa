from __future__ import annotations
from math import sqrt, log

def pgcd(a: int, b: int) -> int:
    """
    Calcul du PGCD de deux nombres en utilisant l'algorithme d'Euclide
    """
    while b != 0:
        a, b = b, a % b
    
    return a

def ppcm(a: int, b: int) -> int: 
    """
    Calcul du PPCM en utilisant la relation 
    |a * b| = pgcd(a, b) * ppcm(a, b)
    où a et b sont des entiers
    """
    return abs(a * b) // pgcd(a, b)

def est_premier(n: int) -> bool: 
    """
    Verifie si n est premier
    """
    if n % 2 == 0:
        return False
    
    for k in range(3, int(sqrt(n)) + 1, 2):
        if n % k == 0:
            return False

    return True

def crible_eratosthene(n: int) -> list[int]: 
    """
    Renvoie tous les nombres premiers inférieurs à n 
    en utilisant le crible d'Erastothène
    """
    if n <= 1:
        return []
    
    tableau = [(False if k % 2 == 0 else True) for k in range(n + 1)]
    tableau[1] = False
    tableau[2] = True

    for i in range(3, n // 2 + 1, 2):
        for j in range(2, n // i + 1):
            tableau[i * j] = False

    return [i for i, est_premier in enumerate(tableau) if est_premier]


def facteurs_premiers(n: int) -> list[tuple[int, int]]:
    """
    Décomposition en facteurs premiers de n.
    Si n est de la forme: (p_1 ** k_1) * (p_2 ** k_2) * ... * (p_q ** k_q)
    Renvoie une liste [(p_1, k_1), (p_2, k_2), ..., (p_q, k_q)] 
    où q désigne le nombre de facteurs premiers que n possède.
    """
    if n <= 1:
        return []

    nb_premiers = crible_eratosthene(n)

    if n == nb_premiers[-1]: 
        return [(n, 1)]

    decomposition = []

    for p in nb_premiers:
        k = max(k for k in range(int(log(n) / log(p) + 1)) if n % p**k == 0)
        if k != 0:
            decomposition.append((p, k))

    return decomposition

def bezout(a: int, b: int) -> tuple[int, int]:
    """
    Renvoie un couple de coefficients (u, v) tel que a * u + b * v = 1
    """
    pass

