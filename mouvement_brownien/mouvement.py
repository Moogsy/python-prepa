""" 
Issu de la partie II du sujet Mines Pont 2021 d'informatique commune
"""
from math import cos, sin, pi 
from random import gauss, uniform

ALPHA = 1e-5
SIGMA = 1e-8 
MASSE = 1e-5

Vecteur = tuple[float, float]
VecteurEtat = tuple[float, float, float, float]

def vma(v1: VecteurEtat, a: float, v2: VecteurEtat) -> VecteurEtat:
    """ 
    Renvoie le vecteur v = v1 + a * v2
    """
    assert len(v1) == len(v2)

    return tuple(x1 + a * x2 for x1, x2 in zip(v1, v2))


def fb() -> Vecteur: 
    """
    Renvoie le vecteur fb ayant une norme aleatoire suivant 
    une direction isotrope aleatoire uniforme, et une norme qui est la valeur 
    absolue d'une variable aleatoire suivant une loi de probabilite gaussienne
    """
    norme = abs(gauss(0, SIGMA))

    theta = uniform(0, 2 * pi)

    return norme * cos(theta), norme * sin(theta)



def derive(E: VecteurEtat) -> VecteurEtat:
    """
    Derive le vecteur d'etat 
    E = (x, y, vx, vy) 
    regi par l'equation differentielle 

    dv/dt = - (ALPHA * v) / m + fb / m

    avec:
        - ALPHA : une constante issue d'un force de frottement fluide
        - fb    : un vecteur de norme et de direction aleatoire
        - m     : la MASSE de la particule etudiee
    """
    _, _, vx0, vy0 = E

    x1 = vx0 
    y1 = vy0

    fbx, fby = fb()

    vx1 = - (ALPHA * vx0) / MASSE + fbx / MASSE
    vy1 = - (ALPHA * vy0) / MASSE + fby / MASSE

    return x1, y1, vx1, vy1


def euler(E0: VecteurEtat, dt: float, n: int) -> list[VecteurEtat]:
    """ 
    Resolution de l'equation differentielle par methode d'euler
    """
    vecteurs_etats = [E0]
    E = E0
    
    for _ in range(n):
        E = vma(E, dt, derive(E))
        vecteurs_etats.append(E)
        
    return vecteurs_etats


def test(): 
    from matplotlib import pyplot as plt

    _, ax = plt.subplots()

    X = [] 
    Y = [] 

    E0 = 0, 0, 0, 0 
    dt = 1e-5
    n = int(1e3)

    for x, y, _, _ in euler(E0, dt, n):
        X.append(x) 
        Y.append(y)

    ax.plot(X, Y, marker='o')
    plt.show()


if __name__ == "__main__":
    test()




