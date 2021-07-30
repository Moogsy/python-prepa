from __future__ import annotations

from typing import Protocol, TypeVar

T = TypeVar("T")
T_co = TypeVar("T_co", covariant=True)

class RealFunction(Protocol[T_co]): 
    def __call__(self, x: float) -> float:
        ...

def derivee(f: RealFunction, x: float, h: float = 1e-5) -> float:
    """
    Valeur approchée de la dérivée de f évaluée au point x en utilisant le pas h 
    dans l'expression de la dérivée symétrique

    Note:
        donne parfois une valeur même si la fonction n'est pas dérivable, e.x abs en 0.
    """
    return (f(x + h) - f(x - h)) / (2 * h)

def integrale_rect_g(f: RealFunction, a: float, b: float, n: int) -> float:
    """
    Intégrale de la fonction f sur le segment [a, b] en utilisant n rectangles
    à gauche 
    
    Note:
        Application directe de la définition d'une somme de Riemann
    """
    pas = (b - a) / n
    return sum(f(a + k * pas) for k in range(n + 1)) * pas

def integrale_rect_d(f: RealFunction, a: float, b: float, n: int) -> float:
    """
    Intégrale de la fonction f sur le segment [a, b] en utilisant n rectangles
    à droite 
    
    Note:
        Application directe de la définition d'une somme de Riemann
    """
    pas = (b - a) / n
    return sum(f(a + (k + 1) * pas) for k in range(n + 1)) * pas


def integrale_rect_m(f: RealFunction, a: float, b: float, n: int) -> float:
    """
    Intégrale de la fonction f sur le segment [a, b] en utilisant n rectangles
    au milieu

    Note:
        Donne de meilleurs résultats que la méthode des trapèze
    """
    pas = (b - a) / n
    return sum(f(a + (k + 0.5) * pas) for k in range(n + 1)) * pas


def integrale_trapeze(f: RealFunction, a: float, b: float, n: int) -> float:
    """
    Intégrale de la fonction f sur le segment [a, b] en utilisant n trapèzes.

    Note:
        Aire d'un trapèze: (base1 + base2) * hauteur / 2
    """
    pas = (b - a) / n
    return sum((f(k * pas) + f((k + 1) * pas)) * pas / 2 for k in range(n + 1))


def dichotomie(f: RealFunction, a: float, b: float, epsilon: float = 1e-5) -> float:
    """
    Recherche d'une racine de la fonction f dans [a, b] avec une precision de epsilon.

    Note:
        Converge toujours, mais assez lentement
        Nécessite d'avoir a < b
    """

    while b - a > epsilon:
        m = (a + b) / 2
        y = f(m)

        if abs(y) < epsilon:
            return m
        
        if f(a) * y > 0:
            a = m
        
        else:
            b = m
        
    return (a + b) / 2

def dichotomie_liste(li: list[int | float], val: int | float) -> int | None:
    """
    Renvoie l'index de la valeur recherchée ou None si elle n'est pas dans la liste
    
    Note:   
        Ne fonctionne que si la liste est triée
    """
    a = 0 
    b = len(li) - 1 

    while a <= b: 
        m = (a + b) // 2
        y = li[m]

        if y == val:
            return m
        
        if y < val:
            a = m + 1
        
        else: 
            b = m - 1

    return None

def newton(f: RealFunction, a: float, epsilon: float = 1e-5) -> float:
    """
    Valeur approchée d'une racine de f située au voisinage de a
    
    Note:
        Converge beaucoup plus rapidement, mais seulement si le point 
        de départ est suffisemment proche et si la dérivée ne s'annule pas

        Fonctionne avec les racines complexes
    """

    while abs(f(a)) > epsilon:
        a -= f(a) / derivee(f, a)

    return a
