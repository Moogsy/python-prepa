from typing import Protocol, TypeVar

T_co = TypeVar("T_co", covariant=True)

class RealFunction(Protocol[T_co]):
    def __call__(self, __x: float) -> float: ...

class ParamFunction(Protocol[T_co]):
    def __call__(self, __x1: float, __x2: float) -> float: ...


def euler(f, x0, y0, b, n) -> tuple[list[float], list[float]]:
    """
    Résolution numérique de l'équation différentielle 
    y'(t) = f(t, y(t)) en utilisant la méthode d'Euler

    Note:
        Diverge assez rapidement si le pas h est trop grand
        Fomule de récurrence: y_{n+1} = y_n + h f(t_n, y_n)
    """
    h = (b - x0) / n
    X = [x0] 
    Y = [y0]

    for _ in range(1, n + 1):
        x1 = x0 + h
        y1 = y0 + h * f(x0, y0)
        X.append(x1)
        Y.append(y1)
        x0 = x1
        y0 = y1

    return X, Y



