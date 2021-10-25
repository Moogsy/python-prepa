from typing import Any, TypeVar, List
from collections.abc import Callable

T = TypeVar('T')

def tau(f: Callable[..., T], g: Callable[[T], Any]) -> Callable[..., T]:
    return lambda *args, **kwargs: f(g(*args, **kwargs))

def inc(x: float) -> float:
    return x + 1

inc2 = tau(inc, inc)

def r_comp(lif: List[Callable[..., T]]) -> Callable[..., T]:
    if len(lif) == 1:
        return lif[0]

    return tau(lif[0], r_comp(lif[1:]))

def _test():
    from random import sample

    li = sample(range(1, 101), k=100)

    for a, b in zip(li, li[1:]):

        assert inc2(a) == a + 2
        assert r_comp([inc] * a)(b) == b + a


    print("ok")


if __name__ == "__main__":
    _test()


