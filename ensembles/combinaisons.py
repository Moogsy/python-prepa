from typing import TypeVar

T = TypeVar("T")


def combinations(li: list[T]) -> list[tuple[T, T]]:

    yield tuple()

    for i, x in enumerate(li):
        for comb in combinations(li[i + 1:]):
            yield tuple([x]) + comb


print([*combinations([1, 2, 3])])
