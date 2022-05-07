from typing import Generator, List, TypeVar

T = TypeVar("T")


def perm(li: list) -> Generator[list, None, None]:

    if len(li) <= 1:
        return

    if len(li) == 2:
        yield li
        yield li[::-1]
        return

    for index, obj in enumerate(li):
        for sub in perm(li[:index] + li[index + 1 :]):
            yield [obj] + sub


def perm2(li: List[T]) -> List[List[T]]:

    if len(li) <= 1:
        return []

    if len(li) == 2:
        return [li, li[::-1]]

    out = []

    for index, obj in enumerate(li):
        for sub in perm2(li[:index] + li[index + 1 :]):
            out.append([obj] + sub)

    return out


def _test(fn):
    from itertools import permutations

    li = list(range(6))

    ref = [list(p) for p in permutations(li)]
    cmp = list(fn(li))

    assert all(x in ref for x in cmp)

    print("ok")


if __name__ == "__main__":
    _test(perm)
    _test(perm2)
