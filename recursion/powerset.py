from typing import Generator, TypeVar

T = TypeVar('T')

def powerset(li: list) -> Generator[list, None, None]:

    yield []

    for i, x in enumerate(li):
        for sub in powerset(li[i+1:]):
            yield [x] + sub


def powerset2(li: list) -> list:
    out = []

    out.append([])

    for i, x in enumerate(li):
        for sub in powerset(li[i+1:]):
            out.append([x] + sub)

    return out


def _test(fn):
    from itertools import combinations

    for n in range(6):
        li = list(range(n))

        cmp = [] 
        for r in range(len(li) + 1):
            cmp.extend([list(x) for x in combinations(li, r=r)])

        assert all(x in cmp for x in fn(li))

    print("ok")


if __name__ == "__main__":
    _test(powerset)
    _test(powerset2)




