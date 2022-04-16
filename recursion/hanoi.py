from typing import Generator, List, Tuple, TypeVar

T = TypeVar('T')

def hanoi(n: int, start: T = 'A', aux: T = 'B', target: T = 'C') -> Generator[Tuple[T, T], None, None]:
    if n == 0:
        return 

    yield from hanoi(n - 1, start=start, aux=target, target=aux)

    yield (start, target)

    yield from hanoi(n - 1, start=aux, aux=start, target=target)


def hanoi2(n: int, start: T = 'A', aux: T = 'B', target: T = 'C') -> List[Tuple[T, T]]:
    if n == 0:
        return []

    out = []

    for mv in hanoi(n - 1, start=start, aux=target, target=aux):
        out.append(mv)

    out.append((start, target))

    for mv in hanoi(n - 1, start=aux, aux=start, target=target):
        out.append(mv)

    return out


def _is_sorted_desc(li: list) -> bool:
    for a, b in zip(li, li[1:]):
        if a < b:
            return False

    return True


def _test_hanoi(fn):
    for n in range(1, 5):
        
        a = list(range(n, 0, -1))
        b = []
        c = []

        for from_, to in fn(n, a, b, c):
            disk = from_.pop()
            to.append(disk)

            assert _is_sorted_desc(a)
            assert _is_sorted_desc(b)
            assert _is_sorted_desc(c)

        assert (not a)
        assert (not b)
        assert len(c) == n

    print("ok")

if __name__ == "__main__":
    _test_hanoi(hanoi)
    _test_hanoi(hanoi2)
