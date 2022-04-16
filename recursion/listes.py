def r_length(li: list) -> int:
    if not li:
        return 0 

    return 1 + r_length(li[1:])

def r_replicate(obj: object, n: int) -> list:

    if n == 0:
        return []

    return [obj] + r_replicate(obj, n - 1)

def r_sum(li: list):
    if not li:
        return 0

    return li[0] + r_sum(li[1:])

def r_product(li: list):

    if not li:
        return 1 

    return li[0] * r_product(li[1:])

def r_reverse(li: list) -> list:
    if not li:
        return []

    return [li[-1]] + r_reverse(li[:-1])

def r_zip(li1, li2) -> list:
    if not (li1 and li2):
        return []

    return [(li1[0], li2[0])] + r_zip(li1[1:], li2[1:])

def r_map(f, li: list) -> list:
    if not li:
        return []

    return [f(li[0])] + r_map(f, li[1:])

def r_max(li: list):

    if len(li) == 1:
        return li[0]

    if li[0] <= li[1]:
        return r_max(li[1:])

    return r_max([li[0]] + li[2:])

def r_est_dans(obj: object, li: list) -> bool:
    if not li:
        return False

    if li[0] == obj:
        return True

    return r_est_dans(obj, li[1:])


def r_horner(poly: list, x: float) -> float:
    if not poly:
        return 0

    return poly[0] + x * r_horner(poly[1:], x)

def _test():
    from random import sample

    from functools import reduce
    from operator import mul

    nums = sample(range(1, 101), k=100)

    for a, b in zip(nums, nums[1:]):
        
        li = sample(range(1, 101), k=a)
        li2 = sample(range(1, 101), k=b)

        def poly(x: float):
            return a * x**2 + b * x + (a + b)

        assert r_length(li) == len(li)
        assert r_replicate(a, b) == [a for _ in range(b)]
        assert r_sum(li) == sum(li)
        assert r_product(li) == reduce(mul, li, 1)
        assert r_zip(li, li2) == list(zip(li, li2))
        assert r_reverse(li) == li[::-1]
        assert r_map(poly, li) == list(map(poly, li))
        assert r_max(li) == max(li)
        assert r_est_dans(b, li) is (b in li)
        assert r_horner([a + b, b, a], a) == poly(a)

    print("ok")

if __name__ == "__main__":
    _test()






