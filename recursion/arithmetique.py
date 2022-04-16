def r_pgcd(n: int, m: int) -> int:
    if n == 0:
        return m 

    if m < n:
        return r_pgcd(m, n)

    return r_pgcd(n, m % n)


def r_mul(a: float, b: int) -> float: 
    if b == 0:
        return 0

    return a + r_mul(a, b - 1)

def r_reste(a: int, b: int) -> int:
    if a < b:
        return a

    return r_reste(a - b, b)

def r_quotient(a: int, b: int) -> int:
    if a < b:
        return 0

    return 1 + r_quotient(a - b, b)

def r_pow(a: int, b: int) -> int: # O(b)
    if b == 0:
        return 1 

    return a * r_pow(a, b - 1)

def _test():
    from random import sample 
    from math import gcd

    nums = sample(range(1, 101), k=100)

    for a, b in zip(nums, nums[1:]):
        assert r_pgcd(a, b) == gcd(a, b)
        assert r_mul(a, b) == a * b
        assert r_reste(a, b) == a % b
        assert r_quotient(a, b) == a // b
        assert r_pow(a, b) == pow(a, b)

    print("ok")

if __name__ == "__main__":
    _test()











if __name__ == "__main__":
    pass
