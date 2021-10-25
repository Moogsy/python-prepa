def r_fib(n: int) -> int: # O(2**n)

    if n == 0:
        return 0

    if n == 1:
        return 1 

    return r_fib(n - 1) + r_fib(n - 2)

def fib(n: int) -> int:

    a = 0 
    b = 1 

    for _ in range(n):
        a, b = b, a + b

    return a


def _test():
    for n in range(10):
        assert r_fib(n) == fib(n)

    print("ok")

if __name__ == "__main__":
    _test()
