from typing import List

def r_decomposition(n: int) -> List[int]:

    if 0 <= n <= 1:
        return [n]

    quotient, remainder = divmod(n, 2)

    return r_decomposition(quotient) + [remainder]

def _test():
    for k in range(100):
        dec = r_decomposition(k)

        cmp = [int(d) for d in bin(k)[2:]]

        assert dec == cmp

    print("ok")

if __name__ == "__main__":
    _test()
