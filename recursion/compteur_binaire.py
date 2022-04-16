from typing import Generator, List

def compteur_binaire(n: int) -> Generator[str, None, None]:

    if n == 1:
        yield '0'
        yield '1'
        
        return

    for num in compteur_binaire(n - 1):
        yield num.zfill(n)

    for num in compteur_binaire(n - 1):
        yield '1' + num.zfill(n - 1)


# on passe facilement de haut en bas

def compteur2(n: int) -> List[str]:
    if n == 1:
        return ['0', '1']

    out = []

    for num in compteur2(n - 1):
        out.append(num.zfill(n))

    for num in compteur2(n - 1):
        out.append('1' + num.zfill(n - 1))

    return out

def _test_compteur(fn):
    for k, num in enumerate(fn(10)):
        assert num == bin(k)[2:].zfill(10)

    print("ok")


if __name__ == "__main__":
    _test_compteur(compteur_binaire)
    _test_compteur(compteur2)
