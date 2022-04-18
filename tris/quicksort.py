from random import randint


def partition(li: list, a: int, b: int, pos_pivot: int) -> int:
    """
    Partitionne les elements de li[a : b]:
    la partie gauche contient les elements inferieurs au pivot,
    et la partie droite, les element superieurs
    """
    pivot = li[pos_pivot]
    li[b], li[pos_pivot] = pivot, li[b]

    sep = a

    for i in range(a, b):
        if li[i] <= pivot:
            li[i], li[sep] = li[sep], li[i]

            sep += 1

    li[sep], li[b] = pivot, li[sep]

    return sep


def tri_rapide(li: list, a: int = 0, b: int = -1) -> list:
    if b == -1:
        b = len(li) - 1

    if a < b:
        pos_pivot = randint(a, b)
        pos_pivot = partition(li, a, b, pos_pivot)
        tri_rapide(li, a, pos_pivot - 1)
        tri_rapide(li, pos_pivot + 1, b)


a = [65, 2, 5, 2, 6, 7, 8]
tri_rapide(a)
print(a)