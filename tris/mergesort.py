def merge(gauche: list, droite: list) -> list:

    sortie = []

    while gauche and droite:

        if gauche[0] <= droite[0]:
            sortie.append(gauche.pop(0))
        else:
            sortie.append(droite.pop(0))

    return sortie + gauche + droite


def merge_sort(li: list) -> list:
    if len(li) <= 1:
        return li

    milieu = len(li) // 2

    gauche = merge_sort(li[:milieu])

    droite = merge_sort(li[milieu:])

    return merge(gauche, droite)
