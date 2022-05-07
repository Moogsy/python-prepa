def min_index(li: list, utilises: list[int]) -> tuple[int, object]:
    """
    Renvoie l'indice et le minimum de la liste
    passee en argument, en excluant ceux dont les indices
    sont contenus dans la liste utilises.
    """
    index = min(i for i in range(len(li)) if i not in utilises)
    mini = li[index]

    for i, x in enumerate(li):
        if x <= mini and i not in utilises:
            mini = x
            index = i

    return index, mini


def tri_selection(li: list) -> list:
    sortie = []
    utilises = []

    for _ in range(len(li)):
        index, mini = min_index(li, utilises)
        utilises.append(index)
        sortie.append(mini)

    return sortie


li = [2, 5, 1, 6, 7, 3]

print(tri_selection(li))
