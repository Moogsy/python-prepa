def insert_sort(li: list) -> list:
    """
    Tri par insertion. 
    Deplace les elements jusqu'a la bonne place
    """
    for i in range(len(li)):
        j = i
        while j > 0 and li[j - 1] > li[j]:
            li[j - 1], li[j] = li[j], li[j - 1]
            j -= 1

    return li
