def r_insert(obj: object, li: list) -> list:
    if not li:
        return [obj]

    if obj <= li[0]:
        return [obj] + li

    return [li[0]] + r_insert(obj, li[1:])


def r_insertion_sort(li: list) -> list:
    if not li:
        return []

    low = min(li)
    li.remove(low) 

    return [low] + r_insertion_sort(li)

def _is_sorted(li: list) -> bool:
    for a, b in zip(li, li[1:]):
        if a > b:
            return False

    return True

def _test():
    from random import sample

    li = sample(range(1, 101), k=100)

    for a, b in zip(li, li[1:]):

        s_li = list(range(1, a + 1))

        try:
            s_li.remove(b)
        except ValueError:
            pass

        assert _is_sorted(r_insert(b, s_li))

        assert _is_sorted(r_insertion_sort(li))

    print("ok")

if __name__ == "__main__":
    _test()



