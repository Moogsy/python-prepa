def divide_and_conquer(orig: str, string: str, start: int, stop: int):

    quotient, remainder = divmod(len(string), 2)
    mid = quotient + remainder

    print(string)

    if len(string) <= 1:
        return 

    divide_and_conquer(orig, string[start:mid], start, stop)


    divide_and_conquer(orig, string[mid:stop], start, stop)

def _test():
    string = "abcdef"

    divide_and_conquer(string, string, 0, len(string) - 1)



if __name__ == "__main__":
    _test()

