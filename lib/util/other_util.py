from math import ceil, log2, floor


def nearest_2_power(n: int):
    upper = 2 ** ceil(log2(n))
    lower = 2 ** floor(log2(n))
    if upper / n < n / lower:
        return upper
    else:
        return lower
