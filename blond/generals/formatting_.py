import math


def si_format(num, precision=2):
    if num == 0:
        return "0"

    prefixes = {
        -24: "y",
        -21: "z",
        -18: "a",
        -15: "f",
        -12: "p",
        -9: "n",
        -6: "µ",
        -3: "m",
        0: "",
        3: "k",
        6: "M",
        9: "G",
        12: "T",
        15: "P",
        18: "E",
        21: "Z",
        24: "Y",
    }

    exponent = int(math.floor(math.log10(abs(num)) / 3) * 3)
    exponent = max(min(exponent, 24), -24)

    value = num / (10**exponent)
    return f"{value:.{precision}f}{prefixes[exponent]}"
