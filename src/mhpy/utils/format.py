import math

COUNT_UNITS = ["", "k", "M", "B", "T", "P"]
SIZE_UNITS = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")


def fcount(num: int) -> str:
    for unit in COUNT_UNITS:
        if abs(num) < 1000:
            return f"{num:g}{unit}"
        num /= 1000.0
    return f"{num:g}E"


def fsize(size_bytes: int) -> str:
    if size_bytes == 0:
        return "0B"

    i = int(math.floor(math.log(size_bytes, 1024)))
    i = min(i, len(SIZE_UNITS) - 1)

    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)

    return f"{s:g}{SIZE_UNITS[i]}"
