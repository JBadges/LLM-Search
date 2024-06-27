def try_cast_int(s):
    try:
        return int(s)
    except ValueError:
        return None