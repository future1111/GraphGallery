__all__ = ["merge_as_list"]

def merge_as_list(*args):
    out = []
    for x in args:
        if x is not None:
            if isinstance(x, (list, tuple)):
                out += x
            else:
                out += [x]
    return out