def mean(xs):
    """Arithmetic mean of a non-empty sequence."""
    if not xs:
        raise ValueError("mean of empty sequence")
    return sum(xs) / len(xs)


def median(xs):
    """Median of a non-empty sequence (average of middle two for even length)."""
    raise NotImplementedError
