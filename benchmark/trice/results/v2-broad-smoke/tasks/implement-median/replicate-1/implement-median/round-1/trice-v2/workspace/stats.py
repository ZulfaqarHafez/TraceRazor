def mean(xs):
    """Arithmetic mean of a non-empty sequence."""
    if not xs:
        raise ValueError("mean of empty sequence")
    return sum(xs) / len(xs)


def median(xs):
    """Median of a non-empty sequence (average of middle two for even length)."""
    if not xs:
        raise ValueError("median of empty sequence")
    values = sorted(xs)
    mid = len(values) // 2
    if len(values) % 2:
        return values[mid]
    return (values[mid - 1] + values[mid]) / 2
