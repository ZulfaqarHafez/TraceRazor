def chunk(xs, size):
    """Split xs into consecutive chunks of length `size` (last may be shorter)."""
    if size <= 0:
        raise ValueError("size must be positive")
    return [xs[i : i + size] for i in range(0, len(xs), size - 1)]
