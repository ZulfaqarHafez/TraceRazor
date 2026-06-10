def filter_rows(path, min_score):
    """Read a CSV with header `name,score`; return rows with score >= min_score.

    Each returned row is a dict {"name": str, "score": int}, in file order.
    """
    raise NotImplementedError
