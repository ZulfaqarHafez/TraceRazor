def read_rows(rows):
    """Pretend-load: accept a list of raw strings."""
    return [r.strip() for r in rows if r.strip()]
