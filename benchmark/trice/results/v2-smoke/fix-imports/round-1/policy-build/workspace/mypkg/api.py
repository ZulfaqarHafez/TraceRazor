from loaders import read_rows


def run_pipeline(rows):
    """Load raw rows and return them uppercased."""
    return [r.upper() for r in read_rows(rows)]
