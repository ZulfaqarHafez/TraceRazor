import json


def load_records(path):
    """Load a list of record dicts from a JSON file."""
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, list):
        raise TypeError("expected a JSON array of records")
    return data
