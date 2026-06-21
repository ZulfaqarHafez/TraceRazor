from core import load_records


def summarize(path):
    records = load_records(path)
    return {"count": len(records), "keys": sorted({k for r in records for k in r})}
