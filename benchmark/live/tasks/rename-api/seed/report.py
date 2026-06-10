from core import fetch_data


def summarize(path):
    records = fetch_data(path)
    return {"count": len(records), "keys": sorted({k for r in records for k in r})}
