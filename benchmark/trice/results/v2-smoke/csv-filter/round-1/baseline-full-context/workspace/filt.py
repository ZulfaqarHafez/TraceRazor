import csv


def filter_rows(path, min_score):
    """Read a CSV with header `name,score`; return rows with score >= min_score.

    Each returned row is a dict {"name": str, "score": int}, in file order.
    """
    out = []
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            score = int(row["score"])
            if score >= min_score:
                out.append({"name": row["name"], "score": score})
    return out
