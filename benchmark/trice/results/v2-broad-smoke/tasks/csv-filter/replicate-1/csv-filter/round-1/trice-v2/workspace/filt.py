import csv


def filter_rows(path, min_score):
    out = []
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            score = int(row["score"])
            if score >= min_score:
                out.append({"name": row["name"], "score": score})
    return out
