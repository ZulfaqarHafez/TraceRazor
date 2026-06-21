from textutil import normalize_name


def label_for(name):
    return f"user:{normalize_name(name)}"
