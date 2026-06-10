def normalize_name(name):
    # strips and lowercases, but does not collapse internal whitespace
    return name.strip().lower()


def label_for(name):
    return f"user:{normalize_name(name)}"
