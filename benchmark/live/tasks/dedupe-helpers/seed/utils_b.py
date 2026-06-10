def normalize_name(name):
    # collapses spaces but forgets tabs/newlines and lowercasing
    return " ".join(name.split(" ")).strip()


def greeting(name):
    return f"hello {normalize_name(name)}"
