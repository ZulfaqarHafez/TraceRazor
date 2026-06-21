import sys

from core import load_records


def main(path):
    records = load_records(path)
    print(f"{len(records)} records")
    return len(records)


if __name__ == "__main__":
    main(sys.argv[1])
