import sys

from core import fetch_data


def main(path):
    records = fetch_data(path)
    print(f"{len(records)} records")
    return len(records)


if __name__ == "__main__":
    main(sys.argv[1])
