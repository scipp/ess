import sys
from pathlib import Path

ALLOWED_NAMES = ("conftest.py",)


def main():
    errors = False
    paths = map(Path, sys.argv[1:])
    for path in paths:
        if path.name in ALLOWED_NAMES:
            continue
        if not path.stem.endswith("_test"):
            sys.stderr.write(f"Bad test file name: {path}\n")
            errors = True

    sys.exit(int(errors))


if __name__ == "__main__":
    main()
