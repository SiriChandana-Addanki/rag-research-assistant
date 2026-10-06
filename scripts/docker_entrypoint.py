"""Dispatch Docker arguments to the existing CLI or requested command."""
import os
import sys


def main():
    args = sys.argv[1:]

    if not args:
        os.execvp("python", ["python", "scripts/ask.py", "--help"])

    if args[0] in {"-m", "-c", "-V", "--version"}:
        os.execvp("python", ["python", *args])

    if args[0] == "pytest":
        os.execvp(args[0], args)

    if args[0] in {"python", "python3"}:
        os.execvp(args[0], args)

    if args[0].endswith(".py"):
        os.execvp("python", ["python", *args])

    # Application options and positional questions are both handled by ask.py.
    os.execvp("python", ["python", "scripts/ask.py", *args])


if __name__ == "__main__":
    main()
