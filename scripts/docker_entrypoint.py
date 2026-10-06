"""Dispatch Docker arguments to the existing CLI or requested command."""
import os
import sys


def main():
    args = sys.argv[1:]

    if not args:
        os.execvp("python", ["python", "scripts/ask.py", "--help"])

    if args[0] in {"-h", "--help"} or args[0].startswith("--"):
        os.execvp("python", ["python", "scripts/ask.py", *args])

    if args[0] in {"-m", "-c", "-V", "--version"}:
        os.execvp("python", ["python", *args])

    if args[0].endswith(".py"):
        os.execvp("python", ["python", *args])

    os.execvp(args[0], args)


if __name__ == "__main__":
    main()
