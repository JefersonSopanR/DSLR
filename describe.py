import argparse
from DSLR.describe_ import describe

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("first", type=str)

    arg = parser.parse_args()

    describe(arg.first)
