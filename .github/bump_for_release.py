import argparse
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'version',
        metavar='N',
        type=int,
        nargs=3,
        help='version is made of exactly three integer values'
    )
    major, minor, patch = parser.parse_args()
    os.system(f"bump2version --new-version {major}.{minor}.{patch} .")