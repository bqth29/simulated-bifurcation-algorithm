import argparse
import datetime
import os
import re


def check_release_year():
    current_year = datetime.date.today().year
    with open("README.md", "r") as readme:
        lines = readme.readlines()
        for line in lines:
            date_match = re.search("year = {(?P<year>\d{4})}", line)
            if date_match is not None and date_match["year"] != str(current_year):
                raise ValueError(
                    f"Release year in README.md ({date_match['year']}) is not consistent with current year ({current_year}). Please update it."
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "version", metavar="N", type=int, nargs=3, help="major / minor / patch"
    )

    check_release_year()

    args = parser.parse_args()
    major, minor, patch = args.version
    os.system(f"bump2version --new-version {major}.{minor}.{patch} --commit .")
    os.system(f"bump2version --new-version {major}.{minor + 1}.{patch}.dev0 --commit .")
