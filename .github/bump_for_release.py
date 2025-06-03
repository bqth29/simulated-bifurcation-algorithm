import argparse
import datetime
import os
import re


def check_release_month(line: str, current_date: datetime.date):
    current_month = current_date.strftime("%B").lower()[:3]
    month_match = re.search("month = (?P<month>[a-z]{3})", line)
    if month_match is not None:
        print(month_match["month"])
    if month_match is not None and month_match["month"] != current_month:
        raise ValueError(
            f"Release month in README.md ({month_match['month']}) is not consistent with current month ({current_month}). Please update it."
        )


def check_release_year(line: str, current_date: datetime.date):
    current_year = str(current_date.year)
    year_match = re.search("year = {(?P<year>\d{4})}", line)
    if year_match is not None and year_match["year"] != current_year:
        raise ValueError(
            f"Release year in README.md ({year_match['year']}) is not consistent with current year ({current_year}). Please update it."
        )


def check_release_date():
    current_date = datetime.date.today()
    with open("README.md", "r") as readme:
        lines = readme.readlines()
        for line in lines:
            check_release_month(line, current_date)
            check_release_year(line, current_date)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "version", metavar="N", type=int, nargs=3, help="major / minor / patch"
    )

    args = parser.parse_args()
    major, minor, patch = args.version

    check_release_date()

    new_branch = f"prepare-release-{major}.{minor}.{patch}"
    os.system(f"git checkout -b {new_branch}")
    os.system(f"bump2version --new-version {major}.{minor}.{patch} --commit .")
    os.system(f"bump2version --new-version {major}.{minor + 1}.0.dev0 --commit .")
    os.system(f"git push --set-upstream origin {new_branch}")
