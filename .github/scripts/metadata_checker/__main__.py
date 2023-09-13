import argparse
import sys

from metadata_checker import metadata_checker


def parse_args() -> bool:
    parser = argparse.ArgumentParser(
        prog="Metadata checker",
        description="Check metadata are valid and consistent across repository.",
    )
    parser.add_argument(
        "--release",
        action="store_true",
        help="Enable release specific checks.",
    )
    args = parser.parse_args()
    release = args.release
    return release


def main() -> None:
    release = parse_args()
    errors = metadata_checker(release)
    print()
    if errors:
        if len(errors) == 1:
            print("Metadata checker: one error occurred.", end="\n\n")
        else:
            print(f"Metadata checker: {len(errors)} errors occurred.", end="\n\n")
        print(*errors, sep="\n\n", end="\n\n")
        sys.exit(1)
    else:
        print("Metadata are valid and consistent.", end="\n\n")
        sys.exit(0)


if __name__ == "__main__":
    main()
