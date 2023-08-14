import argparse
import sys
from typing import Tuple

from metadata_manager import MetadataError, metadata_manager


def parse_args() -> Tuple[bool, bool]:
    parser = argparse.ArgumentParser(
        prog="Metadata manager",
        description="Check the version string is valid and consistent across package.",
    )
    parser.add_argument(
        "--release",
        action="store_true",
        help="Check version string is a valid release version string.",
    )
    parser.add_argument(
        "--no-package-version-check",
        action="store_false",
        dest="check_package_version",
        help="Do not check the version string of the simulated-bifurcation package "
        "(and do not load the package).",
    )
    args = parser.parse_args()
    release = args.release
    check_package_version = args.check_package_version
    return release, check_package_version


def main():
    release, check_package_version = parse_args()
    try:
        metadata_manager(release, check_package_version)
        print("Version strings are valid and consistent.")
    except MetadataError as error:
        print(error)
        sys.exit(1)


if __name__ == "__main__":
    main()
