import sys

from metadata_manager import MetadataError, metadata_manager


def main():
    # TODO argparse
    try:
        metadata_manager()
        print("Package version and setup.py version match.")
    except MetadataError as error:
        print(error)
        sys.exit(1)


if __name__ == "__main__":
    main()
