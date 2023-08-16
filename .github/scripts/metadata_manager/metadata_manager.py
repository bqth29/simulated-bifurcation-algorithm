import re
from enum import Flag, auto
from typing import Any, Callable, Optional

from config import (
    BIBTEX_TEMPLATE,
    DEV_VERSION_REGEX,
    FILES_SHOULD_DEFINE,
    METADATA_MANAGER_INVOCATION,
    RELEASE_VERSION_REGEX,
)


class MetadataErrorCode(Flag):
    PACKAGE_NOT_FOUND = auto()
    PACKAGE_MISSING_VERSION_STRING = auto()
    PACKAGE_INVALID_DEV_VERSION_STRING = auto()
    PACKAGE_INVALID_RELEASE_VERSION_STRING = auto()
    SETUP_NOT_FOUND = auto()
    SETUP_MISSING_VERSION_MARKER = auto()
    SETUP_MISSING_VERSION_LINE = auto()
    SETUP_INVALID_VERSION_LINE = auto()
    SETUP_INVALID_DEV_VERSION_STRING = auto()
    SETUP_INVALID_RELEASE_VERSION_STRING = auto()
    INCONSISTENT_VERSION_STRINGS = auto()

    PACKAGE_ERROR = (
        PACKAGE_NOT_FOUND
        | PACKAGE_MISSING_VERSION_STRING
        | PACKAGE_INVALID_RELEASE_VERSION_STRING
        | PACKAGE_INVALID_DEV_VERSION_STRING
    )
    SETUP_ERROR = (
        SETUP_NOT_FOUND
        | SETUP_MISSING_VERSION_MARKER
        | SETUP_MISSING_VERSION_LINE
        | SETUP_INVALID_VERSION_LINE
        | SETUP_INVALID_RELEASE_VERSION_STRING
        | SETUP_INVALID_DEV_VERSION_STRING
    )
    DEV_VERSION_STRING_ERROR = (
        PACKAGE_INVALID_DEV_VERSION_STRING | SETUP_INVALID_DEV_VERSION_STRING
    )
    RELEASE_VERSION_STRING_ERROR = (
        PACKAGE_INVALID_RELEASE_VERSION_STRING | SETUP_INVALID_RELEASE_VERSION_STRING
    )

    def error_message(self, **kwargs: Any) -> str:
        messages = []
        for error_code in self:
            message_method = error_code.__error_message_method()
            message = message_method(**kwargs)
            if message is not None:
                messages.append(message)
        message = "\n\n".join(messages)
        return message

    def __error_message_method(self) -> Optional[Callable[[...], str]]:
        method_name = f"{self.name.lower()}_message"
        try:
            method = getattr(self, method_name)
        except AttributeError:
            method = None
        return method

    def __invalid_version_string_message(self, version_string: str) -> str:
        def example_formatter(ex: str) -> str:
            return f'"{ex}"'

        if self in self.__class__.PACKAGE_ERROR:
            source = "simulated-bifurcation package"
        elif self in self.__class__.SETUP_ERROR:
            source = "setup.py"
        else:
            raise ValueError(f"Unexpected error code: {self}")

        if self in self.__class__.RELEASE_VERSION_STRING_ERROR:
            mode = "Release"
            regex = RELEASE_VERSION_REGEX
            examples = ["3.14.15", "1.42.0"]
        elif self in self.__class__.DEV_VERSION_STRING_ERROR:
            mode = "Development"
            regex = DEV_VERSION_REGEX
            examples = ["3.14.15", "1.42.0", "2.3.1.dev0", "2.3.1.dev15"]
        else:
            raise ValueError(f"Unexpected error code: {self}")

        message = (
            f"Invalid version string in {source}.\n"
            f'Found: "{version_string}".\n'
            f"{mode} version strings should match the following regular expression.\n"
            f"{regex}\n"
            f"Examples: {', '.join(map(example_formatter, examples))}"
        )
        return message

    @staticmethod
    def package_not_found_message(**_: Any) -> str:
        message = (
            "Could not import simulated-bifurcation package.\n"
            "Make sure the simulated-bifurcation package has been installed before"
            "running this script.\n"
            "If you do not want to check for the package version, run again "
            "with --no-package-check."
        )
        return message

    @staticmethod
    def package_missing_version_string_message(**_: Any) -> str:
        message = (
            'simulated-bifurcation does not define a version string "__version__".'
        )
        return message

    def package_invalid_dev_version_string_message(
        self, version_string: str, **_: Any
    ) -> str:
        return self.__invalid_version_string_message(version_string)

    def package_invalid_release_version_string_message(
        self, version_string: str, **_: Any
    ) -> str:
        return self.__invalid_version_string_message(version_string)

    @staticmethod
    def setup_not_found_message(**_: Any) -> str:
        message = "Could not find setup.py."
        return message

    @staticmethod
    def setup_missing_version_marker_message(**_: Any) -> str:
        message = (
            "Could not find version marker in setup.py.\n"
            "The line right above the package version string should be the following.\n"
            f"{SETUP_VERSION_ASSIGNMENT_MARKER}"
        )
        return message

    @staticmethod
    def setup_missing_version_line_message(**_: Any) -> str:
        message = (
            "Version marker in setup.py should be directly followed by package "
            "version assignment.\n"
            "It is either located at the end of the file or followed by a line of "
            "whitespaces."
        )
        return message

    @staticmethod
    def setup_invalid_version_line_message(
        setup_version_string_assignment_line: str, **_: Any
    ) -> str:
        message = (
            "Invalid version string assignment.\n"
            f"Found: {setup_version_string_assignment_line}.\n"
            "It should match the following regular expression.\n"
            f"{SETUP_VERSION_ASSIGNMENT_LINE_REGEX}"
        )
        return message

    def setup_invalid_dev_version_string_message(
        self, version_string: str, **_: Any
    ) -> str:
        return self.__invalid_version_string_message(version_string)

    def setup_invalid_release_version_string_message(
        self, version_string: str, **_: Any
    ) -> str:
        return self.__invalid_version_string_message(version_string)

    @staticmethod
    def inconsistent_version_strings_message(
        setup_version: str, package_version: str, **_: Any
    ) -> str:
        message = (
            "Version strings do not match.\n"
            f'Version string from setup.py: "{setup_version}"\n'
            f'Version string from simulated-bifurcation package: "{package_version}"'
        )
        return message


class MetadataError(Exception):
    def __init__(self, error_code: MetadataErrorCode, /, **kwargs: Any):
        self.error_code = error_code
        self.kwargs = kwargs

    def __str__(self) -> str:
        return self.error_code.error_message(**self.kwargs)


def read_setup_file() -> str:
    try:
        with open("setup.py", "r", encoding="utf-8") as setup_file:
            setup = setup_file.read()
    except FileNotFoundError:
        raise MetadataError(MetadataErrorCode.SETUP_NOT_FOUND) from None
    return setup


def get_version_string_from_setup(setup: str) -> str:
    version_line_pattern = re.compile(SETUP_VERSION_ASSIGNMENT_LINE_REGEX)

    setup = map(str.strip, setup.strip().splitlines())

    try:
        while next(setup) != SETUP_VERSION_ASSIGNMENT_MARKER:
            pass
    except StopIteration:
        raise MetadataError(MetadataErrorCode.SETUP_MISSING_VERSION_MARKER) from None

    try:
        version_string_assignment_line = next(setup)
    except StopIteration:
        raise MetadataError(MetadataErrorCode.SETUP_MISSING_VERSION_LINE) from None

    if version_string_assignment_line == "":
        raise MetadataError(MetadataErrorCode.SETUP_MISSING_VERSION_LINE) from None

    if version_line_pattern.match(version_string_assignment_line) is None:
        raise MetadataError(
            MetadataErrorCode.SETUP_INVALID_VERSION_LINE,
            setup_version_string_assignment_line=version_string_assignment_line,
        ) from None

    version_string = version_string_assignment_line[
        len(SETUP_VERSION_ASSIGNMENT)
        + len(SETUP_VERSION_STRING_DELIMITER) : -len(SETUP_VERSION_STRING_DELIMITER)
    ]
    return version_string


def get_version_string_from_package() -> str:
    try:
        import simulated_bifurcation as sb
    except ModuleNotFoundError:
        raise MetadataError(MetadataErrorCode.PACKAGE_NOT_FOUND) from None
    return sb.__version__


def check_version_string_is_valid(
    version_string: Optional[str], /, *, release: bool, source_error: MetadataErrorCode
) -> None:
    assert (
        source_error is MetadataErrorCode.PACKAGE_ERROR
        or source_error is MetadataErrorCode.SETUP_ERROR
    )

    if version_string is None:
        return

    release_version_pattern = re.compile(RELEASE_VERSION_REGEX)
    dev_version_pattern = re.compile(DEV_VERSION_REGEX)

    if release:
        if release_version_pattern.match(version_string) is None:
            error_code = MetadataErrorCode.RELEASE_VERSION_STRING_ERROR & source_error
            raise MetadataError(error_code, version_string=version_string)
    else:
        if dev_version_pattern.match(version_string) is None:
            error_code = MetadataErrorCode.DEV_VERSION_STRING_ERROR & source_error
            raise MetadataError(error_code, version_string=version_string)


def metadata_manager(release: bool, check_package_version: bool) -> None:
    setup = read_setup_file()
    setup_version = get_version_string_from_setup(setup)
    if check_package_version:
        package_version = get_version_string_from_package()
    else:
        package_version = None
    check_version_string_is_valid(
        setup_version, release=release, source_error=MetadataErrorCode.SETUP_ERROR
    )
    check_version_string_is_valid(
        package_version, release=release, source_error=MetadataErrorCode.PACKAGE_ERROR
    )
    if package_version is not None and setup_version != package_version:
        raise MetadataError(
            MetadataErrorCode.INCONSISTENT_VERSION_STRINGS,
            setup_version=setup_version,
            package_version=package_version,
        )
