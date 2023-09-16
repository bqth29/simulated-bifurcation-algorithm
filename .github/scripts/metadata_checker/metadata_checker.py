import calendar
import re
from collections import defaultdict
from datetime import date, timedelta
from typing import Dict, Iterator, List, Tuple, Union

from config import *
from errors import *

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



def get_version_string_from_package() -> str:
    try:
        import simulated_bifurcation as sb
    except ModuleNotFoundError:
        raise MetadataError(MetadataErrorCode.PACKAGE_NOT_FOUND) from None
    return sb.__version__



def version_string_is_valid(
    version_string: str, /, filename: str, line_nb: int, *, release: bool
) -> None:
    release_version_pattern = re.compile(RELEASE_VERSION_REGEX)
    dev_version_pattern = re.compile(DEV_VERSION_REGEX)
    if release:
        if release_version_pattern.match(version_string) is None:
            raise InvalidReleaseVersionError(filename, line_nb, version_string)
    else:
        if dev_version_pattern.match(version_string) is None:
            error_code = MetadataErrorCode.DEV_VERSION_STRING_ERROR & source_error
            raise MetadataError(error_code, version_string=version_string)
            raise InvalidDevVersionError(filename, line_nb, version_string)


def version_strings_are_consistent(version_strings, release):
    versions = defaultdict[List]
    for version, filename, line_nb in version_strings:
        if release or filename not in ["CITATION.cff", "README.md"]:
            versions[version].append(filename, line_nb)
    if len(versions) >= 2:
        raise VersionStringsNotMatchingError(versions)


def check_all_version_strings(version_strings, release):
    errors = []
    for version_string, filename, line_nb in version_strings:
        try:
            version_string_is_valid(version_string, filename, line_nb, release=release)
        except InvalidVersionError as error:
            errors.append(error)
    try:
        version_strings_are_consistent(version_strings, release)
    except VersionStringsNotMatchingError as error:
        errors.append(error)
    return errors


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
def metadata_checker(
    release: bool,
) -> List[Union[FileNotFoundError, MetadataCheckerError]]:
    errors = []
    variables = {}
    for filename, should_define in FILES_SHOULD_DEFINE.items():
        file_variables, read_file_errors = read_file(filename)
        errors.extend(read_file_errors)
        try:
            check_should_define(filename, file_variables, should_define)
        except MissingRequiredDefinitionError as error:
            errors.append(error)
        multiple_definitions = unwrap_variables(variables, file_variables)
        errors.extend(multiple_definitions)
    version_errors = check_all_version_strings(variables["version"], release)
    errors.extend(version_errors)
    if release:
        date_errors = check_dates(variables["date"])
        errors.extend(date_errors)
    return errors
