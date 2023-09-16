import calendar
import re
from collections import defaultdict
from datetime import date, timedelta
from typing import Dict, Iterator, List, Tuple, Union

from config import *
from errors import *


def parse_citation_file(content: List[str]) -> Dict[str, List[Tuple[str, int]]]:
    definitions = [("version:", "version"), ("date-released:", "date")]
    variables = defaultdict(list)
    for field, var_name in definitions:
        for line_nb, line in enumerate(content):
            if line.startswith(field):
                value = line[len(field) :].strip()
                variables[var_name].append((value, line_nb))
    return variables


def skip_blank_line(
    lines: List[str], line_nb: int, enumerate_lines: Iterator[Tuple[int, str]]
) -> None:
    if (
        line_nb == 0
        or line_nb == len(lines) - 1
        or lines[line_nb - 1].strip() != ""
        or lines[line_nb + 1].strip() != ""
    ):
        raise BlankLineError
    next(enumerate_lines)


def action_set(lines, line_nb, variables):
    pass


def parse_file(lines, is_markdown):
    variables = {}
    errors = []
    enumerate_lines = enumerate(lines)
    while True:
        try:
            line_nb, line = next(enumerate_lines)
        except StopIteration:
            break
        index = line.find(METADATA_CHECKER_INVOCATION)
        if index == -1:
            continue
        if is_markdown:
            skip_blank_line(lines, line_nb, enumerate_lines)
        action, args = parse_command(line[index:])
        if action == "set":
            pass
        elif action == "begin":
            pass
        elif action == "end":
            raise EndNoBeginError
        else:
            pass
    return errors


def read_file(
    filename: str,
) -> Tuple[
    Dict[str, Tuple[str, int]], List[Union[FileNotFoundError, MetadataCheckerError]]
]:
    try:
        with open(filename, "r", encoding="utf-8") as file:
            file_content = file.readlines()
    except FileNotFoundError as error:
        return {}, [error]
    if filename.endswith("CITATION.cff"):
        variables = parse_citation_file(file_content)
        errors = []
    else:
        is_markdown = path.endswith(".md")
        variables, errors = parse_file(file_content, is_markdown)
    return variables, errors


def check_should_define(filename, variables, should_define):
    defined = set(variables.keys())
    missing = set(should_define).difference(defined)
    if not missing:
        raise MissingRequiredDefinitionError(filename, missing)


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
