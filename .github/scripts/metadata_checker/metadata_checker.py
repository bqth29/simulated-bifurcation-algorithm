import calendar
import datetime
import re
from collections import defaultdict
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


def get_allowed_dates():
    today = date.today()
    yesterday = today - datetime.timedelta(days=1)
    tomorrow = today + datetime.timedelta(days=1)
    # the person doing the release might not be in the same timezone
    # as the server on which the GitHub actions are running
    allowed_dates = {yesterday, today, tomorrow}
    return allowed_dates


def check_citation_date(date_string, citation_file, line_nb):
    try:
        citation_date = datetime.date.fromisoformat(date_string)
    except ValueError:
        raise InvalidDateFormatError(citation_file, line_nb, date_string) from None
    allowed_dates = get_allowed_dates()
    if citation_date not in allowed_dates:
        allowed_dates = [date.isoformat for date in allowed_dates]
        raise WrongDateError(citation_file, line_nb, date_string, allowed_dates)
    return citation_date


def get_month_abbreviation(month_number):
    with calendar.different_locale(("en-US", None)) as encoding:
        abbreviation = calendar.month_abbr[month_number]
        if encoding is not None:
            abbreviation = abbreviation.decode(encoding)
        abbreviation = abbreviation.lower()
        return abbreviation


def check_bibtex_date(month, year, filename, line_nb, allowed_dates):
    allowed_dates = {
        (get_month_abbreviation(date.month), str(date.year)) for date in allowed_dates
    }
    if (month, year) not in allowed_dates:
        date = f"{month}, {year}"
        allowed_dates = [f"{month}, {year}" for month, year in allowed_dates]
        raise WrongDateError(filename, line_nb, date, allowed_dates)


def check_all_dates(dates):
    citation_date = None
    bibtex_date = None
    for date in dates:
        if date[1] == CITATION_FILE:
            citation_date = date
        elif date[1] == BIBTEX_FILE:
            bibtex_date = date
    errors = []
    if citation_date is None:
        allowed_dates = None
    else:
        try:
            citation_date = check_citation_date(*citation_date)
            allowed_dates = {citation_date}
        except (InvalidDateFormatError, WrongDateError) as error:
            errors.append(error)
            allowed_dates = None
    if bibtex_date is not None:
        if allowed_dates is None:
            allowed_dates = get_allowed_dates()
        try:
            check_bibtex_date(*bibtex_date, allowed_dates)
        except WrongDateError as error:
            errors.append(error)
    return errors


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
        date_errors = check_all_dates(variables["date"])
        errors.extend(date_errors)
    return errors
