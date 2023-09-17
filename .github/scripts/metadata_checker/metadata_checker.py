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
    lines: List[str],
    line_nb: int,
    enumerate_lines: Iterator[Tuple[int, str]],
    filename: str,
) -> None:
    if (
        line_nb == 0
        or line_nb == len(lines) - 1
        or lines[line_nb - 1].strip() != ""
        or lines[line_nb + 1].strip() != ""
    ):
        raise BlankLineError(filename, line_nb)
    next(enumerate_lines)


def parse_command(text, filename, line_nb):
    end_action = text.find("}")
    if text[0] != "{" or end_action == -1:
        raise InvalidCommandError(filename, line_nb, "syntax")
    action = text[1:end_action]
    text = text[end_action + 1 :]
    end_args = text.rfind("}")
    if text[0] != "{" or end_args == -1:
        raise InvalidCommandError(filename, line_nb, "syntax")
    args = text[1:end_args]
    return action, args


def simultaneous_read(args, line, curly, iterator, error_factory):
    args = iterator(args)
    line = iterator(line)
    prev_curly = curly
    try:
        while True:
            char = next(args)
            if char == curly:
                if prev_curly:
                    prev_curly = False
                else:
                    prev_curly = True
                    continue
            elif prev_curly:
                break
            if char != next(line):
                raise error_factory("not matching")
    except StopIteration:
        raise error_factory(curly)
    args = "".join(iterator("".join(args)))
    line = "".join(iterator("".join(line)))
    return args, line


def action_set(enumerate_lines, args, variables, filename):
    line_nb, line = next(enumerate_lines)

    def error_factory(issue):
        return InvalidCommandError(filename, line_nb, issue)

    args, line = simultaneous_read(args, line, "{", iter, error_factory)
    args, line = simultaneous_read(args, line, "}", reversed, error_factory)
    variables[args].append((line, line_nb))


def action_begin(enumerate_lines, begin_args, variables, is_markdown, filename):
    content = []
    while True:
        line_nb, line = next(enumerate_lines)
        index = line.find(METADATA_CHECKER_INVOCATION)
        if index == -1:
            content.append(line)
        else:
            break
    command_start = index + len(METADATA_CHECKER_INVOCATION)
    action, args = parse_command(line[command_start:], filename, line_nb)
    if action != "end" or args != begin_args:
        raise InvalidCommandError
    if is_markdown:
        skip_blank_line(lines, line_nb, enumerate_lines, filename)
        content.pop()
    content = "".join(content)
    variables[begin_args].append((content, line_nb))


def parse_file(lines, filename):
    is_markdown = filename.endswith(".md")
    variables = defaultdict(list)
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
            skip_blank_line(lines, line_nb, enumerate_lines, filename)
        command_start = index + len(METADATA_CHECKER_INVOCATION)
        action, args = parse_command(line[command_start:], filename, line_nb)
        if action == "set":
            try:
                action_set(enumerate_lines, args, variables, filename)
            except StopIteration:
                error = InvalidCommandError(filename, line_nb, "end file")
                errors.append(error)
                break
            except (InvalidCommandError, BlankLineError) as error:
                errors.append(error)
        elif action == "begin":
            try:
                action_begin(enumerate_lines, args, variables, is_markdown, filename)
            except StopIteration:
                error = BeginWithoutEndError(filename, line_nb)
                errors.append(error)
                break
            except InvalidCommandError as error:
                errors.append(error)
        elif action == "end":
            raise EndWithoutBeginError(filename, line_nb)
        else:
            raise UnknownActionError(filename, line_nb, action)
    return variables, errors


def parse_bibtex(variables):
    for bibtex, line_nb in variables["BibTeX"]:
        year = None
        month = None
        # use line number from begin command, position inside BibTeX might be off
        # due to blank line skip in markdown
        for line in bibtex.splitlines():
            line = line.strip()
            if line.startswith("version = {"):
                version = line[len("version = {") : -2]
                variables["version"].append((version, line_nb))
            if line.startswith("year = {"):
                year = line[len("year = {") : -2]
            elif line.startswith("month = "):
                month = line[len("month = ") : -1]
        if month is not None and year is not None:
            date = (year, month)
            variables["date"].append((date, line_nb))


def read_file(
    filename: str,
) -> Tuple[Dict[str, Tuple[str, int]], List[MetadataCheckerError]]:
    with open(filename, "r", encoding="utf-8") as file:
        file_content = file.readlines()
    if filename.endswith("CITATION.cff"):
        variables = parse_citation_file(file_content)
        errors = []
    else:
        variables, errors = parse_file(file_content, filename)
        if "BibTeX" in variables:
            parse_bibtex(variables)
    return variables, errors


def check_should_define(filename, variables, should_define):
    defined = set(variables.keys())
    missing = set(should_define).difference(defined)
    if not missing:
        raise MissingRequiredDefinitionError(filename, missing)


def unwrap_variables(variables, file_variables, filename):
    errors = []
    for variable, values in file_variables.items():
        if len(values) >= 2:
            lines = [line_nb for _, line_nb in values]
            error = MultipleDefinitionsError(filename, lines, variable)
            errors.append(error)
        for value, line_nb in values:
            variable_info = (value, filename, line_nb)
            variables[variable].append(variable_info)
    return errors


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
    versions = defaultdict(list)
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
        allowed_dates = [date.isoformat() for date in allowed_dates]
        raise WrongDateError(citation_file, line_nb, date_string, allowed_dates)
    return citation_date


def get_month_abbreviation(month_number):
    with calendar.different_locale(("en-US", None)) as encoding:
        abbreviation = calendar.month_abbr[month_number]
        if encoding is not None:
            abbreviation = abbreviation.decode(encoding)
        abbreviation = abbreviation.lower()
        return abbreviation


def check_bibtex_date(date, filename, line_nb, allowed_dates):
    year, month = date
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
        try:
            file_variables, read_file_errors = read_file(filename)
            errors.extend(read_file_errors)
        except FileNotFoundError as error:
            errors.append(error)
            continue
        try:
            check_should_define(filename, file_variables, should_define)
        except MissingRequiredDefinitionError as error:
            errors.append(error)
        multiple_definitions = unwrap_variables(variables, file_variables, filename)
        errors.extend(multiple_definitions)
    version_errors = check_all_version_strings(variables["version"], release)
    errors.extend(version_errors)
    if release:
        date_errors = check_all_dates(variables["date"])
        errors.extend(date_errors)
    return errors
