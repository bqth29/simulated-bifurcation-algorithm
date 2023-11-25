import calendar
import datetime
import re
from collections import defaultdict
from typing import Callable, Dict, Iterator, List, Set, Tuple, Union

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


def parse_license_file(content: List[str]) -> Dict[str, List[Tuple[str, int]]]:
    variables = defaultdict(list)
    for line_nb, line in enumerate(content):
        if line.strip().startswith(LICENSE_DATE_LINE):
            line = line[len(LICENSE_DATE_LINE) :]
            year = line.split()[0]
            variables["date"].append((year, line_nb))
    return variables


def skip_blank_line(
    lines: List[str],
    line_nb: int,
    enumerate_lines: Iterator[Tuple[int, str]],
    filename: str,
) -> None:
    if line_nb == len(lines) or lines[line_nb] != "":
        raise BlankLineError(filename, line_nb)
    next(enumerate_lines)
    if line_nb == 1 or lines[line_nb - 2] != "":
        raise BlankLineError(filename, line_nb)


def parse_command(text: str, filename: str, line_nb: int) -> Tuple[str, str]:
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


def simultaneous_read(
    args: str,
    line: str,
    curly: str,
    iterator: Callable[[str], Iterator[str]],
    error_factory: Callable[[str], InvalidCommandError],
) -> Tuple[str, str]:
    args = iterator(args)
    line = iterator(line)
    prev_curly = False
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
    args = "".join(iterator(char + "".join(args)))
    line = "".join(iterator("".join(line)))
    return args, line


def action_set(
    enumerate_lines: Iterator[Tuple[int, str]],
    args: str,
    variables: Dict[str, List[Tuple[str, int]]],
    filename: str,
) -> None:
    line_nb, line = next(enumerate_lines)

    def error_factory(issue: str) -> InvalidCommandError:
        return InvalidCommandError(filename, line_nb, issue)

    args, line = simultaneous_read(args, line, "{", iter, error_factory)
    args, line = simultaneous_read(args, line, "}", reversed, error_factory)
    variables[args].append((line, line_nb))


def action_begin(
    lines: List[str],
    enumerate_lines: Iterator[Tuple[int, str]],
    begin_args: str,
    variables: Dict[str, List[Tuple[str, int]]],
    is_markdown: bool,
    filename: str,
) -> None:
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
        line_nb += 1
        content.pop()
    content = "\n".join(content)
    variables[begin_args].append((content, line_nb))


def parse_file(
    lines: List[str], filename: str
) -> Tuple[Dict[str, List[Tuple[str, int]]], List[MetadataCheckerError]]:
    is_markdown = filename.endswith(".md")
    variables = defaultdict(list)
    errors = []
    enumerate_lines = enumerate(lines, 1)
    while True:
        try:
            line_nb, line = next(enumerate_lines)
        except StopIteration:
            break
        index = line.find(METADATA_CHECKER_INVOCATION)
        if index == -1:
            continue
        if is_markdown:
            try:
                skip_blank_line(lines, line_nb, enumerate_lines, filename)
            except BlankLineError as error:
                errors.append(error)
        command_start = index + len(METADATA_CHECKER_INVOCATION)
        action, args = parse_command(line[command_start:], filename, line_nb)
        if action == "set":
            try:
                action_set(enumerate_lines, args, variables, filename)
            except StopIteration:
                error = InvalidCommandError(filename, line_nb, "end file")
                errors.append(error)
                break
            except InvalidCommandError as error:
                errors.append(error)
        elif action == "begin":
            try:
                action_begin(
                    lines, enumerate_lines, args, variables, is_markdown, filename
                )
            except StopIteration:
                error = BeginWithoutEndError(filename, line_nb)
                errors.append(error)
                break
            except (InvalidCommandError, BlankLineError) as error:
                errors.append(error)
        elif action == "end":
            raise EndWithoutBeginError(filename, line_nb)
        else:
            raise UnknownActionError(filename, line_nb, action)
    return variables, errors


def parse_bibtex(variables: Dict[str, List[Tuple[str, int]]]) -> None:
    for bibtex, line_nb in variables["BibTeX"]:
        year = None
        month = None
        # use line number from begin command, position inside BibTeX might be off
        # due to blank line skip in markdown
        for line in bibtex.splitlines():
            if line.startswith("version = {"):
                version = line[len("version = {") : -2]
                variables["version"].append((version, line_nb))
            if line.startswith("year = {"):
                year = line[len("year = {") : -2]
            elif line.startswith("month = "):
                month = line[len("month = ") : -1]
        if month is not None and year is not None:
            date = f"{year} {month}"
            variables["date"].append((date, line_nb))


def read_file(
    filename: str,
) -> Tuple[Dict[str, List[Tuple[str, int]]], List[MetadataCheckerError]]:
    with open(filename, "r", encoding="utf-8") as file:
        file_content = file.readlines()
    if file_content[-1].endswith(("\r", "\n")):
        file_content.append("")
    file_content = list(map(str.strip, file_content))
    if filename == CITATION_FILE:
        variables = parse_citation_file(file_content)
        errors = []
    elif filename == LICENSE_FILE:
        variables = parse_license_file(file_content)
        errors = []
    else:
        variables, errors = parse_file(file_content, filename)
        if "BibTeX" in variables:
            parse_bibtex(variables)
    return variables, errors


def check_should_define(
    filename: str, variables: Dict[str, List[Tuple[str, int]]], should_define: List[str]
):
    defined = set(variables.keys())
    missing = set(should_define).difference(defined)
    if missing:
        raise MissingRequiredDefinitionError(filename, missing)


def unwrap_variables(
    variables: Dict[str, List[Tuple[str, str, int]]],
    file_variables: Dict[str, List[Tuple[str, int]]],
    filename: str,
) -> List[MultipleDefinitionsError]:
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


def version_strings_are_consistent(
    version_strings: List[Tuple[str, str, int]], release: bool
):
    versions = defaultdict(list)
    for version, filename, line_nb in version_strings:
        if release or filename not in [CITATION_FILE, BIBTEX_FILE]:
            versions[version].append((filename, line_nb))
    if len(versions) >= 2:
        raise VersionStringsNotMatchingError(versions)


def check_all_version_strings(
    version_strings: List[Tuple[str, str, int]], release: bool
) -> List[MetadataCheckerError]:
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


def get_allowed_dates() -> Set[datetime.date]:
    today = datetime.date.today()
    yesterday = today - datetime.timedelta(days=1)
    tomorrow = today + datetime.timedelta(days=1)
    # the person doing the release might not be in the same timezone
    # as the server on which the GitHub actions are running
    allowed_dates = {yesterday, today, tomorrow}
    return allowed_dates


def check_citation_date(
    date_string: str, citation_file: str, line_nb: int
) -> datetime.date:
    try:
        citation_date = datetime.date.fromisoformat(date_string)
    except ValueError:
        raise InvalidDateFormatError(citation_file, line_nb, date_string) from None
    allowed_dates = get_allowed_dates()
    if citation_date not in allowed_dates:
        allowed_dates = [date.isoformat() for date in sorted(allowed_dates)]
        raise WrongDateError(
            citation_file, line_nb, date_string, allowed_dates, "yyyy-mm-dd"
        )
    return citation_date


def check_license_date(
    year: str, license_file: str, line_nb: int, allowed_dates: Set[datetime.date]
):
    allowed_dates = {str(date.year) for date in allowed_dates}
    if year not in allowed_dates:
        allowed_dates = list(allowed_dates)
        raise WrongDateError(license_file, line_nb, year, allowed_dates, "year (yyyy)")


def get_month_abbreviation(month_number: int) -> str:
    with calendar.different_locale(("en_US", "utf-8")) as encoding:
        abbreviation = calendar.month_abbr[month_number]
        if encoding is not None:
            abbreviation = abbreviation.decode(encoding)
        abbreviation = abbreviation.lower()
        return abbreviation


def check_bibtex_date(
    date: str, filename: str, line_nb: int, allowed_dates: Set[datetime.date]
):
    year, month = date.split()
    allowed_dates = {
        (get_month_abbreviation(date.month), str(date.year)) for date in allowed_dates
    }
    if (month, year) not in allowed_dates:
        date = f"{month}, {year}"
        allowed_dates = [f"{month}, {year}" for month, year in allowed_dates]
        raise WrongDateError(
            filename,
            line_nb,
            date,
            allowed_dates,
            "standard month abbreviation, year (yyyy)",
        )


def check_all_dates(dates: List[Tuple[str, str, int]]) -> List[MetadataCheckerError]:
    citation_date = None
    license_date = None
    bibtex_date = None
    for date in dates:
        if date[1] == CITATION_FILE:
            citation_date = date
        elif date[1] == LICENSE_FILE:
            license_date = date
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
    if allowed_dates is None:
        allowed_dates = get_allowed_dates()
    if license_date is not None:
        try:
            check_license_date(*license_date, allowed_dates=allowed_dates)
        except WrongDateError as error:
            errors.append(error)
    if bibtex_date is not None:
        try:
            check_bibtex_date(*bibtex_date, allowed_dates=allowed_dates)
        except WrongDateError as error:
            errors.append(error)
    return errors


def metadata_checker(
    release: bool,
) -> List[Union[FileNotFoundError, MetadataCheckerError]]:
    errors = []
    variables = defaultdict(list)
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
