from typing import Optional, Set

from config import (
    DATE_FORMAT,
    DEV_VERSION_REGEX,
    METADATA_CHECKER_INVOCATION,
    RELEASE_VERSION_REGEX,
)


class MetadataCheckerError(Exception):
    def __init_(
        self, filename: Optional[str], line_nb: Optional[int], error_message: str
    ):
        if filename is None:
            message = f"{self.__class__.__name__}:\n{error_message}"
        elif line_nb is None:
            message = (
                f'{self.__class__.__name__} in file "{filename}":\n{error_message}'
            )
        else:
            message = (
                f'{self.__class__.__name__} in file "{filename}" at line {line_nb}:\n'
                f"{error_message}"
            )
        super().__init__(message)


class BeginWithoutEndError(MetadataCheckerError):
    def __init__(self, filename: str, line_nb: int):
        error_message = 'Could not the find corresponding "end" command.'
        super().__init__(filename, line_nb, error_message)


class BlankLineError(MetadataCheckerError):
    def __init__(self, filename: str, line_nb: int):
        error_message = (
            "In this file, command lines should be preceded and followed by a blank "
            "line."
        )
        super().__init__(filename, line_nb, error_message)


class EndWithoutBeginError(MetadataCheckerError):
    def __init__(self, filename: str, line_nb: int):
        error_message = 'Could not the find corresponding "begin" command.'
        super().__init__(filename, line_nb, error_message)


class InvalidCommandError(MetadataCheckerError):
    def __init__(self, filename, line_nb, issue):
        if issue == "syntax":
            error_message = (
                "Commands should be of the form "
                f"{METADATA_CHECKER_INVOCATION}{{action}}{{arguments}}.\n"
                'To include "{" or "}" in the arguments, use "{{" or "}}".'
            )
        elif issue == "end file":
            error_message = "Command should be followed be the line it applies to."
        elif issue == "not matching":
            error_message = "The command and the line it applies to do not match."
        elif issue == "{":
            error_message = (
                "Could not find an opening curly bracket. "
                'To include "{" or "}" in the arguments, use "{{" or "}}".'
            )
        elif issue == "}":
            error_message = (
                "Could not find a closing curly bracket. "
                'To include "{" or "}" in the arguments, use "{{" or "}}".'
            )
        else:
            raise ValueError(f"Unknown issue {issue}.")
        super().__init__(filename, line_nb, error_message)


class InvalidDateFormatError(MetadataCheckerError):
    def __init__(self, filename, line_nb, date):
        error_message = (
            f'The date in the citation file should be in "{DATE_FORMAT}" '
            f'format, got "{date}".'
        )
        super().__init__(filename, line_nb, error_message)


class InvalidVersionError(MetadataCheckerError):
    REGEX = ""
    VERSION_TYPE = ""

    def __init__(self, filename: str, line_nb: int, version_string: str):
        error_message = (
            f"{self.VERSION_TYPE} version string should match the following regular "
            "expression.\n"
            f"{self.REGEX}\n"
            f'Received: "{version_string}".'
        )
        super().__init__(filename, line_nb, error_message)

    def __init_subclass__(cls, regex: str, version_type: str) -> None:
        super().__init_subclass__()
        self.REGEX = regex
        self.VERSION_TYPE = version_type.capitalize()


class InvalidDevVersionError(InvalidVersionError, DEV_VERSION_REGEX, "development"):
    pass


class InvalidReleaseVersionError(InvalidVersionError, RELEASE_VERSION_REGEX, "release"):
    pass


class MissingRequiredDefinitionError(MetadataCheckerError):
    def __init__(self, filename: str, missing_definitions: Set[str]):
        missing = ", ".join(sorted(missing_definitions))
        error_message = f"The file should define the following variables: {missing}."
        super().__init__(filename, None, error_message)


class MultipleDefinitionsError(MetadataCheckerError):
    def __init__(self, filename, lines, variable):
        error_message = (
            f'Variable "{variable}" is defined multiple times '
            f"(at lines {', '.join(lines)})."
        )
        super().__init__(filename, None, error_message)


class UnknownActionError(MetadataCheckerError):
    def __init_(self, filename: str, line_nb: int, action: str):
        error_message = f'Unknown action "{action}".'
        super().__init__(filename, line_nb, error_message)


class VersionStringsNotMatchingError(MetadataCheckerError):
    def __init__(self, versions):
        error_message = ["Version strings are not consistent."]
        for version_string, locations in versions.items():
            locations = [f'"{file}" at line {line_nb}' for file, line_nb in locations]
            locations = ", and ".join(locations)
            locations = f'"{version_string}" found in {locations}'
            error_message.append(locations)
        error_message = "\n".join(error_message)
        super().__init__(None, None, error_message)


class WrongDateError(MetadataCheckerError):
    def __init_(self, filename, line_nb, date, allowed_dates):
        if len(allowed_dates) > 1:
            text = f"Valid dates are {', and'.join(allowed_dates)}."
        else:
            text = f"The only valid date is {allowed_dates[0]}"
        error_message = (
            f'Date "{date}" does not math any valid date in "{DATE_FORMAT}".\n'
            f"{text}"
        )
        super().__init__(filename, line_nb, error_message)


__all__ = [
    "BeginWithoutEndError",
    "BlankLineError",
    "EndWithoutBeginError",
    "InvalidCommandError",
    "InvalidDateFormatError",
    "InvalidVersionError",
    "InvalidDevVersionError",
    "InvalidReleaseVersionError",
    "MetadataCheckerError",
    "MissingRequiredDefinitionError",
    "MultipleDefinitionsError",
    "UnknownActionError",
    "VersionStringsNotMatchingError",
    "WrongDateError",
]
