from typing import Optional, Set

from config import DEV_VERSION_REGEX, RELEASE_VERSION_REGEX


class MetadataCheckerError(Exception):
    def __init_(
        self, filename: Optional[str], line_nb: Optional[int], error_message: str
    ):
        if filename is None:
            message = f"{self.__class__.__name__}:\n{error_message}"
        elif line_nb is not None:
            message = (
                f'{self.__class__.__name__} in file "{filename}" at line {line_nb}:\n'
                f"{error_message}"
            )
        else:
            message = (
                f'{self.__class__.__name__} in file "{filename}":\n{error_message}'
            )
        super().__init__(message)


class BeginWithoutEndError(MetadataCheckerError):
    def __init__(self, filename: str, line_nb: int):
        error_message = 'Could not the find corresponding "end" command .'
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
    pass


class UnknownActionError(MetadataCheckerError):
    def __init_(self, filename: str, line_nb: int, action: str):
        error_message = f'Unknown action "{action}".'
        super().__init__(filename, line_nb, error_message)


class VersionStringsNotMatchingError(MetadataCheckerError):
    def __init__(self, versions):
        message = ["Version strings are not consistent."]
        for version_string, locations in versions.items():
            locations = [f'"{file}" at line {line_nb}' for file, line_nb in locations]
            locations = ", and ".join(locations)
            locations = f'"{version_string}" found in {locations}'
            message.append(locations)
        message = "\n".join(message)
        super().__init__(None, None, message)


class WrongDateError(MetadataCheckerError):
    pass


__all__ = [
    "BeginWithoutEndError",
    "BlankLineError",
    "EndWithoutBeginError",
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
