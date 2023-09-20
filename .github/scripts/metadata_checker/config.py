METADATA_CHECKER_INVOCATION = "!MDC"

CITATION_FILE = "CITATION.cff"
LICENSE_FILE = "LICENSE"
BIBTEX_FILE = "README.md"

FILES_SHOULD_DEFINE = {
    CITATION_FILE: ["date", "version"],
    LICENSE_FILE: ["date"],
    "setup.py": ["version"],
    "src/simulated_bifurcation/__init__.py": ["version"],
}
try:
    FILES_SHOULD_DEFINE[BIBTEX_FILE].append("BibTeX")
except KeyError:
    FILES_SHOULD_DEFINE[BIBTEX_FILE] = ["BibTeX"]

LICENSE_DATE_LINE = "Copyright (c) "

RELEASE_VERSION_REGEX = r"^(([1-9][0-9]*)|0)\.(([1-9][0-9]*)|0)\.(([1-9][0-9]*)|0)$"
DEV_VERSION_REGEX = (
    r"^(([1-9][0-9]*)|0)\.(([1-9][0-9]*)|0)\.(([1-9][0-9]*)|0)"
    r"(\.dev(([1-9][0-9]*)|0))?$"
)
