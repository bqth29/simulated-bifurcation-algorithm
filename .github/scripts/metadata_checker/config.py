METADATA_CHECKER_INVOCATION = "!MDC"

BIBTEX_FILE = "README.md"
CITATION_FILE = "CITATION.cff"
LICENSE_FILE = "LICENSE"

FILES_SHOULD_DEFINE = {
    CITATION_FILE: ["date", "version"],
    LICENSE_FILE: ["date"],
    "README.md": ["BibTeX"],
    "setup.py": ["version"],
    "src/simulated_bifurcation/__init__.py": ["version"],
}

LICENSE_DATE_LINE = "Copyright (c) "

RELEASE_VERSION_REGEX = r"^(([1-9][0-9]*)|0)\.(([1-9][0-9]*)|0)\.(([1-9][0-9]*)|0)$"
DEV_VERSION_REGEX = (
    r"^(([1-9][0-9]*)|0)\.(([1-9][0-9]*)|0)\.(([1-9][0-9]*)|0)"
    r"(\.dev(([1-9][0-9]*)|0))?$"
)
