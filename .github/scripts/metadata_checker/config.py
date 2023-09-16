METADATA_CHECKER_INVOCATION = "!MDC"

FILES_SHOULD_DEFINE = {
    "CITATION.cff": ["date-released", "version"],
    "README.md": ["BibTeX"],
    "setup.py": ["version"],
    "src/simulated_bifurcation/__init__.py": ["version"],
}

RELEASE_VERSION_REGEX = r"^(([1-9][0-9]*)|0)\.(([1-9][0-9]*)|0)\.(([1-9][0-9]*)|0)$"
DEV_VERSION_REGEX = (
    r"^(([1-9][0-9]*)|0)\.(([1-9][0-9]*)|0)\.(([1-9][0-9]*)|0)"
    r"(\.dev(([1-9][0-9]*)|0))?$"
)


__all__ = [
    "DEV_VERSION_REGEX",
    "FILES_SHOULD_DEFINE",
    "METADATA_CHECKER_INVOCATION",
    "RELEASE_VERSION_REGEX",
]
