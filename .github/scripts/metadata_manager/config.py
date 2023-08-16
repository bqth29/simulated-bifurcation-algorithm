METADATA_MANAGER_INVOCATION = "!MDM"

FILES_SHOULD_DEFINE = {
    "setup.py": ["version"],
    "src/simulated_bifurcation/__init__.py": ["version"],
}

RELEASE_VERSION_REGEX = r"^(([1-9][0-9]*)|0)\.(([1-9][0-9]*)|0)\.(([1-9][0-9]*)|0)$"
DEV_VERSION_REGEX = (
    r"^(([1-9][0-9]*)|0)\.(([1-9][0-9]*)|0)\.(([1-9][0-9]*)|0)"
    r"(\.dev(([1-9][0-9]*)|0))?$"
)

BIBTEX_TEMPLATE = """\
@software{{{first_alphabetical_author}_Simulated_Bifurcation_SB_{year},
    author = {{{authors}}},
    month = {month},
    title = {{{{{title}}}}},
    url = {{{url}}},
    version = {{{version}}},
    year = {{{year}}},
}}\
"""
