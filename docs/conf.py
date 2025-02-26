import datetime
import os
import sys

sys.path.insert(0, os.path.abspath("../src"))

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "Simulated Bifurcation"
copyright = f"{datetime.datetime.now().year}, Romain Ageron, Thomas Bouquet and Lorenzo Pugliese"
author = "Romain Ageron, Thomas Bouquet and Lorenzo Pugliese"
release = "2.0.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "numpydoc",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    "sphinx.ext.doctest",
    "sphinx.ext.napoleon",
    "sphinx.ext.todo",
    "sphinx.ext.intersphinx",
    "myst_parser",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
html_static_path = ["_static"]

myst_enable_extensions = ["amsmath", "colon_fence", "dollarmath", "attrs_inline"]
myst_heading_anchors = 6


html_theme_options = {
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/bqth29/simulated-bifurcation-algorithm",
            "icon": "fab fa-github-square",
        }
    ],
    # the following 3 lines enable edit button
    "source_repository": "https://github.com/bqth29/simulated-bifurcation-algorithm/",
    "source_branch": "main",
    "source_directory": "docs/",
}

html_title = "Simulated Bifurcation"
html_short_title = "Simulated Bifurcation"
