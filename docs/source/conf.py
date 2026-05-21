# Configuration file for the Sphinx documentation builder.

import os
import sys

sys.path.insert(0, os.path.abspath("../.."))

project = "MultiChat"
copyright = "2026, Caiwei Zhen"
author = "Caiwei Zhen"

release = "0.1.0"
version = "0.1.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "nbsphinx",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

source_suffix = ".rst"
master_doc = "index"

nbsphinx_execute = "never"
nbsphinx_allow_errors = True

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_logo = "_static/MultiChat_logo.png"
html_theme_options = {
    "logo_only": False,
    "display_version": True,
}

epub_show_urls = "footnote"
