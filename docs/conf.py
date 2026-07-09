import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

project = "MultiChat"
author = "Caiwei Zhen"
copyright = "2026, Caiwei Zhen"
release = "0.2.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "nbsphinx",
    "nbsphinx_link",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_theme_options = {
    "navigation_depth": 2,
    "collapse_navigation": False,
}

html_context = {
    "display_github": True,
    "github_user": "zcaiwei",
    "github_repo": "MultiChat",
    "github_version": "main",
    "conf_py_path": "/docs/",
}

nbsphinx_execute = "never"
nbsphinx_allow_errors = True
nbsphinx_timeout = 600
