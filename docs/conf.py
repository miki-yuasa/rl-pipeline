"""Sphinx configuration for rl-pipeline documentation."""

from __future__ import annotations

import os
import sys
from importlib.metadata import PackageNotFoundError, version as pkg_version

sys.path.insert(0, os.path.abspath(".."))
DOCS_DIR = os.path.dirname(__file__)

project = "rl-pipeline"
copyright = "2026, Mikihisa Yuasa"
author = "Mikihisa Yuasa"

try:
    release = pkg_version("rl-pipeline")
except PackageNotFoundError:
    release = "0.1.0"
version = release

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.githubpages",
]

autosummary_generate = True
autodoc_typehints = "description"
autodoc_member_order = "bysource"
napoleon_numpy_docstring = True
napoleon_google_docstring = False
napoleon_include_special_with_doc = True
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_attr_annotations = False

autodoc_default_options = {
    "members": True,
    "show-inheritance": True,
    "member-order": "bysource",
}

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "gymnasium": ("https://gymnasium.farama.org", None),
    "optuna": ("https://optuna.readthedocs.io/en/stable", None),
}

templates_path = ["_templates"] if os.path.isdir(os.path.join(DOCS_DIR, "_templates")) else []
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "furo"
html_static_path = ["_static"] if os.path.isdir(os.path.join(DOCS_DIR, "_static")) else []
