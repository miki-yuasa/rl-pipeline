"""Sphinx configuration for rl-pipeline documentation."""

project = "rl-pipeline"
author = "Mikihisa Yuasa"

extensions = []

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "alabaster"
html_static_path = ["_static"]
