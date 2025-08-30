#!/usr/bin/env bash
set -euo pipefail
pip install mkdocs-material mkdocstrings[python] mkdocs-jupyter
mkdocs serve -a 0.0.0.0:8000
