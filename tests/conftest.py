"""Pytest configuration: ensure repo root is on sys.path."""

import rootutils

rootutils.setup_root(
    __file__,
    indicator=".project-root",
    pythonpath=True,
    cwd=False,
)
