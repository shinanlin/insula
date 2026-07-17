#!/usr/bin/env python3
"""CLI entry point for corrected functional NMF brain maps."""

from src.nmf.render_brain import parse_args, render


def main() -> None:
    render(parse_args())


if __name__ == "__main__":
    main()
