"""Command-line interface: `uv run python -m src [options]`."""

from __future__ import annotations

import argparse
import os
from typing import Optional, Sequence

DEFAULT_FUNCTIONS_DEFINITION = os.path.join(
    "data",
    "input",
    "functions_definition.json"
)
DEFAULT_INPUT = os.path.join(
    "data",
    "input",
    "function_calling_tests.json"
)
DEFAULT_OUTPUT = os.path.join(
    "data",
    "output",
    "function_calling_results.json"
)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """
    Docstring for parse_args
    :param argv: Description
    :type argv: Optional[Sequence[str]]
    :return: Description
    :rtype: Namespace
    """
    parser = argparse.ArgumentParser(
        prog="python -m src",
        description=(
            "Translate natural language prompts"
            " into structured function calls "
            "using grammar-constrained "
            "decoding on a small LLM."
        ),
    )
    parser.add_argument(
        "--functions_definition",
        default=DEFAULT_FUNCTIONS_DEFINITION,
        help=("Path to the functions definition "
              f"JSON file (default: {DEFAULT_FUNCTIONS_DEFINITION})."),
    )
    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT,
        help=("Path to the input prompts"
              f" JSON file (default: {DEFAULT_INPUT})."),
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help=("Path to write the JSON"
              f" results to (default: {DEFAULT_OUTPUT})."),
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help=("Print full tracebacks and extra "
              "diagnostic info instead of short warnings."),
    )
    return parser.parse_args(argv)
