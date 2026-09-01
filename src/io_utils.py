"""File I/O helpers with defensive error handling.

Every function here is designed to fail with a clear, actionable error
message (via ``ProjectIOError``) rather than letting a raw traceback leak to
the user, since the subject explicitly requires graceful handling of missing
files and malformed JSON.
"""

from __future__ import annotations

import json
import os
from typing import Any


class ProjectIOError(Exception):
    """Raised for any recoverable file/JSON problem with a human message."""


def load_json(path: str, *, description: str) -> Any:
    """Load and parse a JSON file,
    raising ProjectIOError with context on failure.

    Handles:
      * missing file / wrong path
      * directory passed instead of a file
      * empty file
      * malformed JSON (trailing commas, comments, truncation, etc.)
      * permission errors
      * non-UTF-8 encoded files
    """
    if not os.path.exists(path):
        raise ProjectIOError(f"{description} not found: '{path}'.")
    if os.path.isdir(path):
        raise ProjectIOError(f"{description} "
                             f"path is a directory, not a file: '{path}'.")

    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = f.read()
    except PermissionError as exc:
        raise ProjectIOError(f"{description}"
                             " could not be read "
                             f"(permission denied): '{path}'.") from exc
    except UnicodeDecodeError as exc:
        raise ProjectIOError(f"{description}"
                             f" is not valid UTF-8 text: '{path}'.") from exc
    except OSError as exc:
        raise ProjectIOError(f"{description}"
                             f" could not be read: '{path}' ({exc}).") from exc

    if not raw.strip():
        raise ProjectIOError(f"{description} is empty: '{path}'.")

    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ProjectIOError(
            f"{description} contains invalid JSON: '{path}' "
            f"(line {exc.lineno}, column {exc.colno}: {exc.msg})."
        ) from exc


def write_json(path: str, data: Any) -> None:
    """Write ``data`` as pretty-printed JSON,
      creating parent dirs as needed."""
    parent = os.path.dirname(os.path.abspath(path))
    try:
        os.makedirs(parent, exist_ok=True)
    except OSError as exc:
        raise ProjectIOError("Could not create "
                             f"output directory '{parent}': {exc}.") from exc

    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            f.write("\n")
    except OSError as exc:
        raise ProjectIOError("Could not write "
                             f"output file '{path}': {exc}.") from exc
