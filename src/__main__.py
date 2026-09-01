from __future__ import annotations

import sys
import traceback
from typing import Optional, Sequence

from .cli import parse_args
from .io_utils import ProjectIOError
from .pipeline import run
from .schema import SchemaError

try:
    from llm_sdk import Small_LLM_Model
except ImportError as exc:
    print(
        "[fatal] Could not import Small_LLM_Model from llm_sdk. "
        "Make sure the llm_sdk/ package is installed"
        " (see README / `make install`).\n"
        f"Original error: {exc}",
        file=sys.stderr,
    )
    sys.exit(1)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """
    Docstring for main
    :param argv: Description
    :type argv: Optional[Sequence[str]]
    :return: Description
    :rtype: int
    """
    args = parse_args(argv)

    try:
        model = Small_LLM_Model()
    except Exception as exc:
        print(f"[fatal] Could not initialize the LLM model: {exc}",
              file=sys.stderr)
        if args.debug:
            traceback.print_exc(file=sys.stderr)
        return 1

    try:
        run(
            model=model,
            functions_path=args.functions_definition,
            input_path=args.input,
            output_path=args.output,
            debug=args.debug,
        )
    except (ProjectIOError, SchemaError) as exc:
        print(f"[fatal] {exc}", file=sys.stderr)
        if args.debug:
            traceback.print_exc(file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"[fatal] Unexpected error: {exc}", file=sys.stderr)
        if args.debug:
            traceback.print_exc(file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
