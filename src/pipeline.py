"""End-to-end pipeline: prompts in -> structured function calls out."""

from __future__ import annotations

import json
import sys
import traceback
from typing import Any, Dict, List

from .constrained_decoder import ConstrainedDecoder
from .io_utils import load_json, write_json
from .schema import FunctionDef, SchemaError, parse_functions_definition


def _build_context_text(prompt: str, functions: List[FunctionDef]) -> str:
    """
    Docstring for _build_context_text
    :param prompt: Description
    :type prompt: str
    :param functions: Description
    :type functions: List[FunctionDef]
    :return: Description
    :rtype: str
    """
    functions_json = json.dumps([f.to_prompt_dict() for f in functions],
                                indent=2)
    return (
        "You are a function calling assistant. "
        "Given a user request and a list "
        "of available functions, decide "
        "which function to call and with which "
        "arguments.\n\n"
        f"Available functions:\n{functions_json}\n\n"
        f"User request: {prompt}\n\n"
        "Respond with the function call as JSON.\n"
        "JSON:\n"
    )


def _default_value_for(param_type: str) -> Any:
    """
    Docstring for _default_value_for
    :param param_type: Description
    :type param_type: str
    :return: Description
    :rtype: Any
    """
    return {"string": "",
            "number": 0,
            "integer": 0,
            "boolean": False}[param_type]


def process_prompt(
    decoder: ConstrainedDecoder,
    prompt: str,
    functions: List[FunctionDef],
    functions_by_name: Dict[str, FunctionDef],
    *,
    debug: bool = False,
) -> Dict[str, Any]:
    """Run constrained generation for a single test prompt.

    Never raises: any internal failure is caught and results in a best-effort
    but still schema-valid fallback entry, so a single bad/edge-case prompt
    can never abort the whole batch or corrupt the output file.
    """
    try:
        context_text = _build_context_text(prompt, functions)
        ids = decoder.encode_literal(context_text)

        ids += decoder.encode_literal('{"name": "')
        print("    choosing function...", file=sys.stderr, flush=True)
        name, name_ids = decoder.generate_choice(ids,
                                                 [f.name for f in functions]
                                                 )
        ids += name_ids
        ids += decoder.encode_literal('"')
        print(f"    -> {name}", file=sys.stderr, flush=True)

        func = functions_by_name[name]
        ids += decoder.encode_literal(', "parameters": {')

        parameters: Dict[str, Any] = {}
        param_items = list(func.parameters.items())
        for i, (pname, pdef) in enumerate(param_items):
            ids += decoder.encode_literal(f'"{pname}": ')
            print(f"    generating '{pname}'"
                  f" ({pdef.type})...",
                  file=sys.stderr,
                  flush=True)

            if pdef.type == "boolean":
                choice, val_ids = decoder.generate_choice(ids,
                                                          ["true", "false"])
                ids += val_ids
                parameters[pname] = choice == "true"

            elif pdef.type in ("number", "integer"):
                text, val_ids = decoder.generate_number(
                    ids, integer_only=(pdef.type == "integer")
                )
                ids += val_ids
                try:
                    if pdef.type == "integer":
                        parameters[pname] = int(text)
                    else:
                        parameters[pname] = float(text)
                except ValueError:
                    parameters[pname] = _default_value_for(pdef.type)

            elif pdef.type == "string":
                ids += decoder.encode_literal('"')
                text, val_ids = decoder.generate_string(ids)
                ids += val_ids
                ids += decoder.encode_literal('"')
                parameters[pname] = text

            else:  # pragma: no cover - guarded upstream by schema validation
                parameters[pname] = _default_value_for("string")

            print(f"    -> {pname} = {parameters[pname]!r}",
                  file=sys.stderr, flush=True)

            if i < len(param_items) - 1:
                ids += decoder.encode_literal(", ")

        ids += decoder.encode_literal("}}")

        return {"prompt": prompt, "name": func.name, "parameters": parameters}

    except Exception as exc:
        print("[warning] Falling back for prompt"
              f" {prompt!r}: {exc}", file=sys.stderr)
        if debug:
            traceback.print_exc(file=sys.stderr)
        fallback_func = functions[0]
        fallback_params = {p.name: _default_value_for(p.type)
                           for p in fallback_func.parameters.values()}
        return {"prompt": prompt,
                "name": fallback_func.name,
                "parameters": fallback_params}


def run(model: Any,
        functions_path: str,
        input_path: str,
        output_path: str, *,
        debug: bool = False) -> None:
    """Load inputs, run constrained generation for every prompt, write output.

    Raises ProjectIOError / SchemaError (caught by the CLI entry point) for
    problems that make it impossible to proceed at all (unreadable/invalid
    functions_definition.json, unreadable input file, ...).
    """
    if debug:
        print(f"[debug] functions_definition"
              f" = '{functions_path}'", file=sys.stderr)
        print(f"[debug] input = '{input_path}'", file=sys.stderr)
        print(f"[debug] output = '{output_path}'", file=sys.stderr)

    functions_raw = load_json(functions_path,
                              description="Functions definition file")
    functions = parse_functions_definition(functions_raw)
    functions_by_name = {f.name: f for f in functions}
    if debug:
        print(f"[debug] Loaded {len(functions)}"
              f" function(s): {sorted(functions_by_name)}", file=sys.stderr)

    tests_raw = load_json(input_path,
                          description="Function calling tests file")
    if not isinstance(tests_raw, list):
        raise SchemaError(
            f"'{input_path}' must contain a JSON array "
            f"of {{'prompt': ...}} objects, "
            f"got {type(tests_raw).__name__}."
        )

    prompts: List[str] = []
    for i, entry in enumerate(tests_raw):
        if isinstance(entry, dict) and isinstance(entry.get("prompt"), str):
            prompts.append(entry["prompt"])
        elif isinstance(entry, str):
            prompts.append(entry)
        else:
            print(
                "[warning] Skipping malformed"
                f" test entry at index {i}: {entry!r}",
                file=sys.stderr,
            )

    if not prompts:
        print("[warning] No valid prompts found;"
              " writing an empty result file.", file=sys.stderr)
        write_json(output_path, [])
        return

    decoder = ConstrainedDecoder(model)

    results = []
    total = len(prompts)
    for i, prompt in enumerate(prompts, start=1):
        if not prompt.strip():
            print("[warning] Skipping empty prompt.", file=sys.stderr)
            continue

        print(f"[{i}/{total}] {prompt!r}", file=sys.stderr, flush=True)
        results.append(process_prompt(decoder,
                                      prompt,
                                      functions,
                                      functions_by_name,
                                      debug=debug))

    write_json(output_path, results)
    print(f"Wrote {len(results)} result(s) to '{output_path}'.")
