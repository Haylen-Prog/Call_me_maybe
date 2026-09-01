"""Parsing and validation of ``functions_definition.json``.

The subject only guarantees the *shape* of the examples, not the exact key
names, so this parser is deliberately lenient about a couple of equivalent
encodings (dict-of-params vs list-of-params, "type" spelled as "number" vs
"float" vs "integer", optional "required" list, etc.) while still failing
loudly on anything genuinely unusable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

SUPPORTED_TYPES = {"string", "number", "integer", "float", "boolean", "bool"}

_TYPE_ALIASES = {
    "float": "number",
    "double": "number",
    "int": "integer",
    "bool": "boolean",
    "str": "string",
}


def _normalize_type(raw_type: Any) -> str:
    """
    Docstring for _normalize_type
    :param raw_type: Description
    :type raw_type: Any
    :return: Description
    :rtype: str
    """
    if not isinstance(raw_type, str):
        raise SchemaError("Parameter type must be a "
                          f"string, got: {raw_type!r}.")
    t = raw_type.strip().lower()
    t = _TYPE_ALIASES.get(t, t)
    if t not in {"string", "number", "integer", "boolean"}:
        raise SchemaError(
            f"Unsupported parameter type '{raw_type}'. "
            f"Supported types: string, number, integer, boolean."
        )
    return t


class SchemaError(Exception):
    """Raised when functions_definition.json is structurally invalid."""


@dataclass
class ParamDef:
    name: str
    type: str
    description: str = ""
    required: bool = True


@dataclass
class FunctionDef:
    name: str
    description: str = ""
    parameters: "Dict[str, ParamDef]" = field(default_factory=dict)
    returns: Optional[str] = None

    def to_prompt_dict(self) -> Dict[str, Any]:
        """Compact representation injected into the LLM context."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                pname: {"type": p.type, "description": p.description}
                for pname, p in self.parameters.items()
            },
        }


def _parse_parameters(raw_params: Any, func_name: str) -> Dict[str, ParamDef]:
    """
    Docstring for _parse_parameters
    :param raw_params: Description
    :type raw_params: Any
    :param func_name: Description
    :type func_name: str
    :return: Description
    :rtype: Dict[str, ParamDef]
    """
    if raw_params is None:
        return {}

    params: Dict[str, ParamDef] = {}

    if isinstance(raw_params, dict):
        if (
            "properties" in raw_params and
            isinstance(raw_params["properties"], dict)
        ):
            required = set(raw_params.get("required", []))
            for pname, pdef in raw_params["properties"].items():
                if not isinstance(pdef, dict) or "type" not in pdef:
                    raise SchemaError(
                        f"Function '{func_name}': parameter '{pname}'"
                        " is missing a 'type'."
                    )
                params[pname] = ParamDef(
                    name=pname,
                    type=_normalize_type(pdef["type"]),
                    description=str(pdef.get("description", "")),
                    required=(pname in required) if required else True,
                )
            return params

        for pname, pdef in raw_params.items():
            if isinstance(pdef, str):
                params[pname] = ParamDef(name=pname,
                                         type=_normalize_type(pdef))
            elif isinstance(pdef, dict):
                if "type" not in pdef:
                    raise SchemaError(
                        f"Function '{func_name}': parameter "
                        f"'{pname}' is missing a 'type'."
                    )
                params[pname] = ParamDef(
                    name=pname,
                    type=_normalize_type(pdef["type"]),
                    description=str(pdef.get("description", "")),
                    required=bool(pdef.get("required", True)),
                )
            else:
                raise SchemaError(
                    f"Function '{func_name}': parameter '{pname}'"
                    " has an invalid definition "
                    f"(expected object or type string,"
                    " got {type(pdef).__name__})."
                )
        return params

    if isinstance(raw_params, list):
        for i, pdef in enumerate(raw_params):
            if (
                not isinstance(pdef, dict) or "name"
                not in pdef or "type" not in pdef
            ):
                raise SchemaError(
                    f"Function '{func_name}': parameters[{i}] "
                    "must be an object with "
                    f"'name' and 'type' keys, got: {pdef!r}."
                )
            pname = pdef["name"]
            params[pname] = ParamDef(
                name=pname,
                type=_normalize_type(pdef["type"]),
                description=str(pdef.get("description", "")),
                required=bool(pdef.get("required", True)),
            )
        return params

    raise SchemaError(
        f"Function '{func_name}': 'parameters' must be an object or a list, "
        f"got {type(raw_params).__name__}."
    )


def parse_functions_definition(raw: Any) -> List[FunctionDef]:
    """Parse the top-level content of functions_definition.json.

    Accepts either a bare JSON array of function objects, or an object with
    a "functions" key wrapping that array (both appear in the wild across
    similar assignments), and validates each entry thoroughly.
    """
    if isinstance(raw, dict) and "functions" in raw:
        raw = raw["functions"]

    if not isinstance(raw, list):
        raise SchemaError(
            "functions_definition.json must contain a JSON array of function "
            f"definitions (optionally wrapped in {{'functions': [...]}}), got "
            f"{type(raw).__name__}."
        )
    if len(raw) == 0:
        raise SchemaError("functions_definition.json "
                          "contains no function definitions.")

    functions: List[FunctionDef] = []
    seen_names = set()
    for i, entry in enumerate(raw):
        if not isinstance(entry, dict):
            raise SchemaError(f"functions_definition.json[{i}]"
                              " must be an object, got {entry!r}.")
        if (
            "name" not in entry or not isinstance(entry["name"], str)
            or not entry["name"].strip()
        ):
            raise SchemaError(f"functions_definition.json[{i}]"
                              " is missing a non-empty 'name'.")

        name = entry["name"]
        if name in seen_names:
            raise SchemaError(f"Duplicate function name '{name}'"
                              " in functions_definition.json.")
        seen_names.add(name)

        parameters = _parse_parameters(entry.get("parameters"), name)

        functions.append(
            FunctionDef(
                name=name,
                description=str(entry.get("description", "")),
                parameters=parameters,
                returns=entry.get("return_type") or entry.get("returns"),
            )
        )

    return functions
