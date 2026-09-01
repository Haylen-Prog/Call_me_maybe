"""Character-level grammar rules used by the constrained decoder.

These functions never *generate* anything themselves; they only answer the
question "given the text produced so far, is this next character a legal
continuation, and is what we have so far already a complete, valid value?".
The decoder in ``constrained_decoder.py`` uses this to mask invalid tokens
out of the model's logits before sampling, which is what guarantees the
100%-valid-JSON property required by the subject.
"""

from __future__ import annotations

from typing import Optional

_NUMBER_ACCEPTING_STATES = {"int", "frac", "expdigit"}


def number_step(state: str, ch: str, *, integer_only: bool) -> Optional[str]:
    """Return the next state after consuming ``ch`` from ``state``, or None
    if ``ch`` is not a legal continuation at all."""
    if state == "start":
        if ch == "-":
            return "neg"
        if ch.isdigit():
            return "int"
        return None
    if state == "neg":
        if ch.isdigit():
            return "int"
        return None
    if state == "int":
        if ch.isdigit():
            return "int"
        if ch == "." and not integer_only:
            return "dot"
        if ch in "eE" and not integer_only:
            return "exp"
        return None
    if state == "dot":
        if ch.isdigit():
            return "frac"
        return None
    if state == "frac":
        if ch.isdigit():
            return "frac"
        if ch in "eE":
            return "exp"
        return None
    if state == "exp":
        if ch in "+-":
            return "expsign"
        if ch.isdigit():
            return "expdigit"
        return None
    if state == "expsign":
        if ch.isdigit():
            return "expdigit"
        return None
    if state == "expdigit":
        if ch.isdigit():
            return "expdigit"
        return None
    return None


def number_token_step(state: str,
                      token_text: str, *,
                      integer_only: bool) -> Optional[str]:
    """Apply ``number_step`` for every character of a (possibly multi-char)
    token; returns the resulting state, or None if any character is invalid."""
    if token_text == "":
        return None
    current: str = state
    for ch in token_text:
        next_state = number_step(current, ch, integer_only=integer_only)
        if next_state is None:
            return None
        current = next_state
    return current


def number_is_accepting(state: str) -> bool:
    """
    Docstring for number_is_accepting
    :param state: Description
    :type state: str
    :return: Description
    :rtype: bool
    """
    return state in _NUMBER_ACCEPTING_STATES


_FORBIDDEN_STRING_CHARS = {'"', "\\"}


def string_token_is_safe(token_text: str) -> bool:
    """
    Docstring for string_token_is_safe
    :param token_text: Description
    :type token_text: str
    :return: Description
    :rtype: bool
    """
    if token_text == "":
        return False
    for ch in token_text:
        if ch in _FORBIDDEN_STRING_CHARS:
            return False
        if ord(ch) < 0x20:
            return False
    return True
