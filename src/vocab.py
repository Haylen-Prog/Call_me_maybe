"""Loads and normalizes the model's vocabulary file.

The vocab file returned by ``Small_LLM_Model.get_path_to_vocab_file()`` maps
token strings to integer ids (the common HF ``vocab.json`` convention). Some
tokenizers instead serialize the inverse mapping (id -> token) as a JSON
object with string keys; both are supported here.

BPE/SentencePiece tokenizers commonly encode a leading space using a special
marker character (e.g. "Ġ" for GPT-2-style BPE, "▁" for SentencePiece).
``normalize`` converts a raw token string into its "logical" text (the text
it actually contributes to the decoded string) so the grammar engine in
``src/grammar.py`` can reason about real characters instead of marker bytes.
"""

from __future__ import annotations

import json
from typing import Dict, Tuple

from .io_utils import ProjectIOError

_SPACE_MARKERS = ("Ġ", "▁")


def normalize(token: str) -> str:
    """Return the logical text a raw vocab token represents."""
    for marker in _SPACE_MARKERS:
        if token.startswith(marker):
            return " " + token[len(marker):]
    return token


def load_vocab(path: str) -> Tuple[Dict[int, str], Dict[str, int]]:
    """Return (id_to_token, token_to_id) using
    the *raw* (non-normalized) strings.

    Raises ProjectIOError with a clear message if the file is missing,
    unreadable, malformed JSON, or not in a recognizable shape.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except FileNotFoundError as exc:
        raise ProjectIOError(f"Vocabulary file not found: '{path}'.") from exc
    except json.JSONDecodeError as exc:
        raise ProjectIOError(f"Vocabulary file '{path}'"
                             f" is not valid JSON: {exc.msg}.") from exc
    except OSError as exc:
        raise ProjectIOError(f"Vocabulary file '{path}'"
                             f" could not be read: {exc}.") from exc

    if not isinstance(raw, dict) or not raw:
        raise ProjectIOError(f"Vocabulary file '{path}' must "
                             "contain a non-empty JSON object.")

    if (
        "model" in raw and
        isinstance(raw["model"], dict) and "vocab" in raw["model"]
    ):
        raw = raw["model"]["vocab"]
        if not isinstance(raw, dict) or not raw:
            raise ProjectIOError(f"Vocabulary file '{path}' has an "
                                 "empty nested model.vocab.")

    sample_key = next(iter(raw))
    sample_val = raw[sample_key]

    if isinstance(sample_val, int):
        token_to_id = {str(k): int(v) for k, v in raw.items()}
    elif isinstance(sample_key, str) and sample_key.lstrip("-").isdigit():
        token_to_id = {str(v): int(k) for k, v in raw.items()}
    else:
        raise ProjectIOError(
            f"Vocabulary file '{path}' has an unrecognized format "
            "(expected {token: id} or {id: token})."
        )

    id_to_token = {v: k for k, v in token_to_id.items()}
    return id_to_token, token_to_id
