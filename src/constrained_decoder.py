"""Grammar-constrained decoding engine.

The overall generation strategy is "template + slots": everything that is
fixed by the JSON schema (braces, key names, quotes, commas, the literal
words ``true``/``false``) is injected directly as forced tokens (we never
ask the model to "spontaneously" produce correct JSON punctuation). The
model is only ever consulted at *slots*: choosing a function name, choosing
a boolean value, or producing the digits of a number / characters of a
string. At every slot, logits for every token that would break JSON
validity or the field's schema type are set to -inf before argmax, which is
exactly the mechanism described in section 5.3.3 of the subject.

This guarantees 100% syntactically- and schema-valid JSON regardless of how
good (or bad) the underlying model's raw predictions are.
"""

from __future__ import annotations

import math
from typing import Any, Dict, FrozenSet, List, Sequence, Tuple

from . import grammar, vocab as vocab_mod

NEG_INF = float("-inf")

_NUMBER_ALPHABET = set("+-.eE0123456789")


def _to_id_list(x: Any) -> List[int]:
    """Normalize whatever ``model.encode()`` returns into a flat list[int].

    The subject's SDK interface says ``encode`` returns a ``Tensor``, and the
    real implementation returns a 2-D tensor of shape ``(1, seq_len)``
    (batch dimension included). A lightweight/mock implementation might just
    return a plain ``list[int]``. Both are supported here without importing
    torch (so this module has no hard dependency on it).
    """
    if hasattr(x, "tolist"):
        x = x.tolist()
    if (
        isinstance(x, (list, tuple))
        and len(x) > 0 and isinstance(x[0], (list, tuple))
    ):
        x = x[0]
    return [int(i) for i in x]


MAX_NUMBER_CHARS = 24
MAX_STRING_CHARS = 200
MAX_CHOICE_STEPS = 64


class ConstrainedDecoder:
    """Wraps a Small_LLM_Model instance
    with grammar-constrained slot generation.

    Performance note: a real tokenizer vocabulary can have on the order of
    150k entries (e.g. Qwen3). Re-classifying every single vocabulary token
    against the grammar on *every* generated token (as a naive implementation
    would) is O(vocab_size) work per step and, across up to ~200 characters
    for a string field times several fields times many prompts, quickly adds
    up to tens of millions of Python-level operations -- turning what should
    take seconds into a run that looks "stuck" for many minutes.

    To avoid this, everything that only depends on a token's *text* (not on
    the current grammar state) is classified exactly once, here in
    ``__init__``, and reused for the rest of the run:
      * ``_string_safe_ids``: tokens that may appear inside a JSON string.
      * ``_digit_only_ids``: tokens made up entirely of digits -- these are
        *always* a legal continuation of a JSON number regardless of the
        current parser state, so they never need per-step re-checking.
      * ``_number_alphabet_ids``: the small remaining set of tokens that
        contain only number-related characters (``+-.eE0123456789``) but
        aren't pure digits (e.g. ``"."``, ``"e-"``, ...). This set is tiny
        relative to the full vocabulary (real vocabularies are overwhelmingly
        words/subwords), so it is cheap to re-check against the grammar state
        on every step -- only *this* small pool needs that per-step check,
        never the full vocabulary.
    """

    def __init__(self, model: Any) -> None:
        self.model = model
        vocab_path = model.get_path_to_vocab_file()
        self.id_to_token, self.token_to_id = vocab_mod.load_vocab(vocab_path)
        self._id_to_text: Dict[int, str] = {
            tid: vocab_mod.normalize(tok)
            for tid, tok in self.id_to_token.items()
        }

        if not self._id_to_text:
            raise ValueError("Vocabulary is empty; "
                             "cannot perform constrained decoding.")

        string_safe_ids = []
        digit_only_ids = []
        number_alphabet_ids = []
        for tid, text in self._id_to_text.items():
            if not text:
                continue
            if grammar.string_token_is_safe(text):
                string_safe_ids.append(tid)
            if text.isdigit():
                digit_only_ids.append(tid)
            elif all(ch in _NUMBER_ALPHABET for ch in text):
                number_alphabet_ids.append(tid)

        self._string_safe_ids: FrozenSet[int] = frozenset(string_safe_ids)
        self._digit_only_ids: FrozenSet[int] = frozenset(digit_only_ids)

        self._number_special_ids: List[int] = number_alphabet_ids

        self._ids_by_first_char: Dict[str, List[int]] = {}
        for tid, text in self._id_to_text.items():
            if not text:
                continue
            self._ids_by_first_char.setdefault(text[0], []).append(tid)

    def encode_literal(self, text: str) -> List[int]:
        """Force-encode fixed template text (no model call needed: the text
        is fully determined by the schema, so there is nothing to decode)."""
        if text == "":
            return []
        return _to_id_list(self.model.encode(text))

    def _raw_logits(self, ids: Sequence[int]) -> List[float]:
        logits: Any = self.model.get_logits_from_input_ids(list(ids))
        if logits is None or len(logits) == 0:
            raise ValueError("Model returned empty logits;"
                             " cannot continue generation.")
        return [float(v) for v in logits]

    @staticmethod
    def _argmax(logits: Sequence[float]) -> int:
        best_i, best_v = 0, NEG_INF
        for i, v in enumerate(logits):
            if v is None:
                continue
            fv = v if not (isinstance(v, float) and math.isnan(v)) else NEG_INF
            if fv > best_v:
                best_v, best_i = fv, i
        return best_i

    def generate_choice(self,
                        ids: List[int],
                        choices: Sequence[str]) -> Tuple[str, List[int]]:
        """Generate text that must
        exactly match one of ``choices``.

        Implements a simple prefix-trie filter:
          at every step, only tokens
        that keep the generated text a
        prefix of at least one remaining
        candidate are allowed. Stops as
        soon as the generated text uniquely
        and exactly matches one choice.
        """
        if not choices:
            raise ValueError("generate_choice() requires at least one choice.")
        choices = list(dict.fromkeys(choices))

        generated_ids: List[int] = []
        produced = ""
        remaining = list(choices)
        steps = 0

        while steps < MAX_CHOICE_STEPS:
            steps += 1
            if produced in remaining and not any(
                c != produced and c.startswith(produced) for c in remaining
            ):
                break

            needed_first_chars = {c[len(produced)]
                                  for c in remaining
                                  if len(c) > len(produced)}
            pool: List[int] = []
            for ch in needed_first_chars:
                pool.extend(self._ids_by_first_char.get(ch, []))
            candidate_token_ids = [
                tid
                for tid in pool
                if any(c[len(produced):].startswith(self._id_to_text[tid])
                       for c in remaining)
            ]
            if not candidate_token_ids:
                fallback = min(remaining, key=len) if remaining else choices[0]
                extra = fallback[len(produced):]
                if extra:
                    generated_ids.extend(self.encode_literal(extra))
                produced = fallback
                break

            context = list(ids) + generated_ids
            logits = self._raw_logits(context)
            masked = [
                logits[tid] if tid < len(logits)
                and tid in candidate_token_ids else NEG_INF
                for tid in range(len(logits))
            ]
            next_id = self._argmax(masked)
            if masked[next_id] == NEG_INF:
                next_id = candidate_token_ids[0]

            generated_ids.append(next_id)
            produced += self._id_to_text.get(next_id, "")
            remaining = [c for c in remaining if c.startswith(produced)]
            if not remaining:
                fallback = min(choices, key=lambda c: 0
                               if c.startswith(produced[: len(c)]) else 1)
                return fallback, generated_ids

        if produced not in choices:
            produced = choices[0]
        return produced, generated_ids

    def generate_number(self,
                        ids: List[int], *,
                        integer_only: bool) -> Tuple[str, List[int]]:
        generated_ids: List[int] = []
        state = "start"
        text = ""

        for _ in range(MAX_NUMBER_CHARS):
            context = list(ids) + generated_ids
            raw_logits = self._raw_logits(context)
            raw_top = self._argmax(raw_logits)
            raw_top_text = self._id_to_text.get(raw_top, "")

            if grammar.number_is_accepting(state):
                next_state_for_top = grammar.number_token_step(
                    state, raw_top_text, integer_only=integer_only
                )
                if next_state_for_top is None:
                    break

            candidate_ids = set(self._digit_only_ids)
            for tid in self._number_special_ids:
                if grammar.number_token_step(
                    state, self._id_to_text[tid], integer_only=integer_only
                ) is not None:
                    candidate_ids.add(tid)

            if not candidate_ids:
                if grammar.number_is_accepting(state):
                    break

                new_state = grammar.number_step(state,
                                                "0",
                                                integer_only=integer_only)
                text += "0"
                generated_ids.extend(self.encode_literal("0"))
                state = new_state or "int"
                if grammar.number_is_accepting(state):
                    break
                continue

            masked = [
                raw_logits[tid] if tid < len(raw_logits)
                and tid in candidate_ids else NEG_INF
                for tid in range(len(raw_logits))
            ]
            next_id = self._argmax(masked)
            if masked[next_id] == NEG_INF:
                next_id = next(iter(candidate_ids))

            next_text = self._id_to_text.get(next_id, "")
            new_state = grammar.number_token_step(state,
                                                  next_text,
                                                  integer_only=integer_only)
            assert new_state is not None
            state = new_state
            text += next_text
            generated_ids.append(next_id)

        if not grammar.number_is_accepting(state) or text == "":
            text = "0"
            generated_ids = self.encode_literal("0")

        return text, generated_ids

    def generate_string(self,
                        ids: List[int], *,
                        max_chars: int = MAX_STRING_CHARS) -> Tuple[
                            str, List[int]]:
        generated_ids: List[int] = []
        text = ""
        candidate_ids = self._string_safe_ids

        for _ in range(max_chars):
            context = list(ids) + generated_ids
            raw_logits = self._raw_logits(context)
            raw_top = self._argmax(raw_logits)
            raw_top_text = self._id_to_text.get(raw_top, "")

            if (
                raw_top_text.strip('"') == "" or
                not grammar.string_token_is_safe(raw_top_text)
            ):
                break

            if not candidate_ids:
                break

            masked = [
                raw_logits[tid] if tid < len(raw_logits)
                and tid in candidate_ids else NEG_INF
                for tid in range(len(raw_logits))
            ]
            next_id = self._argmax(masked)
            if masked[next_id] == NEG_INF:
                break
            next_text = self._id_to_text.get(next_id, "")
            if len(text) + len(next_text) > max_chars:
                break
            text += next_text
            generated_ids.append(next_id)

        return text, generated_ids
