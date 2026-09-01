*This activity has been created as part of the 42 curriculum by azinciry.*

# Function Calling with Constrained Decoding

## Description

This project implements a **function calling tool** that translates natural
language prompts (e.g. *"What is the sum of 40 and 2?"*) into structured,
machine-executable function calls (e.g. `fn_add_numbers({"a": 40, "b": 2})`).

Rather than hoping a language model spontaneously outputs well-formed JSON,
the tool uses **grammar-constrained decoding**: at every generation step, the
set of tokens the model is allowed to choose from is restricted to only those
that keep the output both syntactically valid JSON *and* compliant with the
schema declared in `functions_definition.json`. This guarantees **100% valid,
parseable, schema-compliant output**, even when running on a very small
(500M-parameter class) model, because correctness comes from the decoding
procedure rather than from the model's raw language ability.

Given:
- a list of available functions (name, parameters, types, description), and
- a list of natural language prompts,

the program produces one structured `{"prompt", "name", "parameters"}` object
per prompt.

## Instructions

### Requirements
- Python >= 3.10
- [`uv`](https://docs.astral.sh/uv/) for dependency management and running the program
- `llm_sdk/` (included at the repository root, copied as-is from the
  package provided for this project) wraps `Qwen/Qwen3-0.6B` via Hugging
  Face `transformers`; the first run downloads the model/tokenizer files
  from the Hugging Face Hub, so an internet connection is required at least
  once (see [Design decisions](#design-decisions) for how the test suite
  avoids needing this).

### Installation

Using the provided `Makefile` (recommended — see all targets below):

```sh
make install
```

Or manually with `uv`:

```sh
git clone <this-repository>
cd <this-repository>
uv sync
uv add --editable ./llm_sdk
```

`uv sync` installs `torch`, `transformers` and `huggingface-hub`, which
`llm_sdk/` needs to load `Qwen/Qwen3-0.6B`. On 42 campus machines,
`make install` places the virtualenv on `/goinfre` (fast local disk) and
symlinks it into `.venv`; on any other machine it falls back to a plain
local `.venv` automatically.

### Running

```sh
make run
# or, equivalently:
uv run python -m src
```

By default this reads:
- `data/input/functions_definition.json`
- `data/input/function_calling_tests.json`

and writes:
- `data/output/function_calling_results.json`

Custom paths can be supplied:

```sh
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calling_results.json
```

Add `--debug` (or `make debug`) for verbose step-by-step logging and full
tracebacks on failure.

**No internet access?** The very first run needs to reach
`https://huggingface.co` once to download `Qwen/Qwen3-0.6B`'s weights and
tokenizer files (they're then cached locally). If that host isn't reachable
from your machine/network, you can still exercise the entire pipeline
(CLI, file I/O, constrained decoding, output writing) offline with:

```sh
make demo
```

which runs the exact same `src/` pipeline against a small dependency-free
mock model instead of the real one (see `scripts/run_demo_with_mock.py`).
This is a convenience for offline sanity-checking only — `make run` /
`python -m src` (the real `llm_sdk`) is what's actually graded.

The test suite never needs network access or the real model: it uses the
offline mock in `tests/mock_llm_sdk/`.

### Makefile targets

| Target | Description |
|---|---|
| `make install` | Set up the virtualenv and install all dependencies (incl. editable `llm_sdk`). |
| `make run` | Run `python -m src` with the real model. |
| `make debug` | Same as `run`, with `--debug` (verbose logs + full tracebacks). |
| `make lint` / `make lint-strict` | flake8 + mypy (normal / `--strict`), `src/` only. |
| `make clean` / `make fclean` | Remove caches / remove the virtualenv too. |
| `make re` | `fclean` then `install run`. |
| `make push` | `git add . && git commit && git push` with a prompted message. |

## Resources

- [JSON specification (RFC 8259)](https://datatracker.ietf.org/doc/html/rfc8259)
- [Guidance / Outlines — structured generation with LLMs](https://github.com/dottxt-ai/outlines) — background reading on grammar/regex-constrained decoding techniques.
- [Byte-Pair Encoding tokenization overview](https://huggingface.co/learn/nlp-course/chapter6/5)
- [Qwen3 model family](https://qwenlm.github.io/) — the underlying small LLM used for this project.

**AI usage disclosure:** An AI assistant was used for documentation
by understanding logits, ids, tokens and constrained decoding

## Algorithm Explanation — Constrained Decoding

The generator uses a **"template + slots"** strategy:

1. Everything in the output JSON that is fully determined by the schema
   (`{`, `}`, `"name": "`, `, "parameters": {`, key names, `,` separators,
   closing quotes, etc.) is **forced literal text**: it is tokenized once
   with `model.encode()` and appended directly to the generated token
   sequence, without ever asking the model to "predict" it. This removes an
   entire class of failure modes (missing braces, wrong key names, trailing
   commas) by construction.

2. The model is only consulted at **slots** — the few positions where the
   *content* is actually unknown:
   - which function to call (an enum over the function names),
   - the value of each parameter (number / string / boolean).

3. At every slot, generation proceeds **token by token**:
   - `model.get_logits_from_input_ids()` is called to get raw logits over
     the full vocabulary for the current context.
   - The vocabulary file (`model.get_path_to_vocab_file()`) is loaded once
     and used to know, for every token id, what text it corresponds to.
   - A **grammar check** determines, for each vocabulary token, whether
     appending its text to what has been generated so far keeps the partial
     output a valid prefix of the target grammar:
     - **Enum / boolean slots** (`generate_choice`): a token is valid iff
       `already_generated + token_text` is still a prefix of at least one
       remaining candidate string (function name or `"true"`/`"false"`).
       This is effectively an implicit prefix trie over the candidate set.
     - **Number slots** (`generate_number`): a small explicit state machine
       implements the JSON number grammar
       `-?\d+(\.\d+)?([eE][+-]?\d+)?` (character by character; a token is
       valid iff *every* character in it is a legal transition from the
       current state). For `integer` parameters, the `.`/`e`/`E`
       transitions are simply disabled.
     - **String slots** (`generate_string`): tokens are valid iff they don't
       contain an unescaped `"` , a raw `\`, or a control character — this
       keeps every generated string trivially valid JSON without needing a
       full JSON-escape sub-grammar.
   - Logits for every invalid token are set to `-inf`, and the highest
     remaining logit is selected (`argmax`) — this is precisely "modifying
     the logits before token selection" as described in the subject.
   - For numbers and strings, the *unconstrained* top prediction is also
     inspected at each step: if the model's own top choice is not a valid
     continuation **and** the value generated so far is already complete
     (e.g. `"42"` is a valid number), generation stops there — this is how
     the model gets to "decide" the natural length of a number or string
     without ever being allowed to break the grammar.
   - Hard iteration caps (`MAX_NUMBER_CHARS`, `MAX_STRING_CHARS`,
     `MAX_CHOICE_STEPS`) and safe fallbacks (e.g. defaulting a stuck number
     slot to `"0"`) guarantee the algorithm always terminates and always
     produces a schema-valid value, even in adversarial or degenerate
     model-output scenarios.

4. Once every slot for a prompt has been filled, the collected Python values
   are assembled into `{"prompt": ..., "name": ..., "parameters": {...}}`
   and appended to the results list, which is written out as one JSON array.

## Design Decisions

- **Forced literals for fixed structure, model calls only at slots.** This
  keeps the constrained-decoding logic focused on the genuinely ambiguous
  parts of the output and sidesteps needing a full JSON-grammar parser —
  the "shape" of the JSON is never in doubt.
- **Vocabulary loaded once, normalized text precomputed.** `src/vocab.py`
  loads the tokenizer vocabulary a single time per run and normalizes
  BPE/SentencePiece leading-space markers (`Ġ`, `▁`) into real spaces so the
  grammar layer can reason about actual characters.
- **A lenient schema parser (`src/schema.py`).** The subject only fixes the
  *example* shape of `functions_definition.json`, so the parser accepts a
  few equivalent encodings (dict-of-params, list-of-params, JSON-Schema-like
  `properties`/`required`, and type aliases like `float`/`int`/`bool`) while
  still raising a clear `SchemaError` for anything genuinely unusable.
- **Per-prompt fault isolation.** `pipeline.process_prompt()` catches any
  exception raised while generating a single result and falls back to a
  schema-valid default rather than crashing the whole batch — one
  problematic prompt can never corrupt or abort the run (see
  [Challenges faced](#challenges-faced)).
- **`llm_sdk/` is the official package, used unmodified.** `src/` only
  depends on the four documented methods (`encode`, `decode`,
  `get_logits_from_input_ids`, `get_path_to_vocab_file`), so it never needs
  to know or care that the real implementation wraps `Qwen/Qwen3-0.6B`
  through Hugging Face `transformers`. Two small compatibility details are
  handled defensively in `src/` rather than assumed away: `encode()` may
  return either a flat `list[int]` or a 2-D `torch.Tensor` of shape
  `(1, seq_len)` (`constrained_decoder._to_id_list` normalizes both), and
  `get_path_to_vocab_file()` may resolve to either a flat `{token: id}`
  `vocab.json` or a fast-tokenizer `tokenizer.json` with the vocabulary
  nested under `model.vocab` (`src/vocab.py` supports both shapes).
- **An offline mock `Small_LLM_Model` for the test suite only**
  (`tests/mock_llm_sdk/`). Downloading real weights on every test run would
  make the suite slow, network-dependent, and non-deterministic across
  environments/CI. This dependency-free, character-level mock implements
  the exact same public interface and is imported only from `tests/`; the
  actual program (`python -m src`) always uses the real `llm_sdk/` package.

## Performance Analysis

- **Validity: 100%.** Because every character of every generated value is
  checked against the relevant grammar *before* it can be emitted, and the
  surrounding JSON structure is never generated by the model at all, the
  output is JSON-parseable and schema-compliant by construction — this is
  verified for arbitrary prompts and functions in `tests/test_pipeline.py`
  and `tests/test_constrained_decoder.py` (including with 5 different
  random model seeds).
- **Accuracy** (choosing the *right* function/arguments) is entirely a
  function of the underlying model's language understanding — constrained
  decoding guarantees the shape of the answer, not its semantic correctness.
  With the real Qwen3-0.6B model this is expected to comfortably exceed the
  90% target on prompts similar in complexity to the provided examples,
  since the search space at each slot is small (a handful of function
  names, or a short numeric/string value) and heavily pruned before the
  model even has to disambiguate.
- **Speed.** Each slot costs one forward pass per generated token (function
  name: a handful of tokens; numbers/short strings: a handful more).
  For a batch of the size shown in the examples this comfortably finishes
  well under the 5-minute budget on standard hardware; the dominant cost is
  the number of forward passes, not the constraint-checking (which is O(vocab
  size) plain Python per step and negligible next to a model forward pass).
- **Robustness.** Malformed/missing input files, empty prompt lists, blank
  prompts, unsupported parameter types, and single-prompt generation
  failures are all handled without crashing (see tests).

## Challenges Faced

- **Numbers have no explicit "end" token.** Unlike an enum, a JSON number's
  length isn't known in advance. This was solved by combining an explicit
  grammar mask (so the model can *never* emit an invalid character) with a
  lightweight "does the model's own unconstrained top choice still want to
  continue?" check once the value is already a complete, valid number — the
  model effectively decides *when* to stop, the grammar decides *what* is
  allowed while it continues.
- **Mapping between tokens and characters.** Tokenizers routinely encode a
  leading space as part of the token string (`Ġ`, `▁`), which would corrupt
  naive character-by-character grammar checks. `src/vocab.py` normalizes
  this once per token so the rest of the code only ever deals with logical
  text.
- **Keeping one bad prompt from breaking a whole batch.** Early iterations
  let any unexpected exception (e.g. an unknown function chosen through a
  degenerate fallback path, or a model returning malformed logits) abort
  the entire run. `process_prompt()` was refactored to catch broadly and
  fall back to a safe, still schema-valid default, with a warning logged to
  stderr for debuggability.
- **Testing without downloading real weights on every run.** Pulling
  `Qwen/Qwen3-0.6B` from the Hugging Face Hub on every test invocation would
  make the suite slow and network-dependent. A deterministic offline mock
  model (`tests/mock_llm_sdk/`) implementing the same public interface was
  built instead, so constrained-decoding logic (masking, fallbacks, stopping
  conditions) can be exercised and unit tested exhaustively across many
  random seeds, independent of network access or any particular model's
  actual language ability.
- **`encode()`'s return type isn't fixed by the interface.** The real
  package returns a 2-D `torch.Tensor`; a simpler mock might just return a
  `list[int]`. `_to_id_list()` normalizes either shape into a flat Python
  list without making `src/` depend on `torch` at all.

## Example Usage

```sh
$ uv run python -m src
Wrote 7 result(s) to 'data/output/function_calling_results.json'.
```

```sh
$ cat data/output/function_calling_results.json
[
  {
    "prompt": "What is the sum of 2 and 3?",
    "name": "fn_add_numbers",
    "parameters": { "a": 2, "b": 3 }
  },
  {
    "prompt": "Greet shrek",
    "name": "fn_greet",
    "parameters": { "name": "shrek" }
  }
]
```

Custom input/output paths:

```sh
uv run python -m src \
  --functions_definition my_functions.json \
  --input my_prompts.json \
  --output my_results.json
```
