"""prompts.py — read the prompt files under `prompts/` and fill their slots.

Every prompt is its own text file. A slot is written `{name}`; filling replaces only
the slot names the caller supplies, so JSON braces inside a prompt (`{"query": "..."}`)
are left alone. `str.format` is deliberately NOT used — it would need the JSON literals
escaped and would make the files unreadable at sign-off.

Type specs (`prompts/types/<type>.md`) and any other data-carrying prompt file hold one
fenced ```yaml block; `load_spec` returns it parsed.
"""
from __future__ import annotations

import json
import random
import re
from functools import lru_cache
from pathlib import Path

import yaml

from .config import PROMPTS_DIR

_SLOT_RE = re.compile(r"\{([a-z_][a-z0-9_]*)\}")
_YAML_BLOCK_RE = re.compile(r"```yaml\n(.*?)\n```", re.S)


# A prompt file may carry HTML comments, e.g. an ablation variant noting its parent
# with `<!-- ablation: ... -->`. It must never travel to the model inside the
# prompt. Stripping is a byte-level no-op on every live prompt file (none contains one;
# test_prompt_render asserts it), so this changes nothing the live pipeline sends.
_HTML_COMMENT_RE = re.compile(r"<!--.*?-->\n?", re.S)


# Default writer prompt files. A config `writer_prompts:` block can replace either one;
# every other prompt is reused untouched.
WRITER_PROMPT_DEFAULTS = {"grounded": "writer_grounded.md",
                          "ungrounded": "writer_ungrounded.md"}


def prompts_root(root: str | Path | None = None) -> Path:
    return Path(root) if root else PROMPTS_DIR


@lru_cache(maxsize=256)
def _read(path: str) -> str:
    return Path(path).read_text()


def strip_comments(text: str) -> str:
    """Remove HTML comments (`<!-- ... -->`) from a prompt body."""
    return _HTML_COMMENT_RE.sub("", text)


def load(name: str, root: str | Path | None = None) -> str:
    """Read one prompt file by name relative to prompts/ (e.g. 'groups/escalation.md').

    HTML comments are stripped, so notes in a prompt file are never sent as
    instructions to the model."""
    p = prompts_root(root) / name
    if not p.exists():
        raise FileNotFoundError(f"prompt file not found: {p}")
    return strip_comments(_read(str(p)))


def slots(text: str) -> set[str]:
    """Every `{slot}` name present in a prompt body."""
    return set(_SLOT_RE.findall(text))


def fill(text: str, **values) -> str:
    """Replace `{name}` for each supplied name. Unknown leftovers are an error.

    Leftover slots are checked in the template before substitution, because a supplied
    value such as a paper abstract may itself contain `{at}`, `{n}` or any other brace
    pair (chemistry, maths, OCR noise) that is not a template slot."""
    left = slots(text) - set(values)
    if left:
        raise ValueError(f"unfilled prompt slots: {sorted(left)}")
    for k, v in values.items():
        text = text.replace("{" + k + "}", "" if v is None else str(v))
    return text


def load_spec(name: str, root: str | Path | None = None) -> dict:
    """Parse the single fenced yaml block out of a spec file."""
    text = load(name, root)
    m = _YAML_BLOCK_RE.search(text)
    if not m:
        raise ValueError(f"{name}: no ```yaml block")
    return yaml.safe_load(m.group(1))


def load_type_spec(type_id: str, root=None) -> dict:
    return load_spec(f"types/{type_id}.md", root)


def load_group_block(group: str, root=None) -> str:
    """The group's task block, verbatim (its only slot is {seed_entity})."""
    return load(f"groups/{group}.md", root).rstrip("\n")


def load_exemplars(group: str, root=None) -> list[dict]:
    """The group's pool of verbatim real NLM questions."""
    return json.loads(load(f"exemplars/{group}.json", root))


def draw_exemplars(group: str, seed: int, lo: int = 3, hi: int = 5, root=None
                   ) -> list[dict]:
    """Draw 3-5 of the group's pool for one cell, deterministically.

    The pool is sorted by query_id and sampled with the cell's seed, so a different
    subset is drawn per cell — not the whole pool every time."""
    pool = sorted(load_exemplars(group, root), key=lambda e: e["query_id"])
    rng = random.Random(seed)
    k = rng.randint(lo, min(hi, len(pool)))
    return rng.sample(pool, k)


def style_block(style: str, root=None) -> str:
    return load(f"style_{style}.md", root).rstrip("\n")


def length_block(length: str, pasted: bool, document_type: str | None, root=None) -> str:
    """The length instruction; a pasted long cell uses the pasted template for its document type."""
    if length == "long" and pasted:
        if not document_type:
            raise ValueError("a pasted long cell needs a document_type")
        return fill(load("length_long_pasted.md", root).rstrip("\n"),
                    document_type=document_type)
    return load(f"length_{length}.md", root).rstrip("\n")


def rule1_block(masked: bool, root=None) -> str:
    """Rule 1 of the grounded writer: `rule1_masked.md` (latent entity) when the concept
    is hidden, `rule1_named.md` when it may be named."""
    return load("rule1_masked.md" if masked else "rule1_named.md", root).rstrip("\n")


def writer_prompt_files(cfg) -> dict:
    """The writer prompt files {"grounded": ..., "ungrounded": ...} for this config.

    An optional `writer_prompts:` block in the config names one or both; anything it does
    not name keeps the live file. `02_write_questions.py --writer-prompt <name>` sets both
    for one process, which is how an ablation arm selects its own prompt without a
    code change and without a second copy of the prompt directory."""
    over = cfg.get("writer_prompts") or {} if hasattr(cfg, "get") else {}
    return {**WRITER_PROMPT_DEFAULTS, **{k: v for k, v in over.items() if v}}


def render_writer(cfg, cell: dict, grounding_block: str | None = None,
                  exemplars: list[dict] | None = None) -> str:
    """Assemble one cell's writer prompt.

    Failure-group cells use the ungrounded writer with the group block and 3-5 real-NLM
    style exemplars. Ely-type and subclass cells use the grounded writer with the question
    type, its asks, the entity and the grounding papers. Style and length are drawn slots
    in both."""
    root = cfg.prompts_dir if hasattr(cfg, "prompts_dir") else None
    files = writer_prompt_files(cfg)
    style = style_block(cell["style"], root)
    length = length_block(cell["length"], cell["pasted"], cell.get("document_type"), root)
    entity = cell["entity"]["display_name"]


    if cell["source"] == "failure_groups":
        group = cell["type_or_group"]
        ex_cfg = cfg.get("exemplars", {}) if hasattr(cfg, "get") else {}
        ex = exemplars if exemplars is not None else draw_exemplars(
            group, int(cell["seed"]), int(ex_cfg.get("min_in_prompt", 3)),
            int(ex_cfg.get("max_in_prompt", 5)), root)
        block = fill(load_group_block(group, root), seed_entity=entity)
        return fill(load(files["ungrounded"], root),
                    exemplar_block=exemplar_block(ex), seed_entity=entity,
                    group_block=block, length_instruction=length,
                    style_instruction=style)

    if cell["source"] == "subclasses":
        sc = cell["subclass"]
        question_type = f"{sc['type_name'].lower()} — {sc['subtype_name'].lower()}"
        asks = [sc["ask"]]
    else:
        spec = load_type_spec(cell["type_or_group"], root)
        question_type = cell["type_or_group"].replace("_", " ")
        asks = list(spec["asks"])
    return fill(load(files["grounded"], root),
                question_type=question_type,
                ask_template='" or "'.join(asks),
                entity=entity,
                synonyms=", ".join(cell["entity"].get("synonyms") or []) or "(none)",
                grounding_block=grounding_block if grounding_block is not None
                else "(no grounding papers supplied)",
                rule1=rule1_block(cell["masked"], root),
                length_instruction=length, style_instruction=style)


def grounding_query(cfg, cell: dict) -> str:
    """The search query used to find grounding papers for a cell.

    Ely types use their own template from `prompts/types/<type>.md`; subclasses use the
    config's per-track template (`grounding_by_track`, else `grounding_default`)."""
    entity = cell["entity"]["display_name"]
    if cell["source"] == "subclasses":
        by_track = cfg["grounding_by_track"]
        tpl = by_track.get(cell["track"], cfg["grounding_default"])
    else:
        tpl = load_type_spec(cell["type_or_group"],
                             getattr(cfg, "prompts_dir", None))["grounding"]
    return fill(tpl, entity=entity)


def exemplar_block(exemplars: list[dict]) -> str:
    """Join 3-5 non-empty exemplar texts, one per line."""
    ex = [(e["text"] if isinstance(e, dict) else e).strip() for e in exemplars]
    ex = [e for e in ex if e]
    if not 3 <= len(ex) <= 5:
        raise ValueError(f"3-5 non-empty exemplars required, got {len(ex)}")
    return "\n".join(ex)
