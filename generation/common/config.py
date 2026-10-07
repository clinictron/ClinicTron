"""Load a stage config (YAML) and expand ${VAR} / ${VAR:-default} references from the
environment.

No absolute machine path appears in code. Any string value in the YAML may name an
environment variable; the variables are listed in `env.example.sh`. A default that starts
with `resources` or `prompts` is resolved against this generation directory."""
from __future__ import annotations

import os
import re
from pathlib import Path

import yaml

_VAR_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")

# Repository anchors, derived from this file's location — never hardcoded.
GENERATION_DIR = Path(__file__).resolve().parent.parent
SYNTHDATA_DIR = GENERATION_DIR.parent
PROMPTS_DIR = GENERATION_DIR / "prompts"


def expand(value):
    """Recursively expand ${VAR} / ${VAR:-default} in strings, lists and dicts."""
    if isinstance(value, str):
        def sub(m):
            name, default = m.group(1), m.group(2)
            got = os.environ.get(name)
            if got is not None and got != "":
                return got
            if default is None:
                raise KeyError(f"environment variable {name} is not set and the config "
                               f"gives no default (write it into env.example.sh)")
            if default.startswith(("resources", "prompts")):
                return str(GENERATION_DIR / default)
            return default
        return _VAR_RE.sub(sub, value)
    if isinstance(value, list):
        return [expand(v) for v in value]
    if isinstance(value, dict):
        return {k: expand(v) for k, v in value.items()}
    return value


class Config(dict):
    """The parsed config plus its own path (stages record it in their manifests)."""

    def __init__(self, data: dict, path: str):
        super().__init__(data)
        self.path = os.path.abspath(path)

    @property
    def prompts_dir(self) -> Path:
        return Path(self.get("paths", {}).get("prompts_dir") or PROMPTS_DIR)

    def out_dir(self, stage: str) -> Path:
        """<run_root>/<run_id>/<stage>, created on demand."""
        d = Path(self["paths"]["run_root"]) / self["run"]["run_id"] / stage
        d.mkdir(parents=True, exist_ok=True)
        return d


def load_config(path: str) -> Config:
    with open(path) as fh:
        raw = yaml.safe_load(fh)
    return Config(expand(raw), path)


def load_secrets(path: str | None = None) -> dict[str, str]:
    """Parse a KEY=value secrets file (the machine's api_keys.env) into a dict.

    Never returns values into logs; callers pass them straight to a client.
    """
    path = path or os.environ.get("GEN_SECRETS")
    if not path:
        raise RuntimeError("GEN_SECRETS is not set (see env.example.sh)")
    out: dict[str, str] = {}
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip().strip('"').strip("'")
    return out
