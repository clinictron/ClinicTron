"""logging.py — run logs that ride with their data.

Two loggers per stage, both append-only and fsynced, both echoed to stdout:
  run.log    progress and decisions
  skips.log  every skip / discard / gate failure, with the verb first

A guard that declines must say so: `SkipLog.note` is the only way a row leaves the
pipeline, and it counts what it wrote so the stage summary can report it.
"""
from __future__ import annotations

import json
import os
import time
from collections import Counter
from pathlib import Path


def _ts() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


class Log:
    def __init__(self, path: str | Path, echo: bool = True):
        self.path = str(path)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.echo = echo

    def __call__(self, msg: str) -> None:
        line = f"{_ts()}  {msg}"
        with open(self.path, "a") as fh:
            fh.write(line + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        if self.echo:
            print(line, flush=True)


class SkipLog(Log):
    """Skip/discard logger that also tallies reasons for the stage summary."""

    def __init__(self, path: str | Path, echo: bool = True):
        super().__init__(path, echo)
        self.counts: Counter = Counter()

    def note(self, verb: str, **fields) -> None:
        self.counts[verb] += 1
        detail = "  ".join(f"{k}={v!r}" for k, v in fields.items())
        self(f"{verb}  {detail}")

    def summary(self) -> dict:
        return dict(self.counts)


def append_jsonl(path: str | Path, obj: dict) -> None:
    """Append + fsync: a paid artifact is never lost to a crash."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as fh:
        fh.write(json.dumps(obj, ensure_ascii=False) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def load_keys(path: str | Path, key: str) -> set:
    """Existing ids in a jsonl, for resume."""
    out = set()
    p = Path(path)
    if p.exists():
        with open(p) as fh:
            for line in fh:
                try:
                    out.add(json.loads(line)[key])
                except Exception:
                    continue
    return out
