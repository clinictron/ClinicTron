"""Nothing in this package opens a network connection, a GPU, or a database at import
time. Every external system is reached through a client object built by an explicit
factory call, so the stage scripts import and their tests run offline."""
from __future__ import annotations

__all__ = ["config", "prompts", "seeds", "logging", "llm", "corpus", "retrieval"]
