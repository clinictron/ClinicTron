"""Model lanes: one OpenAI-compatible chat client per configured model.

The endpoint is LLM_BASE_URL (default: the OpenRouter API); the key is read from the
environment variable named by `models.api_key_env`. The token ceiling, retry count and
fallback prices below are the values the original run used. Nothing here opens a connection
at import time; a stage builds lanes with `build_lanes(cfg, budget, log)` and a test injects
`StubLane`.

Dropped from the original run's helper: the OpenRouter `provider` routing block (it sent
order=[], quantizations=[], sort="price", require_parameters=False, allow_fallbacks=True,
i.e. no pin), which an ordinary OpenAI-compatible endpoint does not accept.
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import threading
import time

MAX_TOKENS_CEILING = 16384     # any lane max_tokens is clamped to this
REQ_RETRIES = 5                # default attempts per call; empty replies are retried free
FALLBACK_PRICE_IN = 0.034      # USD per 1M prompt tokens, used when the reply has no cost
FALLBACK_PRICE_OUT = 0.168     # USD per 1M completion tokens, same


def _extract_usage(resp) -> dict:
    """Tokens and cost from an OpenAI-SDK response (`usage.cost` when the endpoint sends it)."""
    out = {"prompt_tokens": 0, "completion_tokens": 0, "cost": None, "provider": None}
    u = getattr(resp, "usage", None)
    if u is not None:
        out["prompt_tokens"] = int(getattr(u, "prompt_tokens", 0) or 0)
        out["completion_tokens"] = int(getattr(u, "completion_tokens", 0) or 0)
        cost = getattr(u, "cost", None)
        if cost is None:
            cost = (getattr(u, "model_extra", None) or {}).get("cost")
        out["cost"] = float(cost) if cost is not None else None
    prov = getattr(resp, "provider", None)
    if prov is None:
        prov = (getattr(resp, "model_extra", None) or {}).get("provider")
    out["provider"] = prov
    return out


class BudgetAbort(RuntimeError):
    """Raised when cumulative spend crosses the hard abort threshold."""


def parse_json_lenient(text: str):
    """Strict json.loads, then fenced-block strip, then outermost {...} or [...]."""
    text = (text or "").strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if m:
        try:
            return json.loads(m.group(1).strip())
        except Exception:
            pass
    starts = [i for i in (text.find("{"), text.find("[")) if i >= 0]
    if starts:
        i = min(starts)
        j = text.rfind("}" if text[i] == "{" else "]")
        if j > i:
            return json.loads(text[i:j + 1])
    raise ValueError(f"no JSON object found in {text[:200]!r}")


class Budget:
    """Global cumulative OpenRouter spend across all lanes, persisted after EVERY
    recorded call. `warn_usd` sets .warned; `abort_usd` is a hard clean stop with
    partials kept; `initial_usd` seeds the total so the rails are cumulative."""

    def __init__(self, abort_usd: float, state_path: str, *,
                 warn_usd: float = 0.0, initial_usd: float = 0.0):
        self.abort_usd = float(abort_usd)
        self.warn_usd = float(warn_usd or 0.0)
        self.state_path = state_path
        self.lock = threading.Lock()
        self.aborted = False
        self.warned = False
        self.state = {"total_usd": float(initial_usd or 0.0), "lanes": {},
                      "n_calls": 0, "n_empty_200": 0, "started": time.time(),
                      "initial_usd": float(initial_usd or 0.0)}
        if os.path.exists(state_path):          # resume-safe (same run dir)
            try:
                with open(state_path) as fh:
                    prev = json.load(fh)
                self.state.update({k: prev[k] for k in
                                   ("total_usd", "lanes", "n_calls", "n_empty_200")
                                   if k in prev})
            except Exception:
                pass
        if self.warn_usd and self.state["total_usd"] >= self.warn_usd:
            self.warned = True

    def record(self, lane_id: str, model: str, usage: dict, estimated: bool) -> None:
        with self.lock:
            cost = float(usage.get("cost") or 0.0)
            self.state["total_usd"] += cost
            self.state["n_calls"] += 1
            ls = self.state["lanes"].setdefault(lane_id, {
                "model": model, "usd": 0.0, "in_tok": 0, "out_tok": 0,
                "n": 0, "estimated_calls": 0})
            ls["usd"] += cost
            ls["in_tok"] += int(usage.get("prompt_tokens") or 0)
            ls["out_tok"] += int(usage.get("completion_tokens") or 0)
            ls["n"] += 1
            if estimated:
                ls["estimated_calls"] += 1
            self._persist()
            if self.warn_usd and not self.warned and \
                    self.state["total_usd"] >= self.warn_usd:
                self.warned = True
            if self.state["total_usd"] >= self.abort_usd:
                self.aborted = True

    def record_empty(self) -> None:
        with self.lock:
            self.state["n_empty_200"] += 1
            self._persist()

    def _persist(self) -> None:
        tmp = self.state_path + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(self.state, fh, indent=1)
        os.replace(tmp, self.state_path)

    def check(self) -> None:
        if self.aborted:
            raise BudgetAbort(
                f"cumulative spend ${self.state['total_usd']:.2f} >= "
                f"${self.abort_usd:.2f} — clean stop, partials kept")

    @property
    def total(self) -> float:
        return self.state["total_usd"]


class NullBudget:
    """No rails — for stages whose spend is metered elsewhere, and for tests."""

    warned = False

    def check(self):
        return None

    def record(self, *a, **k):
        return None

    def record_empty(self):
        return None

    @property
    def total(self):
        return 0.0


class Lane:
    """One model lane: async client, concurrency semaphore, retries and usage accounting.
    The API key may come from a secrets dict or from the environment."""

    def __init__(self, lane_id: str, cfg: dict, budget, log_fn,
                 api_key: str | None = None, api_key_env: str = "OPENROUTER_API_KEY"):
        from openai import AsyncOpenAI
        key = api_key or os.environ.get(api_key_env)
        if not key:
            raise RuntimeError(f"no API key ({api_key_env} unset and none supplied)")
        self.id = lane_id
        self.model = cfg["model"]
        self.temperature = float(cfg.get("temperature", 0.2))
        self.max_tokens = min(int(cfg.get("max_tokens", 4096)), MAX_TOKENS_CEILING)
        self.sem = asyncio.Semaphore(int(cfg.get("concurrency", 4)))
        self.retries = int(cfg.get("retries", REQ_RETRIES))
        self.json_retries = int(cfg.get("json_retries", 2))
        self.reasoning_effort = cfg.get("reasoning_effort")
        self.budget = budget
        self.log = log_fn
        self.client = AsyncOpenAI(base_url=os.environ.get("LLM_BASE_URL") or "https://openrouter.ai/api/v1", api_key=key)

    async def _call_once(self, messages, temperature, max_tokens, json_mode, timeout):
        eb = {"usage": {"include": True}}
        if self.reasoning_effort:
            eb["reasoning"] = {"effort": self.reasoning_effort}
        kw = dict(model=self.model, messages=messages, temperature=temperature,
                  max_tokens=max_tokens, timeout=timeout, extra_body=eb)
        if json_mode:
            kw["response_format"] = {"type": "json_object"}
        return await self.client.chat.completions.create(**kw)

    async def generate(self, prompt, *, system: str | None = None,
                       temperature: float | None = None, max_tokens: int | None = None,
                       json_mode: bool = True, timeout: float = 300.0) -> str:
        self.budget.check()
        if isinstance(prompt, list):
            messages = prompt
        else:
            messages = ([{"role": "system", "content": system}] if system else []) + \
                       [{"role": "user", "content": prompt}]
        temperature = self.temperature if temperature is None else temperature
        max_tokens = self.max_tokens if max_tokens is None else max_tokens
        last = None
        async with self.sem:
            for attempt in range(self.retries):
                try:
                    resp = await self._call_once(messages, temperature, max_tokens,
                                                 json_mode, timeout)
                except Exception as exc:
                    last = f"{type(exc).__name__}: {exc}"
                    self.log(f"LLM_ERR lane={self.id} attempt={attempt} {last}")
                    await asyncio.sleep(min(2 ** attempt * 2.0, 30.0))
                    continue
                u = _extract_usage(resp)
                txt = ""
                if resp.choices:
                    txt = (resp.choices[0].message.content or "").strip()
                # empty-200: HTTP 200, empty body, prompt_tokens == 0 -> nothing served,
                # retry is FREE and is never counted as billed.
                if not txt and not u.get("prompt_tokens"):
                    self.budget.record_empty()
                    last = "empty_200"
                    continue
                estimated = u.get("cost") is None
                if estimated:
                    u = dict(u)
                    u["cost"] = ((u.get("prompt_tokens", 0) / 1e6) * FALLBACK_PRICE_IN +
                                 (u.get("completion_tokens", 0) / 1e6) * FALLBACK_PRICE_OUT)
                self.budget.record(self.id, self.model, u, estimated)
                if txt:
                    return txt
                last = "empty_with_usage"
        raise RuntimeError(f"lane {self.id}: {last} after {self.retries} attempts")

    async def generate_json(self, prompt, *, system: str | None = None,
                            temperature: float | None = None,
                            max_tokens: int | None = None, timeout: float = 300.0):
        """Strict-JSON with `json_retries` extra attempts at nudged-down temperature."""
        t = self.temperature if temperature is None else temperature
        last_exc = None
        for attempt in range(1 + self.json_retries):
            txt = await self.generate(prompt, system=system,
                                      temperature=max(t - 0.3 * attempt, 0.0),
                                      max_tokens=max_tokens, timeout=timeout)
            try:
                return parse_json_lenient(txt)
            except Exception as exc:
                last_exc = exc
                self.log(f"LLM_JSONFAIL lane={self.id} attempt={attempt} err={exc}")
        raise RuntimeError(f"lane {self.id}: strict-JSON failed after "
                           f"{1 + self.json_retries} attempts: {last_exc}")


class StubLane:
    """Offline stand-in with the same surface as Lane. Tests and dry runs inject it.

    `responses` is either a callable(prompt) -> dict, or a list consumed in order.
    Every prompt it is handed is recorded in `.prompts` so a test can assert on it.
    """

    def __init__(self, lane_id: str = "stub", responses=None, model: str = "stub"):
        self.id = lane_id
        self.model = model
        self.prompts: list = []
        self._responses = responses

    def _next(self, prompt):
        self.prompts.append(prompt)
        if callable(self._responses):
            return self._responses(prompt)
        if isinstance(self._responses, list) and self._responses:
            return self._responses.pop(0)
        return {}

    async def generate_json(self, prompt, **_kw):
        return self._next(prompt)

    async def generate(self, prompt, **_kw) -> str:
        return json.dumps(self._next(prompt))


def build_lanes(cfg, budget, log, *, secrets: dict | None = None) -> dict:
    """Construct the lanes a stage needs from `models:` in the config.

    Every lane uses the one OpenAI-compatible endpoint. Called only when a stage is about to make real calls,
    so importing a stage script never opens a client."""
    secrets = secrets or {}
    key = secrets.get("OPENROUTER_API_KEY")
    lanes: dict = {}
    for lane_id, lane_cfg in (cfg.get("models") or {}).items():
        if not isinstance(lane_cfg, dict) or "model" not in lane_cfg:
            continue
        lanes[lane_id] = Lane(lane_id, lane_cfg, budget, log,
                              api_key=key)
    return lanes
