"""Fill a prompt; make one model call, on either backend. Both return (reply text, raw response, session id, cost in USD).

  claude_cli   the headless Claude CLI (claude -p). The system prompt is replaced by ours and
               every tool, setting, plugin, MCP server and skill of this machine is switched off, so the
               model sees only our prompt. The conversation lives in a CLI session; each call sends one
               new message. `cost` is the CLI's own API-price estimate, not a bill, and it is the
               running total of the whole session, not of the one call.
  openrouter   any other model; the whole conversation is sent on every call.

"""
import json
import os
import re
import subprocess

import requests

OPENROUTER = "https://openrouter.ai/api/v1/chat/completions"


def fill(template, values):
    """Replace every {name} placeholder of the template in one pass (inserted text is never rescanned).
    Not str.format: the prompt holds literal JSON braces."""
    def one(m):
        if m.group(1) not in values:
            raise SystemExit(f"no value for placeholder {m.group(0)}")
        return str(values[m.group(1)])
    return re.sub(r"\{([a-z_]+)\}", one, template)


def parse(text):
    """The reply's JSON object, or None. Text around the object (a code fence) is tolerated."""
    a, b = text.find("{"), text.rfind("}")
    try:
        obj = json.loads(text[a:b + 1])
    except ValueError:
        return None
    return obj if isinstance(obj, dict) else None


def check_claude_cwd(cwd):
    """A CLAUDE.md in the working folder or any parent would be loaded into the model's context."""
    os.makedirs(cwd, exist_ok=True)
    probe = os.path.realpath(cwd)
    while True:
        if os.path.exists(os.path.join(probe, "CLAUDE.md")):
            raise SystemExit(f"{probe}/CLAUDE.md would leak into the prompt; move AGENT_EVAL_CLAUDE_CWD")
        if probe == os.path.dirname(probe):
            return
        probe = os.path.dirname(probe)


def call_claude(model, messages, sid, cwd):
    """Send the LAST message; earlier turns live in session `sid` (None starts a session). The reply
    comes back in a new session that copies `sid` (--fork-session), so `sid` itself never changes and
    a call that failed or is repeated leaves no trace in the conversation. The message goes via
    stdin because a results message can exceed the OS limit on one argument."""
    cmd = ["claude", "-p", "--model", model, "--system-prompt", messages[0]["content"],
           "--exclude-dynamic-system-prompt-sections", "--disallowedTools", "*",
           "--setting-sources=", "--strict-mcp-config", "--disable-slash-commands",
           "--output-format", "json"]
    if sid:
        cmd += ["--resume", sid, "--fork-session"]
    r = subprocess.run(cmd, input=messages[-1]["content"], capture_output=True, text=True, cwd=cwd, timeout=1200)
    if r.returncode != 0:
        raise RuntimeError(f"claude rc={r.returncode} {r.stderr[-200:]} {r.stdout[-200:]}")
    data = json.loads(r.stdout)
    if data.get("is_error"):  # e.g. a usage limit: the "result" is then an error text
        raise RuntimeError(f"claude error: {str(data.get('result'))[:200]}")
    return data.get("result") or "", data, data.get("session_id"), float(data.get("total_cost_usd") or 0.0)


def call_openrouter(model, messages, key):
    body = {"model": model, "messages": messages, "max_tokens": 16000,
            "usage": {"include": True}, "provider": {"allow_fallbacks": True}}
    r = requests.post(OPENROUTER, json=body, timeout=600, headers={"Authorization": f"Bearer {key}"})
    data = r.json()
    text = (data.get("choices") or [{}])[0].get("message", {}).get("content") or ""
    return text, data, None, float((data.get("usage") or {}).get("cost") or 0.0)

