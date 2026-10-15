#!/usr/bin/env python3
"""Claude Code hook: keep Supabase keys out of tool output.

pre  (PreToolUse, Bash): block commands that could print a secret.
post (PostToolUse, any): if a key value appears in tool output anyway, alert the user.

Exit code 2 blocks the tool call; stderr is shown to Claude as the reason.
"""

import json
import re
import sys

# Real key files, but not their committed .example templates.
SECRET_FILES = re.compile(r"(?<![\w.])\.env(?!\.example)(?![\w.])|secrets\.toml(?!\.example)")
DANGEROUS = [
    (
        SECRET_FILES,
        "references .env or secrets.toml; read keys only inside scripts that never print them",
    ),
    (re.compile(r"--reveal\b"), "reveals API keys"),
    (
        re.compile(r"(^|[|;&(]\s*)(printenv|env|set|export -p|declare -x)\s*($|[|;&)])"),
        "dumps the environment",
    ),
    (re.compile(r"\$\{?SUPABASE_(SECRET|PUBLISHABLE)_KEY"), "expands a key variable in the shell"),
    (
        re.compile(
            r"environ[^\n]*(SECRET|PUBLISHABLE)|dotenv_values|getenv\([^)]*(SECRET|PUBLISHABLE)"
        ),
        "reads a key in inline code that may print it",
    ),
    (re.compile(r"find-generic-password[^\n]*\s-w\b"), "prints a keychain secret"),
]
# Key values: Supabase new-style keys and JWTs (legacy keys, access tokens).
KEY_VALUE = re.compile(
    r"sb_secret_[A-Za-z0-9_-]{6,}|sb_publishable_[A-Za-z0-9_-]{6,}|sbp_[A-Za-z0-9]{20,}|eyJ[A-Za-z0-9_-]{20,}\.eyJ"
)


def pre(event: dict) -> int:
    command = event.get("tool_input", {}).get("command", "")
    for pattern, why in DANGEROUS:
        if pattern.search(command):
            print(f"secret_guard blocked this command: it {why}.", file=sys.stderr)
            return 2
    return 0


USER_ALERT = "secret_guard: a Supabase key or token appeared in tool output. Rotate it."
MODEL_NOTE = (
    "A secret value appeared in the last tool output. Do not repeat it; tell the user to rotate it."
)


def post(event: dict) -> int:
    if KEY_VALUE.search(json.dumps(event.get("tool_response", ""))):
        output = {
            "systemMessage": USER_ALERT,
            "hookSpecificOutput": {"hookEventName": "PostToolUse", "additionalContext": MODEL_NOTE},
        }
        print(json.dumps(output))
    return 0


if __name__ == "__main__":
    sys.exit({"pre": pre, "post": post}[sys.argv[1]](json.load(sys.stdin)))
