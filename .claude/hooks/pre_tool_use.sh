#!/usr/bin/env bash
# PreToolUse guardrail for shell tools (Bash + PowerShell).
# Claude Code passes the tool call as JSON on stdin; exit 2 blocks the call and
# shows stderr to the agent. Rules come from CLAUDE.md "Hard rules" + test.expectations #6.
tool_call="$(cat)"

# Decode the real command string so JSON escapes (\n, \") can't hide a command from the checks.
# If python is unavailable, fall back to the raw JSON rather than letting everything through.
command="$(printf '%s' "$tool_call" | python -c 'import json,sys; print(json.load(sys.stdin).get("tool_input", {}).get("command", ""))' 2>/dev/null)" \
  || command="$tool_call"

if echo "$command" | grep -qiE 'Co-authored-by:.*(claude|anthropic|\[bot\])'; then
  echo "Blocked: AI co-author trailer not allowed in commits (CLAUDE.md test.expectations #6)." >&2
  exit 2
fi

# npm as a standalone word, incl. npm.cmd / npm.exe. "-", "." and "_" count as part of a word so
# "pnpm" and "npm-check" pass; a trailing "/" means a directory path (e.g. AppData/Roaming/npm/).
if echo "$command" | grep -qE '(^|[^[:alnum:]_.-])npm(\.cmd|\.exe)?([^[:alnum:]_./-]|$)'; then
  echo "Blocked: npm is banned on this machine (virus confirmed 2026-05-25). Use pnpm." >&2
  exit 2
fi

exit 0
