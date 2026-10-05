"""Guards the CLAUDE.md hard rule that Chainlit's MCP feature stays off (it can run local executables)."""

import tomllib

from buddy.config import PROJECT_ROOT

CONFIG = tomllib.loads((PROJECT_ROOT / ".chainlit" / "config.toml").read_text(encoding="utf-8"))
MCP = CONFIG["features"]["mcp"]


def test_mcp_is_disabled():
    assert MCP["enabled"] is False


def test_no_mcp_servers_are_configured():
    assert MCP.get("servers", []) == []


def test_users_cannot_add_mcp_servers():
    assert MCP["user_servers"]["enabled"] is False


def test_no_pre_2_12_mcp_sections_remain():
    # Chainlit 2.12 ignores these with a startup warning; they must not creep back in.
    assert not {"sse", "stdio", "streamable-http"} & MCP.keys()
