import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
PLUGIN = ROOT / "plugins" / "tracerazor"


def test_codex_plugin_manifest_and_components_are_consistent():
    manifest = json.loads((PLUGIN / ".codex-plugin" / "plugin.json").read_text())
    assert manifest["name"] == "tracerazor"
    assert manifest["version"] == "1.1.0"
    assert manifest["skills"] == "./skills/"
    assert "mcpServers" not in manifest
    assert (PLUGIN / "skills" / "tracerazor" / "SKILL.md").is_file()
    assert (PLUGIN / "hooks" / "hooks.json").is_file()


def test_codex_plugin_mcp_is_local_stdio_only():
    config = json.loads((PLUGIN / ".mcp.json").read_text())
    server = config["tracerazor"]
    assert server == {"command": "tracerazor-mcp", "args": []}


def test_codex_hooks_use_non_mutating_agent_lifecycle_entrypoint():
    config = json.loads((PLUGIN / "hooks" / "hooks.json").read_text())
    hooks = config["hooks"]
    assert set(hooks) == {"SessionStart", "SubagentStart", "SubagentStop", "Stop"}
    commands = [
        handler["command"]
        for event in hooks.values()
        for group in event
        for handler in group["hooks"]
    ]
    assert all(command.startswith("tracerazor agent hook --host codex --event ") for command in commands)
    assert not any("apply" in command or "enforce" in command for command in commands)


def test_claude_plugin_is_versioned_and_advisory():
    plugin = ROOT / "extensions" / "claude-code" / "tracerazor"
    manifest = json.loads((plugin / ".claude-plugin" / "plugin.json").read_text())
    hooks = json.loads((plugin / "hooks" / "hooks.json").read_text())["hooks"]
    mcp = json.loads((plugin / ".mcp.json").read_text())
    assert manifest["name"] == "tracerazor"
    assert manifest["version"] == "1.1.0"
    assert "SessionStart" in hooks and "SessionEnd" in hooks
    assert mcp["mcpServers"]["tracerazor"]["command"] == "tracerazor-mcp"


def test_gemini_extension_uses_json_hook_contract_and_local_mcp():
    extension = ROOT / "extensions" / "gemini-cli" / "tracerazor"
    manifest = json.loads((extension / "gemini-extension.json").read_text())
    hooks = json.loads((extension / "hooks" / "hooks.json").read_text())["hooks"]
    assert manifest["name"] == "tracerazor"
    assert manifest["version"] == "1.1.0"
    assert manifest["mcpServers"]["tracerazor"]["command"] == "tracerazor-mcp"
    commands = [
        handler["command"]
        for event in hooks.values()
        for group in event
        for handler in group["hooks"]
    ]
    assert all("--host gemini" in command for command in commands)
    for groups in hooks.values():
        assert all("matcher" not in group for group in groups)
