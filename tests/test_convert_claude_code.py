"""Unit tests for the Claude Code transcript -> TraceRazor trace converter."""

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from benchmark.convert_claude_code import convert  # noqa: E402


def _write_transcript(tmp_path: Path, entries: list[dict]) -> Path:
    p = tmp_path / "session-abc.jsonl"
    p.write_text(
        "\n".join(json.dumps(e) for e in entries) + "\n", encoding="utf-8"
    )
    return p


def _assistant_line(
    mid: str,
    block: dict,
    usage: dict | None = None,
    sidechain: bool = False,
) -> dict:
    return {
        "type": "assistant",
        "isSidechain": sidechain,
        "message": {
            "id": mid,
            "role": "assistant",
            "model": "claude-haiku-4-5-20251001",
            "content": [block],
            "usage": usage
            or {
                "input_tokens": 10,
                "cache_creation_input_tokens": 90,
                "cache_read_input_tokens": 500,
                "output_tokens": 100,
            },
        },
    }


def _user_prompt(text: str) -> dict:
    return {"type": "user", "message": {"role": "user", "content": text}}


def _tool_result(tool_use_id: str, text: str, is_error: bool = False) -> dict:
    return {
        "type": "user",
        "message": {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": tool_use_id,
                    "content": [{"type": "text", "text": text}],
                    "is_error": is_error,
                }
            ],
        },
    }


def _basic_entries() -> list[dict]:
    """One prompt, one message split over 3 lines (text + 2 tool calls)."""
    return [
        _user_prompt("Fix the failing test"),
        _assistant_line("msg_1", {"type": "text", "text": "Let me look around."}),
        _assistant_line(
            "msg_1",
            {"type": "tool_use", "id": "tu_1", "name": "Read",
             "input": {"file_path": "a.py"}},
        ),
        _assistant_line(
            "msg_1",
            {"type": "tool_use", "id": "tu_2", "name": "Bash",
             "input": {"command": "python3 -m pytest -q"}},
        ),
        _tool_result("tu_1", "contents of a.py"),
        _tool_result("tu_2", "1 failed", is_error=True),
        _assistant_line(
            "msg_2",
            {"type": "text", "text": "Found it; the loop bound is wrong."},
            usage={
                "input_tokens": 5,
                "cache_creation_input_tokens": 15,
                "cache_read_input_tokens": 800,
                "output_tokens": 30,
            },
        ),
    ]


def test_usage_counted_once_per_message(tmp_path):
    """Repeated usage objects on split assistant lines must not double-count."""
    trace = convert(_write_transcript(tmp_path, _basic_entries()))
    # msg_1 is the first turn: its cache_creation (the harness-prefix
    # encoding, 90) is excluded -> 10 + 100 = 110.  msg_2 = 5 + 15 + 30 = 50.
    assert trace["total_tokens"] == 160
    msg1_steps = trace["steps"][:3]
    assert sum(s["tokens"] for s in msg1_steps) == 110


def test_gross_accounting_includes_everything(tmp_path):
    trace = convert(
        _write_transcript(tmp_path, _basic_entries()), include_cache_read=True
    )
    # Gross keeps cache reads AND the first-turn prefix encoding.
    assert trace["total_tokens"] == 250 + 500 + 800
    assert "gross" in trace["metadata"]["token_accounting"]


def test_first_turn_prefix_encoding_excluded_only_once(tmp_path):
    entries = [
        _user_prompt("go"),
        _assistant_line("m1", {"type": "text", "text": "first"}),
        _assistant_line("m2", {"type": "text", "text": "second"}),
    ]
    trace = convert(_write_transcript(tmp_path, entries))
    s1, s2 = trace["steps"]
    assert s1["tokens"] == 10 + 100  # creation (90) excluded on turn 1
    assert s2["tokens"] == 10 + 90 + 100  # counted from turn 2 on


def test_step_structure_and_tool_results(tmp_path):
    trace = convert(_write_transcript(tmp_path, _basic_entries()))
    steps = trace["steps"]
    assert [s["type"] for s in steps] == [
        "reasoning", "tool_call", "tool_call", "reasoning",
    ]
    read, bash = steps[1], steps[2]
    assert read["tool_name"] == "Read"
    assert read["tool_success"] is True
    assert read["output"] == "contents of a.py"
    assert bash["tool_success"] is False
    assert bash["tool_error"] == "1 failed"
    assert "output" not in bash


def test_input_context_only_on_first_step_of_message(tmp_path):
    trace = convert(_write_transcript(tmp_path, _basic_entries()))
    steps = trace["steps"]
    assert "Fix the failing test" in steps[0]["input_context"]
    assert "input_context" not in steps[1]
    assert "input_context" not in steps[2]
    # msg_2's context is the new tool results fed back in between.
    assert "contents of a.py" in steps[3]["input_context"]
    assert "1 failed" in steps[3]["input_context"]


def test_task_metadata_and_value(tmp_path):
    p = _write_transcript(tmp_path, _basic_entries())
    trace = convert(p, task="Custom goal", task_value=0.0)
    assert trace["metadata"]["task"] == "Custom goal"
    assert trace["task_value_score"] == 0.0
    # Default goal falls back to the first user prompt.
    trace2 = convert(p)
    assert trace2["metadata"]["task"] == "Fix the failing test"
    assert "task_value_score" not in trace2


def test_sidechain_steps_get_agent_id(tmp_path):
    entries = _basic_entries() + [
        _assistant_line(
            "msg_3", {"type": "text", "text": "subagent says hi"}, sidechain=True
        )
    ]
    trace = convert(_write_transcript(tmp_path, entries))
    assert trace["steps"][-1]["agent_id"] == "subagent"
    assert all("agent_id" not in s for s in trace["steps"][:-1])


def test_thinking_blocks_become_reasoning(tmp_path):
    entries = [
        _user_prompt("think hard"),
        _assistant_line("m1", {"type": "thinking", "thinking": "private plan"}),
        _assistant_line("m1", {"type": "text", "text": "public answer"}),
    ]
    trace = convert(_write_transcript(tmp_path, entries))
    [step] = trace["steps"]
    assert step["type"] == "reasoning"
    assert "private plan" in step["content"]
    assert "public answer" in step["content"]


def test_ids_are_sequential_from_one(tmp_path):
    trace = convert(_write_transcript(tmp_path, _basic_entries()))
    assert [s["id"] for s in trace["steps"]] == list(
        range(1, len(trace["steps"]) + 1)
    )
