"""Regression tests for the experimental tools/context_tree.py prototype."""
import importlib.util
import json
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location("context_tree", Path(__file__).resolve().parents[1] / "tools" / "context_tree.py")
ct = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(ct)

BODY = "\n".join(f"def function_{i}_alpha(): return value_{i}_omega" for i in range(10))


def _assistant(mid, *blocks, usage=None):
    return {"type": "assistant", "message": {"role": "assistant", "id": mid, "content": list(blocks), **({"usage": usage} if usage else {})}}


def _use(tid, name="Read", **inp):
    return {"type": "tool_use", "id": tid, "name": name, "input": inp}


def _result(tid, text):
    return {"type": "user", "message": {"role": "user", "content": [{"type": "tool_result", "tool_use_id": tid, "content": text}]}}


def _user(text, **extra):
    return {"type": "user", "message": {"role": "user", "content": text}, **extra}


def _write(tmp_path, lines, name="s.jsonl"):
    p = tmp_path / name
    p.write_text("\n".join(json.dumps(x) for x in lines))
    return p


def _leaf(result, kind):
    return [l for l in result["leaves"] if l["kind"] == kind]


def test_reused_line_is_live_and_unused_lines_are_dead(tmp_path):
    p = _write(tmp_path, [
        _user("please inspect the module"),
        _assistant("m1", _use("t1", file_path="mod.py")),
        _result("t1", BODY),
        _assistant("m2", {"type": "text", "text": "function_3_alpha looks wrong"}),
        _assistant("m3", {"type": "text", "text": "done"}),
    ])
    r = ct.analyse(ct.load_claude_code(p))
    leaf = _leaf(r, "tool:Read")[0]
    assert r["calls"] == 3 and leaf["window"] == 2
    assert 0 < leaf["live_chars"] < leaf["cell_chars"]
    assert leaf["carry"] == leaf["chars"] * leaf["window"]
    assert leaf["dead_carry"] + leaf["stale_carry"] <= leaf["carry"]


def test_compaction_ends_carry_counts_once_and_summary_is_not_a_prompt(tmp_path):
    p = _write(tmp_path, [
        _user("please inspect the module"),
        _assistant("m1", _use("t1", file_path="mod.py")),
        _result("t1", BODY),
        _assistant("m2", {"type": "text", "text": "ok"}),
        {"type": "system", "subtype": "compact_boundary"},
        _user("Summary: we looked at function_1_alpha and function_2_alpha " * 6, isCompactSummary=True),
        *[_assistant(f"n{i}", {"type": "text", "text": "continuing"}) for i in range(6)],
    ])
    r = ct.analyse(ct.load_claude_code(p))
    leaf = _leaf(r, "tool:Read")[0]
    assert r["scars"] == 1
    assert leaf["window"] == 1  # left context at the compaction, not at the session end
    assert leaf["live_chars"] == 0  # the summary repeating its tokens does not make it live
    assert r["prompts"] == 1


def test_reread_content_still_counts_as_carry(tmp_path):
    p = _write(tmp_path, [
        _user("read it twice"),
        _assistant("m1", _use("t1", file_path="mod.py")),
        _result("t1", BODY),
        _assistant("m2", _use("t2", file_path="mod.py")),
        _result("t2", BODY),
        _assistant("m3", {"type": "text", "text": "done"}),
    ])
    first, second = _leaf(ct.analyse(ct.load_claude_code(p)), "tool:Read")
    assert second["cells"] == 0 and second["carry"] == second["chars"] * second["window"] > 0
    assert second["untracked_carry"] == second["carry"]


def test_harness_messages_are_not_user_prompts_and_trailing_prompt_does_not_revive(tmp_path):
    p = _write(tmp_path, [
        _user("please inspect the module"),
        _assistant("m1", _use("t1", file_path="mod.py")),
        _result("t1", BODY),
        _user("<task-notification> function_5_alpha finished " * 10),
        _user("injected skill text mentioning function_6_alpha " * 10, isMeta=True),
        _assistant("m2", {"type": "text", "text": "done"}),
        _user("what about function_7_alpha"),  # no model call ever reads this
    ])
    r = ct.analyse(ct.load_claude_code(p))
    assert r["prompts"] == 2 and len(_leaf(r, "injected")) == 2
    assert _leaf(r, "tool:Read")[0]["live_chars"] == 0


def test_rings_do_not_double_count_and_ignore_zero_usage(tmp_path):
    u = lambda n: {"input_tokens": n, "output_tokens": 5}
    p = _write(tmp_path, [
        _user("go"),
        _assistant("m1", {"type": "text", "text": "a"}, usage=u(1000)),
        _user("x" * 4000),
        _assistant("m2", {"type": "text", "text": "b"}),
        _assistant("m3", {"type": "text", "text": "c"}, usage=u(3000)),
        _assistant("m4", {"type": "text", "text": "d"}, usage=u(0)),
    ])
    r = ct.analyse(ct.load_claude_code(p))
    assert r["soil_tokens"] == 1000
    assert sum(r["rings"]) == 2000


def test_cli_outputs_carry_no_message_text_and_survive_bad_files(tmp_path):
    proj = tmp_path / "projects" / "secret-repo-name"
    proj.mkdir(parents=True)
    _write(proj, [
        _user("my private question about zebra_crossing_token"),
        _assistant("m1", _use("t1", file_path="/home/me/secret_path_token.py")),
        _result("t1", BODY),
        _assistant("m2", {"type": "text", "text": "function_3_alpha"}),
    ], name="0a1b2c3d-session.jsonl")
    (tmp_path / "projects" / "conversations.json").write_text("")  # malformed export must not abort the run
    (tmp_path / "projects" / "conversations-000.json").write_text(json.dumps([{"title": "t", "mapping": {}}]))
    out = tmp_path / "out"
    assert ct.main([str(tmp_path / "projects"), "--out", str(out)]) == 0
    blob = "".join(f.read_text() for f in out.iterdir())
    for secret in ("zebra_crossing_token", "secret_path_token", "secret-repo-name", "0a1b2c3d-session", "function_3_alpha"):
        assert secret not in blob
    summary = json.loads((out / "forest_summary.json").read_text())
    assert summary["trees"] == 1 and summary["caveats"]


def test_claude_ai_export_uses_result_name_and_is_flagged_estimated(tmp_path):
    export = [{"uuid": "c1", "chat_messages": [
        {"sender": "human", "text": "look this up", "content": [{"type": "text", "text": "look this up"}]},
        {"sender": "assistant", "content": [
            {"type": "tool_use", "name": "web_search", "input": {"q": "x"}},
            {"type": "tool_result", "name": "web_search", "content": [{"type": "knowledge", "title": "T", "url": "https://e.example/a", "text": BODY}]},
            {"type": "text", "text": "function_4_alpha is the answer"},
        ]},
    ]}]
    p = tmp_path / "conversations.json"
    p.write_text(json.dumps(export))
    (events,) = ct.load_claude_ai_export(p).values()
    r = ct.analyse(events)
    assert r["tokens_estimated"] and r["soil_tokens"] is None
    assert "web_search" in r["roots"] and _leaf(r, "tool:web_search")[0]["live_chars"] > 0
